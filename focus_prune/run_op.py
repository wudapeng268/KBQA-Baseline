# coding:utf-8
# author:wup
# description: train dev test code in focus_prune, we will get output in fp_output directory
# e-mail:wup@nlp.nju.cn
# date:2018.4.6

import time
import tensorflow as tf
import os
import numpy as np
from util import FileUtil
import pdb


def run_dev(model, tf_transition_params_test):
    step = 1
    losses = []
    model.qa.batch_index_dev = 0
    ans = []
    all_corr = 0
    all_label = 0
    tp = 0
    fn = 0
    fp = 0

    while (step - 1) * model.batch_size < model.deving_iters:
        x_lens, batch_x, batch_y = model.qa.load_data(model.batch_size, "dev")

        batch_loss, tf_unary_scores = model.sess.run([model.loss, model.unary_scores],
                                                     feed_dict={model.question_ids: batch_x, model.y: batch_y,
                                                                model.x_lens: x_lens})
        losses.append(batch_loss)

        index = 0
        for tf_unary_scores_, batch_y_, sequence_lengths_t_ in zip(tf_unary_scores, batch_y,
                                                                   x_lens):

            tf_unary_scores_ = tf_unary_scores_[:sequence_lengths_t_]

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params_test)
            for i, p in enumerate(viterbi_sequence):
                if p == 1 and batch_y_[i] == 1:
                    tp += 1
                elif p == 0 and batch_y_[i] == 1:
                    fn += 1
                elif p == 1 and batch_y_[i] == 0:
                    fp += 1

            all_corr += np.sum(np.equal(viterbi_sequence, batch_y_[:sequence_lengths_t_]))
            all_label += sequence_lengths_t_
            index += 1
            ans.append(viterbi_sequence)
        step += 1
    pred = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f1 = 2 * pred * recall / (pred + recall)
    return f1


def new_save_model(model, current_epoch, tf_transition_params_test):
    mm = time.time()
    fold_path = model.model_path
    if not os.path.isdir(fold_path):
        os.system("mkdir -p %s" % fold_path)
    model_path = os.path.join(fold_path, "fp_model.ckpt")
    np.savetxt("{}/transition_martix-{}".format(fold_path, current_epoch), tf_transition_params_test)

    save_path = model.saver.save(model.sess, model_path, current_epoch)
    print("Model saved in file: %s" % save_path)
    mm2 = time.time()
    print("Save model time: %f" % (mm2 - mm))


def train(model, config):
    init = tf.global_variables_initializer()
    model.sess.run(init)

    current_epoch = 0
    model.global_step = 0
    model_path = config.model_path
    if config.continue_train:
        model.saver.restore(model.sess, tf.train.latest_checkpoint(model_path))
        restore_filename = tf.train.latest_checkpoint(model_path)
        current_epoch = int(restore_filename.split("-")[-1]) + 1
        model.global_step = current_epoch * (model.training_iters / model.fix_batch_size)
        tf_transition_params = np.loadtxt(
            "{}/transition_martix-{}".format(model_path, restore_filename.split("-")[-1]))

    best_pred = 0  # max value

    while current_epoch <= config.epoch:
        step = 1
        model.qa.batch_index_train = 0
        while (step - 1) * config.batch_size < model.training_iters:
            model.global_step += 1
            ss = time.time()
            # pdb.set_trace()
            x_lens, batch_x, batch_y = model.qa.load_data(config.batch_size)

            tf_unary_scores, tf_transition_params, _ = model.sess.run(
                [model.unary_scores, model.transition_params, model.optimizer],
                feed_dict={model.question_ids: batch_x, model.y: batch_y,
                           model.x_lens: x_lens})
            if (step - 1) % config.display_step == 0:
                print("epoch:\t%d,rate:\t%d/%d,time:\t%f" % (
                    current_epoch, step, (model.training_iters // config.batch_size), time.time() - ss))
            step += 1

        if current_epoch % config.dev_epoch == 0:
            new_pred = run_dev(model, tf_transition_params)
            print("last pred:\t%f,current pred:\t%f" % (best_pred, new_pred))
            if new_pred > best_pred:
                new_save_model(model, current_epoch, tf_transition_params)
                print("save epoch: %d" % current_epoch)
                best_pred = new_pred

        current_epoch += 1
    print("Finish train!")
    print("Start test!")
    test(model, config)


def label2entity(question, label):
    question = question.split(" ")
    label = [int(l) for l in label]
    entityList = []
    entity = ""
    for i in range(len(label)):
        if label[i] == 0:
            if entity != "":
                entity = entity[:-1]
                entityList.append(entity)
                entity = ""
            continue
        else:
            entity += question[i] + " "
    if entity != "":
        entity = entity[:-1]
        entityList.append(entity)
        entity = ""
    return entityList


def test(model, config):
    model_path = config.model_path
    model.saver.restore(model.sess, tf.train.latest_checkpoint(model_path))
    restore_filename = tf.train.latest_checkpoint(model_path)
    print("restore filename: %s" % restore_filename)
    tf_transition_params_test = np.loadtxt(
        "{}/transition_martix-{}".format(model_path, restore_filename.split("-")[-1]))
    print("type of tf_transition_params_test:\t", type(tf_transition_params_test))

    test_save(model, config, tf_transition_params_test, "test")
    test_save(model, config, tf_transition_params_test, "dev")


def test_save(model, config, tf_transition_params_test, dataset):
    step = 1
    ans = []
    t = 0
    if dataset == "test":
        iters = model.testing_iters
    else:
        iters = model.deving_iters
    tp = 0
    fn = 0
    fp = 0
    model.qa.batch_index_dev = 0
    model.qa.batch_index_test = 0
    while (step - 1) * config.batch_size < iters:
        x_lens, test_x, y = model.qa.load_data(config.batch_size, dataset)
        t += len(x_lens)
        tf_unary_scores = model.sess.run(model.unary_scores,
                                         feed_dict={model.question_ids: test_x,
                                                    model.x_lens: x_lens})

        for tf_unary_scores_, y_, sequence_lengths_t_ in zip(tf_unary_scores, y,
                                                             x_lens):
            tf_unary_scores_ = tf_unary_scores_[:sequence_lengths_t_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params_test)
            ans.append(viterbi_sequence)
            for i, p in enumerate(viterbi_sequence):
                if p == 1 and y_[i] == 1:
                    tp += 1
                elif p == 0 and y_[i] == 1:
                    fn += 1
                elif p == 1 and y_[i] == 0:
                    fp += 1
        step += 1

    pred = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f1 = 2 * pred * recall / (pred + recall)
    print("{} pred: {} recall: {} f1: {}".format(dataset, pred, recall, f1))
    questions = model.qa.get_questions(dataset)
    print("ans len:\t", t)
    print("question:\t", len(questions))
    output = []
    for i in range(len(questions)):
        output.append("qid: {}".format(i))
        output.append(FileUtil.list2str(label2entity(questions[i], ans[i]), split="\t"))
    FileUtil.writeFile(output, "{}/sq.{}.label".format(config.answer_path, dataset))
