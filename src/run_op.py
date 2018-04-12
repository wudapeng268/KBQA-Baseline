#coding:utf-8
#author:wup
#description: train dev test code when change trainging funcation please change this file
#e-mail:wup@nlp.nju.cn
#date: 2018.4.4

from util import FileUtil
import time
import random
import tensorflow as tf
from src.summary_add import SummaryWriter
import os
import pdb


def dev(model):
    step = 1
    acc = 0
    qid = 0

    while (step - 1) * model.dev_batch_size < model.deving_iters:
        model.dev_global_step += 1
        value = model.qa.load_test_data(
            model.dev_batch_size, "dev")
        feed = {model.question_ids: value['batch_x_anonymous'],
                model.relation_index: value['batch_relation_index'],
                model.x_lens: value['batch_x_anonymous_lens'],
                model.subject_emb: value['batch_subject_emb'],
                model.subject_relation: value['batch_subject_relation'],
                model.is_training: False, }

        loss_relation, loss_subject, dev_loss, predict_relation, predict_subject = model.sess.run(
            [model.loss_relation, model.loss_subject, model.loss_all, model.predict_relation_index,
             model.predict_subject_index],
            feed_dict=feed)

        model.writer.add_summary("dev/loss_subject", loss_subject,
                                 model.dev_global_step * 1.0)
        model.writer.add_summary("dev/loss_relation", loss_relation,
                                 model.dev_global_step * 1.0)
        model.writer.add_summary("dev/all_loss", dev_loss,
                                 model.dev_global_step * 1.0)

        for i in range(value['batch_size']):
            ans_relation_id = predict_relation[i]
            subject_index = predict_subject[i]
            if subject_index >= len(value['batch_subject_id'][i]):
                continue
            ans_subject_id = value['batch_subject_id'][i][subject_index]
            if i >= len(value['questions']):
                continue
            if value['gold_relation'][i] == ans_relation_id and value['gold_subject'][i] == ans_subject_id:
                acc += 1
            qid += 1
        step += 1
    acc = (acc * 1.0 / qid)
    return acc


def train(model, config):
    model.global_step = 0
    model.dev_global_step = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate / (1 + 0.0001 * model.global_step),
                                       name="rel_optimer").minimize(model.loss_all)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    current_epoch = 0
    best_acc = 0
    if config.log_path != None:
        model.writer = SummaryWriter(config.log_path)
    if config.continue_train:
        model.saver.restore(model.sess, tf.train.latest_checkpoint("model"))
        current_epoch = int(tf.train.latest_checkpoint("model").split('-')[-1]) + 1
        model.global_step = current_epoch*(model.training_iters/model.fix_batch_size)
        model.dev_global_step = current_epoch*(model.deving_iters/model.dev_batch_size)
        best_acc = dev(model)
        tf.logging.log(tf.logging.INFO,
                       "epoch:\t%d,best acc in dev:\t%f",
                       current_epoch - 1, best_acc)
    else:
        print("new train!")
        if config.first_dev:
            best_acc = dev(model)
            tf.logging.log(tf.logging.INFO,
                           "epoch:\t%d,best acc in dev:\t%f",
                           current_epoch - 1, best_acc)

    # begin train!
    continue_not_save = 0
    early_stop = False
    while current_epoch <= config.epoch and not early_stop:
        ss = time.time()
        model.qa.itemIndexTrain = 0
        model.qa.itemIndexTest = 0
        model.qa.itemIndexDev = 0
        random.shuffle(model.qa.train_data)
        step = 1
        mm = time.time()
        while (step - 1) * model.fix_batch_size < model.training_iters:
            model.global_step += 1
            value = model.qa.load_train_data(model.fix_batch_size)

            if (step - 1) % model.display_step == 0:
                tf.logging.log(tf.logging.INFO, "epoch:\t%d,rate:\t%d/%d,time:\t%f",
                               current_epoch, step, model.training_iters // model.fix_batch_size, time.time() - mm)
                mm = time.time()

            feed = {model.question_ids: value['batch_x_anonymous'],
                    model.relation_index: value['batch_relation_index'],
                    model.x_lens: value['batch_x_anonymous_lens'],
                    model.subject_emb: value['batch_subject_emb'],
                    model.is_training: True, }

            _, loss_subject, loss_relation, loss_all, l2_loss = model.sess.run(
                [optimizer, model.loss_subject, model.loss_relation, model.loss_all, model.l2_loss],
                feed_dict=feed)

            model.writer.add_summary("train/loss_subject", loss_subject,
                                     model.global_step * 1.0)
            model.writer.add_summary("train/loss_relation", loss_relation,
                                     model.global_step * 1.0)
            model.writer.add_summary("train/l2_loss", l2_loss,
                                     model.global_step * 1.0)
            model.writer.add_summary("train/train_loss", loss_all,
                                     model.global_step * 1.0)
            step += 1
        if current_epoch % config.dev_epoch == 0:
            new_acc = dev(model)
            model.writer.add_summary("dev/acc", new_acc, current_epoch)
            tf.logging.log(tf.logging.INFO,
                           "epoch:\t%d,new_acc_relation:\t%f",
                           current_epoch, new_acc)
            if new_acc > best_acc:
                continue_not_save = 0
                tf.logging.log(tf.logging.INFO, "save model:\t%d", current_epoch)
                save_model(model, current_epoch)
                best_acc = new_acc
            else:
                continue_not_save += 1
        if current_epoch % (config.dev_epoch * 2) == 0:
            test(model, config, False)
        if continue_not_save == 10:
            tf.logging.log(tf.logging.INFO,
                           "early stop! now epoch: %d, glodle step: %d" % (current_epoch, model.global_step))
            break

        ee = time.time()
        print("epoch:\t%d,time:\t%f" % (current_epoch, ee - ss))
        current_epoch += 1

    print("Finish train")
    print("Start test")
    test(model, config, True)


def save_model(model, e):
    mm = time.time()
    fold_path = "model"
    model_path = os.path.join(fold_path, "model.ckpt")
    save_path = model.saver.save(model.sess, model_path, e)
    print("Model saved in file: %s" % save_path)
    mm2 = time.time()
    print("Save model time: %f" % (mm2 - mm))


def test(model, config, final_test):
    if final_test:
        init = tf.global_variables_initializer()
        model.sess.run(init)
        model.saver.restore(model.sess, tf.train.latest_checkpoint("model"))

    # begin test!
    step = 1
    output = []
    acc = 0
    qid = 0
    relation_acc = 0
    subject_acc = 0
    only_relation_acc = 0
    model.qa.itemIndexTest=0
    while (step - 1) * model.dev_batch_size < model.testing_iters:
        ss = time.time()
        value = model.qa.load_test_data(
            model.dev_batch_size)

        feed = {model.question_ids: value['batch_x_anonymous'],
                model.relation_index: value['batch_relation_index'],
                model.x_lens: value['batch_x_anonymous_lens'],
                model.subject_emb: value['batch_subject_emb'],
                model.subject_relation: value['batch_subject_relation'],
                model.is_training: False, }

        predict_relation, predict_subject, only_relation_predict = model.sess.run(
            [model.predict_relation_index, model.predict_subject_index, model.only_rel_predict],
            feed_dict=feed)

        for i in range(value['batch_size']):
            ans_relation = predict_relation[i]
            subject_index = predict_subject[i]

            if subject_index >= len(value['batch_subject_id'][i]):
                output.append("{}\t{}\t{}\t{}\t{}\t{}".format(
                    value['qids'][i], value['questions'][i], "not_candidate!", value['gold_subject'][i],
                    ans_relation,
                    value['gold_relation'][i]))
                qid += 1
                continue
            ans_subject = value['batch_subject_id'][i][subject_index]
            if i >= len(value['questions']):
                qid += 1
                print("2333!")
                continue
            qid += 1
            output.append("{}\t{}\t{}\t{}\t{}\t{}".format(
                value['qids'][i], value['questions'][i], ans_subject, value['gold_subject'][i],
                model.qa.rel_voc[ans_relation],
                model.qa.rel_voc[value['gold_relation'][i]]))

            if value['gold_relation'][i] == ans_relation and value['gold_subject'][i] == ans_subject:
                acc += 1
            if value['gold_relation'][i] == ans_relation:
                relation_acc += 1
            if value['gold_subject'][i] == ans_subject:
                subject_acc += 1
            if value['gold_relation'][i] == only_relation_predict[i]:
                only_relation_acc += 1

        if (step - 1) % model.display_step == 0:
            print("rate:\t%d/%d" % (step, (model.testing_iters / model.dev_batch_size)))
            ee = time.time()
            print("time:\t" + str(ee - ss))

        step += 1
    assert qid == len(model.qa.test_data)
    print("subject_acc\trelation_acc\tacc")
    print("{}\t{}\t{}".format(subject_acc * 1.0 / qid, relation_acc * 1.0 / qid, acc * 1.0 / qid))
    print("only_relation_acc: {}".format(only_relation_acc * 1.0 / qid))
    FileUtil.writeFile(output, "{}/sq.all.txt".format(config.answer_path))
