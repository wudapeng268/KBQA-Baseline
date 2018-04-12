# coding:utf-8
# author:wup
# description: build neural network for our model, when add some new structure, change this file
# e-mail:wup@nlp.nju.cn
# date: 2018.4.4
import tensorflow as tf
from src.load_data import SimpleQA
from util import NN
import pdb


class BiGRU(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.n_steps = config.n_step
        self.n_hidden = config.n_hidden
        self.fix_batch_size = config.batch_size

        self.display_step = 100
        self.dev_batch_size = config.dev_batch_size
        self.keep_prob = config.keep_prob
        self.normalize = config.normalize

        self.qa = SimpleQA(config)
        self.deving_iters = len(self.qa.dev_data)
        self.training_iters = len(self.qa.train_data)
        self.testing_iters = len(self.qa.test_data)

        self.relation_vocabulary_size = int(self.qa.relation_size)
        self.relation_embedding_size = int(config.relation_embedding_size)

        self.build_model()

    def matmul_query_relation(self, query_vec, rel_vec, name):
        query_vec = NN.linear(query_vec, int(rel_vec.shape[-1]), name=name)
        score = tf.matmul(query_vec, tf.transpose(rel_vec))
        # [batch,cand_rel]
        return score

    def relation_network(self, x):
        '''
        :param x: query embedding after look-up
        :return: rel_score:[batch_size,relation_voc_size], loss:[1]
        '''
        with tf.variable_scope("relation_network"):
            _, query4relation = NN.bi_gru(x, self.x_lens, self.n_hidden, "bi_gru4relation_query", self.keep_prob,
                                          self.is_training)
            # [batch_size,n_hidden*2]
            query4relation = tf.concat(query4relation, -1)

            # [batch,relation_voc_size]
            rel_score = self.matmul_query_relation(query4relation, self.weight['relationEmbedding'],
                                                   "rel_score")
            self.only_rel_predict = tf.argmax(rel_score, 1)
            # [1]
            loss_relation = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.relation_index, logits=rel_score), axis=0)
            return rel_score, loss_relation

    def subject_network(self, x):
        '''
        :param x: query embedding after look-up
        :return:subject_score:[batch_size,cand_sub_size] loss_subject:[1]
        '''
        with tf.variable_scope("subject_part"):
            _, query4subject = NN.bi_gru(x, self.x_lens, self.n_hidden, "bi_gru4subject_query", self.keep_prob,
                                         self.is_training)
            query4subject = tf.concat(query4subject, -1)
            # [batch,type_len]
            query4subject = tf.sigmoid(NN.linear(query4subject, self.qa.type_len, name="query_trans_subject"))
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                self.type_emb_w_martix = tf.get_variable("query_trans_subjectkernel")
                self.type_emb_b_martix = tf.get_variable("query_trans_subjectbias")

            # [batch_size,emb_size]
            gold_subject = self.subject_emb[:, 0, :]
            self.loss_subject = gold_subject * tf.log(tf.clip_by_value(query4subject, 1e-10, 1.0)) + (
                                                                                                         1 - gold_subject) * tf.log(
                tf.clip_by_value(1 - query4subject, 1e-10, 1.0))
            # [1]
            loss_subject = tf.reduce_mean(- tf.reduce_sum(self.loss_subject, axis=1), axis=0)

            with tf.variable_scope("test"):
                # [batch,1,cand_sub]
                subject_score = tf.matmul(tf.expand_dims(query4subject, 1),
                                          tf.transpose(self.subject_emb, perm=[0, 2, 1]))
                # [batch,cand_sub,1]
                subject_score = tf.squeeze(subject_score, 1)
                # subject_score = tf.transpose(subject_score, [0, 2, 1])
            return subject_score, loss_subject

    def build_model(self):
        self.weight = {
            'wordEmbedding': tf.Variable(self.qa.word_embedding, trainable=True,
                                         name="word_embedding"),

            'relationEmbedding': tf.Variable(
                tf.random_uniform(shape=[self.relation_vocabulary_size, self.relation_embedding_size],
                                  minval=-0.08,
                                  maxval=0.08), trainable=True, name="relation_all_embedding"),
        }
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.question_ids = tf.placeholder(tf.int32, [None, self.n_steps])
        self.x_lens = tf.placeholder(tf.int32, [None])
        self.relation_index = tf.placeholder(tf.int32, [None, self.relation_vocabulary_size])
        self.subject_relation = tf.placeholder(tf.float32,
                                               [None, self.qa.top_subject, self.relation_vocabulary_size])
        self.subject_emb = tf.placeholder(tf.float32, [None, None, self.qa.type_len])

        x = tf.nn.embedding_lookup(self.weight['wordEmbedding'], ids=self.question_ids)

        rel_score, self.loss_relation = self.relation_network(x)

        subject_score, self.loss_subject = self.subject_network(x)

        with tf.variable_scope("merge_part"):
            # remove score which have not path from subject to relation
            subject_score = tf.expand_dims(subject_score, 2)
            rel_score = tf.expand_dims(rel_score, 1)
            self.output_score = (rel_score + subject_score) * self.subject_relation
            self.rel_score = rel_score
            self.subject_score = subject_score
            self.predict_subject_index = tf.argmax(tf.reduce_max(self.output_score, -1), -1)
            self.predict_relation_index = tf.argmax(tf.reduce_max(self.output_score, 1), -1)

        # add l2 loss
        self.l2_loss = tf.zeros([])
        if self.normalize:
            self.l2_loss = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(0.001),
                weights_list=tf.trainable_variables())
        self.loss_all = self.loss_relation + self.loss_subject + self.l2_loss

        self.saver = tf.train.Saver()
