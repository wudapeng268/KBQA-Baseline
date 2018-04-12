# coding:utf-8
# author:wup
# description: build bi_gru+crf to named-entity recogniztion
# e-mail:wup@nlp.nju.cn
# date:2018.4.6

from focus_prune.load_data import SimpleQA
import tensorflow as tf
from util import NN


class fp_model():
    def __init__(self, sess, config):
        self.sess = sess
        self.qa = SimpleQA(config)
        self.batch_size = config.batch_size
        self.training_iters = len(self.qa.train_data)
        self.deving_iters = len(self.qa.dev_data)
        self.testing_iters = len(self.qa.test_data)
        self.n_input = self.qa.word_vector_len
        self.n_steps = self.qa.n_step
        self.n_tags = 2
        self.model_path = config.model_path
        self.keep_prob = config.keep_prob
        self.n_hidden = config.n_hidden
        self.learning_rate = config.learning_rate
        self.build_network()

    def build_network(self):
        # Define weights
        weights = {
            'wordEmbedding': tf.Variable(self.qa.word_embedding, trainable=True,
                                         name="word_embeddding"),

            'crf_W': tf.get_variable("crf_W",
                                     initializer=tf.random_uniform(shape=[2 * self.n_hidden, self.n_tags],
                                                                   minval=-0.08, maxval=0.08)),
            'crf_b': tf.get_variable("crf_b", initializer=tf.random_uniform(shape=[self.n_tags],
                                                                            minval=-0.08, maxval=0.08))

        }
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.question_ids = tf.placeholder(tf.int32, [None, self.n_steps])
        self.x_lens = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.n_steps])

        x = tf.nn.embedding_lookup(weights['wordEmbedding'], ids=self.question_ids)

        query4ner, _ = NN.bi_gru(x, self.x_lens, self.n_hidden, "query4ner", self.keep_prob, self.is_training)

        crf_x = tf.concat(query4ner, 2)
        matricized_x_t = tf.reshape(crf_x, [-1, 2 * self.n_hidden])  # maybe it is reasonable but pass it now!

        matricized_unary_scores = tf.matmul(matricized_x_t, weights['crf_W']) + weights['crf_b']
        self.unary_scores = tf.reshape(matricized_unary_scores,
                                       [-1, self.n_steps, self.n_tags])

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_scores, self.y, self.x_lens)

        self.loss = tf.reduce_mean(-log_likelihood)

        all_var = tf.global_variables()
        grads = tf.gradients(self.loss, all_var)
        grads = [tf.clip_by_value(grad, -10, 10) for grad in grads]

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).apply_gradients(
            zip(grads, all_var))
        self.saver = tf.train.Saver()
