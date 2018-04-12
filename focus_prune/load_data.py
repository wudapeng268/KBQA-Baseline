# coding:utf-8
# author:wup
# description: read train dev data for focus prune
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:2017-5-3


import numpy as np

from util import FileUtil
import pickle as pkl


class SimpleQA:
    def __init__(self, config):
        self.n_step = config.n_step
        self.word_vector_len = 300
        self.word_embedding_len = 100002
        self.batch_index_train = 0
        self.batch_index_dev = 0
        self.batch_index_test = 0

        self.word_embedding = self.wordEmbedding(config.word_embedding_path, self.word_embedding_len)

        self.train_data = pkl.load(open(config.train_path, "rb"))
        self.dev_data = pkl.load(open(config.dev_path, "rb"))
        self.test_data = pkl.load(open(config.test_path, "rb"))

    def wordEmbedding(self, filename, len_word_vocabulary):
        word_embedding = np.zeros((len_word_vocabulary, self.word_vector_len), dtype=np.float32)
        context = FileUtil.readFile(filename)
        self.word_vocabulary = {}
        for i, c in enumerate(context):
            if i >= len_word_vocabulary:
                break
            wordVector = c.split(" ")
            word_embedding[i, :] = np.array([float(t) for t in wordVector[1:]])
            self.word_vocabulary[wordVector[0]] = i
        return word_embedding



    def get_questions(self, dataset="test"):
        if dataset == "test":
            runitem = self.test_data
        else:
            runitem = self.dev_data
        questions = []
        for item in runitem:
            questions.append(item.question)
        return questions

    def load_data(self, batch_size, data_set="train"):
        if data_set == "train":
            run_item = self.train_data
            run_index = self.batch_index_train
        elif data_set == "test":
            run_item = self.test_data
            run_index = self.batch_index_test
        else:
            run_item = self.dev_data
            run_index = self.batch_index_dev
        if run_index + batch_size > len(run_item):
            batch_size = len(run_item) - run_index
        batch_x = np.zeros((batch_size, self.n_step), dtype=np.int32)
        batch_y = np.zeros((batch_size, self.n_step), dtype=np.int32)
        question_len = []
        for ind, it in enumerate(run_item[run_index:run_index + batch_size]):
            question = it.question
            question_len.append(len(question.split(" ")))
            vector = np.zeros(self.n_step)
            for i, word in enumerate(question.split(" ")):
                if i >= self.n_step:
                    break
                if word in self.word_vocabulary:
                    vector[i] = self.word_vocabulary[word]
                else:
                    vector[i] = self.word_embedding_len - 1

            batch_x[ind, :] = vector
            index_in_anony = 0
            index_in_raw = 0
            anonymous = it.anonymous_question.split(" ")
            while (index_in_anony < min(self.n_step, len(anonymous))):
                if anonymous[index_in_anony] != "X":
                    batch_y[ind, index_in_raw] = 0
                    index_in_raw += 1
                    index_in_anony += 1
                else:
                    subject_text_len = len(it.subject_text.split(" "))
                    for i in range(subject_text_len):
                        batch_y[ind, index_in_raw + i] = 1
                    index_in_raw += subject_text_len
                    index_in_anony += 1

        if data_set == "train":
            self.batch_index_train += batch_size
        elif data_set == "test":
            self.batch_index_test += batch_size
        else:
            self.batch_index_dev += batch_size
        return np.array(question_len, dtype=np.int32), batch_x, batch_y
