# coding:utf-8
# author:wup
# description: prepare train test data pre batch
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:2017-5-3
import pdb
import pickle as pkl

import numpy as np
import time
from util import FileUtil
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words.add("'s")  # add 's


class SimpleQA:
    def __init__(self, config):
        self.word_embedding_len = 100002
        self.n_step = config.n_step
        self.type_len = 500
        self.top_subject = config.top_subject

        self.word_embedding = self.wordEmbedding(config.word_embedding_path, self.word_embedding_len)
        self.rel_voc = pkl.load(open(config.rel_voc_path, "rb"))
        self.relation_size = int(len(self.rel_voc) / 2)
        self.entity2type_id = pkl.load(open(config.entity2type_path, "rb"))

        self.dev_data = pkl.load(open(config.dev_path, "rb"))
        self.train_data = pkl.load(open(config.train_path, "rb"))
        self.test_data = pkl.load(open(config.test_path, "rb"))
        print("test data lens: {}".format(len(self.test_data)))

    def wordEmbedding(self, filename, len_word_vocabulary):
        word_embedding = np.zeros((len_word_vocabulary, 300), dtype=np.float32)
        context = FileUtil.readFile(filename)
        self.word_vocabulary = {}
        for i, c in enumerate(context):
            if i >= len_word_vocabulary - 2:
                break
            wordVector = c.split(" ")
            word_embedding[i, :] = np.array([float(t) for t in wordVector[1:]])
            self.word_vocabulary[wordVector[0]] = i
        # len_word_vocabulary-2是 X -1 是真正的 unk
        self.word_vocabulary['X'] = len_word_vocabulary - 2
        return word_embedding

    itemIndexTrain = 0
    itemIndexDev = 0
    itemIndexTest = 0

    def word2id(self, query):
        vector = np.zeros(self.n_step)
        for i, word in enumerate(query.split(" ")):
            if i >= self.n_step:
                break
            if word in self.word_vocabulary:
                vector[i] = self.word_vocabulary[word]
            else:
                vector[i] = self.word_embedding_len - 1
        return vector

    def load_train_data(self, batch_size):
        if self.itemIndexTrain >= len(self.train_data):
            print("########bigger?!")
            self.itemIndexTrain = 0
        if self.itemIndexTrain + batch_size > len(self.train_data):
            batch_size = len(self.train_data) - self.itemIndexTrain

        batch_x_anonymous = np.zeros((batch_size, self.n_step), dtype=np.int32)
        batch_x_anonymous_lens = np.zeros((batch_size), dtype=np.float32)

        batch_subject_emb = np.zeros((batch_size, 1, self.type_len))
        batch_relation_index = np.zeros((batch_size, self.relation_size))
        for ind, it in enumerate(self.train_data[self.itemIndexTrain:self.itemIndexTrain + batch_size]):
            question = it.question
            anonymous_question = it.anonymous_question
            batch_x_anonymous_lens[ind] = len(anonymous_question.split(" "))

            batch_x_anonymous[ind, :] = self.word2id(anonymous_question)

            if it.subject in self.entity2type_id:
                type_ids = self.entity2type_id[it.subject]
            else:
                type_ids = []
            type_vector = np.zeros(self.type_len)
            for t in type_ids:
                type_vector[t] = 1
            batch_subject_emb[ind, 0, :] = type_vector

            batch_relation_index[ind, it.relation] = 1

        self.itemIndexTrain += batch_size
        return_value = {}
        value_name = ['batch_x_anonymous', 'batch_x_anonymous_lens', 'batch_relation_index',
                      'batch_subject_emb']

        for name in value_name:
            return_value[name] = eval(name)
        return return_value

    def load_test_data(self, batch_size, dataset="test"):
        if dataset == "test":
            runitem = self.test_data
            run_item_index = self.itemIndexTest
        else:
            runitem = self.dev_data
            run_item_index = self.itemIndexDev
        if run_item_index + batch_size > len(runitem):
            batch_size = len(runitem) - run_item_index

        batch_x_anonymous = np.zeros((batch_size, self.n_step), dtype=np.int32)
        batch_x_anonymous_lens = np.zeros((batch_size), dtype=np.float32)

        batch_subject_emb = np.zeros((batch_size, self.top_subject, self.type_len))
        batch_relation_index = np.zeros((batch_size, self.relation_size))

        qids = []
        batch_subject_id = []
        questions = []
        gold_relation = []
        gold_subject = []

        batch_subject_relation = np.zeros((batch_size, self.top_subject, self.relation_size),
                                          dtype=np.float32)
        for ind, it in enumerate(runitem[run_item_index:run_item_index + batch_size]):
            question = it.question
            questions.append(question)
            # print("qid {}".format(it.qid))
            qids.append(it.qid)
            anonymous_question = it.anonymous_question
            batch_x_anonymous_lens[ind] = len(anonymous_question.split(" "))
            for i, word in enumerate(anonymous_question.split(" ")):
                if i >= self.n_step:
                    break
                if word in self.word_vocabulary:
                    batch_x_anonymous[ind,  i] = self.word_vocabulary[word]
                else:
                    batch_x_anonymous[ind,  i] = self.word_embedding_len - 1
            gold_relation.append(it.relation)
            gold_subject.append(it.subject)

            if not hasattr(it, "cand_rel"):
                batch_subject_id.append([])
                continue
            relation_index = it.cand_rel

            batch_subject = it.cand_sub
            batch_subject = batch_subject[:self.top_subject]
            batch_subject_id.append(batch_subject)

            for i, subject in enumerate(batch_subject):
                if subject in self.entity2type_id:
                    type_ids = self.entity2type_id[subject]
                    for id in type_ids:
                        batch_subject_emb[ind, i, id] = 1
            sub_rels = it.sub_rels
            sub_rels = sub_rels[:self.top_subject]
            if hasattr(it,"sub_rels"):
                for i, rel in enumerate(sub_rels):
                    for r in rel:
                        if r in relation_index:
                            batch_subject_relation[ind, i, r] = 1
                        else:
                            print("qid {} relation {} not found in cand_rel".format(it.qid,r))
            else:
                print("qid: {} not have sub_rells attribute".format(it.qid))
            batch_relation_index[ind, it.relation] = 1

        if dataset == "test":
            self.itemIndexTest += batch_size
        else:
            self.itemIndexDev += batch_size
        return_value = {}

        value_name = ['batch_x_anonymous', 'batch_x_anonymous_lens',
                      'batch_size', 'questions', 'batch_subject_emb', 'batch_subject_relation', 'gold_subject',
                      'gold_relation', 'batch_subject_id',
                      ]
        value_name.append("batch_relation_index")
        value_name.append("qids")
        for name in value_name:
            return_value[name] = eval(name)
        return return_value
