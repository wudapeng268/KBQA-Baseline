import os


class setting():
    def __init__(self):
        self.data_prefix = "/home/user_data/wup/kbqa_data/"
        self.epoch = 1000
        self.continue_train = False
        self.normalize = True
        self.dev_epoch = 5
        self.learning_rate = 1e-3
        self.log_path = "_log"
        self.test_path = os.path.join(self.data_prefix, "test.data.cfo.pickle")
        self.dev_path = os.path.join(self.data_prefix, "dev.small.pickle")
        self.train_path = os.path.join(self.data_prefix, "train.data.pickle")
        self.word_embedding_path = os.path.join(self.data_prefix, "glove.6B.300d.txt")
        self.rel_voc_path = os.path.join(self.data_prefix, "rel_voc.pickle")
        self.entity2type_path = os.path.join(self.data_prefix, "entity2type.fb5m.pickle")
        self.optimizer = "adam"
        self.top_subject = 100
        self.batch_size = 256
        self.dev_batch_size = 256
        self.n_step = 40
        self.answer_path = "output"
        self.n_hidden = 256
        self.relation_embedding_size = 300
        self.keep_prob = 0.5
        self.first_dev = False
