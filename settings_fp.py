import os
class setting_fp():
    def __init__(self):
        self.continue_train = False
        self.dev_epoch = 10
        self.add_bn = False
        self.answer_path = "fp_output"
        self.learning_rate = 0.02
        self.batch_size = 256
        self.display_step = 100
        self.n_step = 40
        # Network Parameters
        self.n_hidden = 256
        self.n_tags = 2
        self.epoch = 500
        self.save_epoch = 2
        self.keep_prob = 0.5
        self.num_layer = 2
        self.data_prefix = "/home/user_data/wup/kbqa_data/"
        self.test_path = os.path.join(self.data_prefix, "test.data.cfo.pickle")
        self.dev_path = os.path.join(self.data_prefix, "dev.small.pickle")
        self.train_path = os.path.join(self.data_prefix, "train.data.pickle")
        self.word_embedding_path = os.path.join(self.data_prefix, "glove.6B.300d.txt")
        self.model_path = "fp_model"