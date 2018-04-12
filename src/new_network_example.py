from network import BiGRU


class new_network(BiGRU):
    def __init__(self, sess, config, a, b):
        BiGRU.__init__(sess, config)
        self.a = a
        self.b = b

    # overrides function
    def subject_network(self, x):
        pass


if __name__ == '__main__':
    aaa = new_network(None, None)
