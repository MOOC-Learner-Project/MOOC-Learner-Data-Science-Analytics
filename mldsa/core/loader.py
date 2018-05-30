import os
import numpy as np
from torch import from_numpy
from torch.autograd import Variable


class Loader:
    def __init__(self, paras):
        assert os.path.isdir(paras.config.data_dir), "{}: indicated data directory {} " \
                                                     "does not exist.".format(self.__class__.__name__,
                                                                              paras.config.data_dir)
        if not paras.config.suffix:
            suffix = set([])
            for file in os.listdir(paras.config.data_dir):
                if (file.startswith('x') or file.startswith('y')) and file.endswith('.npy'):
                    suffix.add(file[1:-4])
        elif isinstance(paras.config.suffix, list) and all([isinstance(p, str) for p in paras.config.suffix]):
            suffix = set(paras.config.suffix)
        elif isinstance(paras.config.suffix, str):
            suffix = {paras.config.suffix}
        else:
            assert False, "{}: invalid suffix {}, it should be either None, " \
                          "string or a list of strings.".format(self.__class__.__name__, paras.config.suffix)
        self.suffix = list(suffix)
        self.num = len(suffix)
        self.x = []
        self.y = []
        for s in self.suffix:
            assert (os.path.isfile(paras.config.data_dir + 'x' + s + '.npy') and
                    os.path.isfile(paras.config.data_dir + 'y' + s + '.npy')), \
                "{}: feature and label files with suffix {} do not appear in pairs.".format(self.__class__.__name__, s)
            self.x.append(np.load(paras.config.data_dir + 'x' + s + '.npy'))
            self.y.append(np.load(paras.config.data_dir + 'y' + s + '.npy'))
        self.size = []
        self.idx = []
        for i, s in enumerate(self.suffix):
            assert self.x[i].shape[0] == self.y[i].shape[0], \
                "{}: the number of samples in feature set and label set " \
                "with suffix {} does not match.".format(self.__class__.__name__, s)
            self.size.append(self.x[i].shape[0])
            self.idx.append(np.arange(self.x[i].shape[0]))
        print("{}: Data loaded:".format(self.__class__.__name__))
        for i, p in enumerate(self.suffix):
            print(" - suffix: {} - feature shape: {} - label shape: {}".format(p, self.x[i].shape, self.y[i].shape))
        self.process_options = {
            'temporal': Loader.temporal,
        }
        self.split_options = {
            'cross_validation': Loader.cross_validation,
        }
        self.feed_options = {
            'predictor': Loader.predictor,
            'autoencoder': Loader.autoencoder,
            'combined': Loader.combined,
        }
        assert paras.data.process_method in self.process_options, \
            "{}: invalid process method {}, valid options are {}." \
            "".format(self.__class__.__name__, paras.data.process_method, self.process_options)
        assert paras.data.split_method in self.split_options, \
            "{}: invalid split method {}, valid options are {}." \
            "".format(self.__class__.__name__, paras.data.split_method, self.split_options)
        assert paras.data.feed_method in self.feed_options, \
            "{}: invalid feed method {}, valid options are {}." \
            "".format(self.__class__.__name__, paras.data.feed_method, self.feed_options)
        self.process_method = paras.data.process_method
        self.split_method = paras.data.split_method
        self.feed_method = paras.data.feed_method
        self.xp = self.yp = None
        self.x_train = self.y_train = self.x_test = self.y_test = None

    def shuffle(self):
        for i in range(self.num):
            np.random.shuffle(self.idx[i])
            self.x[i] = self.x[i][self.idx[i]]
            self.y[i] = self.y[i][self.idx[i]]

    def process(self, opt):
        self.xp = []
        self.yp = []
        for i in range(self.num):
            xp, yp = opt(self.x[i], self.y[i])
            self.xp.append(xp)
            self.yp.append(yp)

    def split_and_feed(self, split_opt, feed_opt):
        self.x_train, self.y_train, self.x_test, self.y_test = feed_opt(*split_opt(self.xp, self.yp))

    def chunk(self, batch_size):
        return Loader._pack_list_of_chunks([Loader._chunk_single(x, self.x_train[0].shape[0], batch_size)
                                            for x in self.x_train],
                                           [Loader._chunk_single(y, self.x_train[0].shape[0], batch_size)
                                            for y in self.y_train]),\
               Loader._pack_list_of_chunks([Loader._chunk_single(x, self.x_test[0].shape[0], batch_size)
                                            for x in self.x_test],
                                           [Loader._chunk_single(y, self.x_test[0].shape[0], batch_size)
                                            for y in self.y_test]),

    @staticmethod
    def _chunk_single(data_matrix, size, batch_size):
        return [Variable(from_numpy(data_matrix[i * batch_size:(i + 1) * batch_size]).float())
                for i in range(size // batch_size)] if data_matrix is not None else [None] * (size // batch_size)

    @staticmethod
    def _pack_list_of_chunks(x_lists, y_lists):
        size = len(x_lists[0])
        assert (all([len(x_list) == size for x_list in x_lists]) and all([len(y_list) == size for y_list in y_lists])), \
            "Loader: all chuck lists should have the same length."
        return [(tuple([x_list[i] for x_list in x_lists]), tuple([y_list[i] for y_list in y_lists])) for i in range(size)]

    def __call__(self, paras):
        self.shuffle()
        self.process(self.process_options[self.process_method](paras))
        self.split_and_feed(self.split_options[self.split_method](paras),
                            self.feed_options[self.feed_method]())
        # TODO: Desiging ways of collecting parameters form various control objects
        paras.feature_shape = self.x_train[0].shape[1:]
        paras.label_shape = self.y_train[0].shape[1:]
        return self.chunk(paras.train.batch_size)

    @staticmethod
    def temporal(paras):
        def _opt(x, y):
            return np.swapaxes(x[:, :paras.data.temporal_dim], 1, 2), y[:, paras.data.temporal_dim].reshape(y.shape[0], 1)
        return _opt

    @staticmethod
    def cross_validation(paras):
        def _opt(xp, yp):
            if not (len(xp) == 1 and len(yp) == 1):
                raise ValueError("Loader: cross validation could only be used on one data set.")
            xp = xp[0]
            yp = yp[0]
            batch_size = paras.train.batch_size
            num_of_folds = xp.shape[0] // batch_size
            num_of_trials = paras.get_iter_lengths()['train:trial']
            cur_trial = paras.train.trial
            x_train = np.concatenate((xp[:(num_of_folds // num_of_trials * cur_trial) * batch_size],
                                      xp[(num_of_folds // num_of_trials * (cur_trial + 1)) * batch_size:]), axis=0)
            y_train = np.concatenate((yp[:(num_of_folds // num_of_trials * cur_trial) * batch_size],
                                      yp[(num_of_folds // num_of_trials * (cur_trial + 1)) * batch_size:]), axis=0)
            x_test = xp[(num_of_folds // num_of_trials * cur_trial) * batch_size:
                        (num_of_folds // num_of_trials * (cur_trial + 1)) * batch_size]
            y_test = yp[(num_of_folds // num_of_trials * cur_trial) * batch_size:
                        (num_of_folds // num_of_trials * (cur_trial + 1)) * batch_size]
            return x_train, y_train, x_test, y_test
        return _opt

    @staticmethod
    def predictor():
        def _opt(x_train, y_train, x_test, y_test):
            return [x_train], [y_train], [x_test], [y_test]
        return _opt

    @staticmethod
    def autoencoder():
        def _opt(x_train, y_train, x_test, y_test):
            return [x_train], [x_train, None], [x_test], [x_test, None]
        return _opt

    @staticmethod
    def combined():
        def _opt(x_train, y_train, x_test, y_test):
            return [x_train], [x_train, None, y_train], [x_test], [x_test, None, y_test]
        return _opt
