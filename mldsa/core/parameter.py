import itertools
import collections
from collections import namedtuple

PARAMETER_VALIDATION = {
    'config': {
        'task_name': None,
        'data_dir': lambda x: isinstance(x, str),
        'result_dir': lambda x: isinstance(x, str),
        'suffix': lambda x: isinstance(x, str),
    },
    'data': {
        'temporal_dim': None,
        'contextual_dim': None,
        'feature_dim': None,
        'label_dim': None,
        'process_method': lambda x: x in ['temporal'],
        'split_method': lambda x: x in ['cross_validation'],
        'feed_method': lambda x: x in ['predictor', 'autoencoder', 'combined'],
    },
    'model': {
        'type': None,
        'architecture': None,
        'out_activation': None,
        'regularization': None,
        'dropout_prob': None,
    },
    'train': {
        'nepochs': None,
        'optimizer': None,
        'batch_size': None,
        'learning_rate': None,
        'decay': None,
        'trial': None,
    },
    'log': {
        'metric': lambda x: isinstance(x, dict),
        'callback': lambda x: isinstance(x, list),
        'verbosity': None,
    },
    'analyze': {

    }
}

# TODO: Think of better ways of integrating the Parameter class with Namedtuples
Config = namedtuple('config', PARAMETER_VALIDATION['config'].keys())
Data = namedtuple('data', PARAMETER_VALIDATION['data'].keys())
Model = namedtuple('model', PARAMETER_VALIDATION['model'].keys())
Train = namedtuple('train', PARAMETER_VALIDATION['train'].keys())
Log = namedtuple('log', PARAMETER_VALIDATION['log'].keys())
Analyze = namedtuple('analyze', PARAMETER_VALIDATION['analyze'].keys())


class Parameter:
    def __init__(self, paras):
        self._paras = Parameter.flatten(paras)
        self._iterators = Parameter.filter(self._paras)
        self._iter_lists = Parameter.traversal(self._iterators)
        assert Parameter.validate(self._paras, Parameter.flatten(PARAMETER_VALIDATION)), \
            "{}: invalid input parameter dictionary.".format(self.__class__.__name__)
        self._it = None
        self._cur_iter_paras = None

        # Namedtuple fields
        self.config = self.data = self.model = self.train = self.log = self.analyze = None
        self.update()

    @staticmethod
    def flatten(d, sep=':'):
        return {pk + sep + ck: v for pk, sd in d.items() for ck, v in sd.items()}

    @staticmethod
    def validate(d, c):
        for k, v in d.items():
            assert k in c, "Parameter: input parameter dictionary contain invalid field {}.".format(k)
            if c[k] is None:
                continue
            if isinstance(v, collections.Iterator):
                if not all(c[k](sv) for sv in list(itertools.tee(v)[1])):
                    assert False, "Parameter: input field {} contains invalid values.".format(k)
            elif not c[k](v):
                assert False, "Parameter: input field {} contains invalid value {}.".format(k, v)
        return True

    @staticmethod
    def filter(d):
        return {k: v for k, v in d.items() if isinstance(v, collections.Iterator)}

    @staticmethod
    def traversal(its):
        d = {}
        for k, it in its.items():
            it, it_copy = itertools.tee(it)
            its[k] = it
            d[k] = list(it_copy)
        return d

    @staticmethod
    def replace(d, r):
        for k in r:
            d[k] = r[k]

    def update(self):
        self.config = Config(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('config')})
        self.data = Data(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('data')})
        self.model = Model(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('model')})
        self.train = Train(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('train')})
        self.log = Log(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('log')})
        self.analyze = Analyze(**{k.split(':')[1]: v for k, v in self._paras.items() if k.startswith('analyze')})

    def destroy(self):
        self.config = self.data = self.model = self.train = self.log = self.analyze = None

    def get_all_keys(self):
        return list(self._paras.keys())

    def get_iter_keys(self):
        return list(self._iterators.keys())

    def get_iter_lists(self):
        return self._iter_lists

    def get_iter_lengths(self):
        return {k: len(l) for k, l in self._iter_lists.items()}

    def get_cur_iter_paras(self):
        return self._cur_iter_paras

    def __iter__(self):
        self._it = itertools.product(*self._iterators.values())
        self._cur_iter_paras = None
        return self

    def __next__(self):
        self._cur_iter_paras = dict(zip(self._iterators.keys(), next(self._it)))
        Parameter.replace(self._paras, self._cur_iter_paras)
        self.update()
        return self
