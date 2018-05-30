import torch
import pickle


class Result:
    def __init__(self, paras):
        self._results = {
            'loss': [],
            'metric': [],
            'output': None,
            'callback': None,
        }
        self._model = None

    def append(self, loss, metric):
        self._results['loss'].append(loss)
        self._results['metric'].append(metric)

    def collect(self, model, output, callback):
        self._model = model
        self._results['output'] = output
        self._results['callback'] = callback

    def get_singe_loss(self, pos=-1):
        assert len(self._results['loss']) > 0, \
            "{}: intend to retrieve loss record form empty result.".format(self.__class__.__name__)
        return self._results['loss'][pos]

    def get_singe_metric(self, pos=-1):
        assert len(self._results['metric']) > 0, \
            "{}: intend to retrieve metric record form empty result.".format(self.__class__.__name__)
        return {k: (m_train, m_test) for k, m_train, m_test in zip(self._results['metric'][pos][0].keys(),
                                                                   self._results['metric'][pos][0].values(),
                                                                   self._results['metric'][pos][1].values())}

    def get_all_loss(self):
        return self._results['loss']

    def get_all_metric(self):
        return {k: [(metric[0][k], metric[1][k])
                    for i, metric in enumerate(self._results['metric'])]
                for k in self._results['metric'][0][0].keys()}

    def get_output(self):
        return self._results['output']

    def pop_model(self):
        model = self._model
        self._model = None
        return model
