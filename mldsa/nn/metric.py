import torch
from torch.nn import Module
from torch.nn import functional as F


class Metric:
    def __init__(self, metric_dict, feed_method):
        self.metric_options = {
            'acc': ['acc', Metric.accuracy_binary_with_logits],
            'fpr': ['fpr', Metric.fpr_binary_with_logits],
            'mse': ['mse', Metric.mse]
        }
        # TODO: Find a better way of metric-output matching
        self.output_options = {
            'predictor': ['pdt'],
            'autoencoder': ['rct', 'ebd'],
            'combined': ['rct', 'ebd', 'pdt'],
        }
        self.output_names = self.output_options[feed_method]
        if not all([k in self.output_names for k in metric_dict.keys()]):
            raise ValueError("{}: invalid output name indicated.".format(self.__class__.__name__))
        if not all([m in self.metric_options for m in metric_dict.values()]):
            raise ValueError("{}: invalid metric option indicated.".format(self.__class__.__name__))

        self.metrics = [metric_dict.get(k, None) for k in self.output_names]

    def __call__(self, input, target):
        return {n + '_' + self.metric_options[m][0]:
                    self.metric_options[m][1](i, t).data.cpu().numpy()
                for i, t, n, m in zip(input, target, self.output_names, self.metrics) if m is not None}

    def get_short_name(self):
        return [n + '_' + self.metric_options[m][0] for n, m in zip(self.output_names, self.metrics) if m is not None]

    @staticmethod
    def accuracy_binary_with_logits(input, target):
        input = (input > 0).float()
        return 1 - torch.abs(input - target).mean()

    @staticmethod
    def fpr_binary_with_logits(input, target):
        return (input < 0).__and__(target == 1).float().mean()

    @staticmethod
    def mse(input, target):
        return F.mse_loss(input, target)

