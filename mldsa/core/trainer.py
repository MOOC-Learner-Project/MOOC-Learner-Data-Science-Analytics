import torch
from ..nn.metric import Metric
from ..nn.optimizer import Optimizer
from .result import Result
from .callback import CallBack


class Trainer:
    def __init__(self, builder, loader, logger, recorder):
        self.builder = builder
        self.loader = loader
        self.logger = logger
        self.recorder = recorder
        self.use_cuda = torch.cuda.is_available()

    def __call__(self, paras):
        print("{}: Training started.".format(self.__class__.__name__))
        self.logger.start_mission(paras)
        paras_proto = paras
        for paras in paras_proto:
            self.logger.log_mission(paras)
            data_train, data_test = self.loader(paras)
            model, loss_funcs, weights = self.builder(paras)
            if self.use_cuda:
                model.cuda()
            metric_funcs = Metric(paras.log.metric, paras.data.feed_method)
            optim = Optimizer(paras.train.optimizer)(model, paras)
            results = Result(paras)
            callbacks = [CallBack(cb) for cb in paras.log.callback]
            self.logger.start_epoch(paras)
            os_train = os_test = None
            for cur_epoch in range(paras.train.nepochs):
                loss_train = 0
                loss_test = 0
                metric_train = {m: 0 for m in metric_funcs.get_short_name()}
                metric_test = {m: 0 for m in metric_funcs.get_short_name()}
                for xs, ys in data_train:
                    if self.use_cuda:
                        xs = [x.cuda() for x in xs]
                        ys = [y.cuda() for y in ys]
                    optim.zero_grad()
                    os_train = model(*xs)
                    loss = sum([loss_func(o, y) * weight
                                for o, y, loss_func, weight in zip(os_train, ys, loss_funcs, weights) if y is not None])
                    # TODO: check how to avoid this for LSTM layer and others
                    loss.backward(retain_graph=True)
                    loss_train += loss.data.cpu().numpy()
                    metric = metric_funcs(os_train, ys)
                    metric_train = {m: metric_train[m] + metric[m] for m in metric_train}
                    optim.step()
                for xs, ys in data_test:
                    if self.use_cuda:
                        xs = [x.cuda() for x in xs]
                        ys = [y.cuda() for y in ys] 
                    os_test = model(*xs)
                    loss = sum([loss_func(o, y) * weight
                                for o, y, loss_func, weight in zip(os_test, ys, loss_funcs, weights) if y is not None])
                    loss_test += loss.data.cpu().numpy()
                    metric = metric_funcs(os_test, ys)
                    metric_test = {m: metric_test[m] + metric[m] for m in metric_test}
                loss_train = loss_train / len(data_train)
                loss_test = loss_test / len(data_test)
                metric_train = {m: metric_train[m] / len(data_train) for m in metric_train}
                metric_test = {m: metric_test[m] / len(data_test) for m in metric_test}
                results.append((loss_train, loss_test), (metric_train, metric_test))
                self.logger.log_epoch(paras, results)
            results.collect(model, (os_train, os_test), None)
            [cb(paras, results) for cb in callbacks]
            self.recorder(paras, results)
        print("{}: Training finished.".format(self.__class__.__name__))
