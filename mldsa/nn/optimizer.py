from torch import optim


class Optimizer:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer_options = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'RMSprop': optim.RMSprop,
        }
        self.scheduler_options = {
            'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau
        }
        if optimizer not in self.optimizer_options:
            raise NotImplementedError("{}: invalid optimizer {} indicated."
                                      "".format(self.__class__.__name__, optimizer))
        self.optimizer = self.optimizer_options[optimizer]
        self.scheduler = None
        if scheduler is not None:
            if scheduler not in self.scheduler_options:
                raise NotImplementedError("{}: invalid scheduler {} indicated."
                                          "".format(self.__class__.__name__, scheduler))
            self.scheduler = self.scheduler_options[scheduler]

    def __call__(self, model, paras):
        optimizer = self.optimizer(model.parameters(), lr=paras.train.learning_rate)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer)
            return optimizer, scheduler
