from .builder import Builder
from .loader import Loader
from .logger import Logger
from .trainer import Trainer
from .recorder import Recorder
from .parameter import Parameter

from ..model.predictor import Predictor
from ..model.autoencoder import AutoEncoder
from ..model.advanced import AutoEncoderWithPredictor

# TODO: Change the way of achieving this mapping

MODEL_TYPE_TO_CLASS = {
    'predictor': Predictor,
    'autoencoder': AutoEncoder,
    'combined': AutoEncoderWithPredictor
}


class Manager:
    def __init__(self, paras):
        self.paras = Parameter(paras)

    def __call__(self):
        builder = MODEL_TYPE_TO_CLASS[self.paras.model.type](self.paras.model.architecture)
        loader = Loader(self.paras)
        logger = Logger(self.paras)
        recorder = Recorder(self.paras)
        trainer = Trainer(builder, loader, logger, recorder)
        trainer(self.paras)
        Recorder.save(recorder)
