from ..core.builder import Builder
from ..nn.abstract import *
from .predictor import Predictor
from .autoencoder import AutoEncoder


class AutoEncoderWithPredictor(Builder):

    def __init__(self, build_method):
        self.autoencoder_builder = AutoEncoder(build_method.split('+')[0])
        self.predictor_builder = Predictor(build_method.split('+')[1])
        self.ratio = 1
        super(AutoEncoderWithPredictor, self).__init__(None, {None: None})

    def __call__(self, paras):
        ae_model, ae_loss_funcs, ae_weights = self.autoencoder_builder(paras)
        feature_dim = paras.data.feature_dim
        paras.data = paras.data._replace(feature_dim=ae_model.out_size.size[1][0])
        p_model, p_loss_funcs, p_weights = self.predictor_builder(paras)
        paras.data = paras.data._replace(feature_dim=feature_dim)
        identical_encode = Identical(in_size=ae_model.out_size.size[0])
        repeat_embedding = Repeat(in_size=ae_model.out_size.size[1], num_of_copies=2)
        identical_embedding = Identical(in_size=ae_model.out_size.size[1])
        model = Block(net_list=[ae_model, (identical_encode,
                                           Block(net_list=[repeat_embedding, (identical_embedding, p_model)]))])
        loss_funcs = [ae_loss_funcs[0], None, p_loss_funcs[0]]
        weights = [1, None, self.ratio]
        return model, loss_funcs, weights

