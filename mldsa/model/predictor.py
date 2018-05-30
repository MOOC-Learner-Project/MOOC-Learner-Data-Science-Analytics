from ..core.builder import Builder
from ..nn.basic import *
from ..nn.abstract import *
from ..nn.loss import BCELoss, BCEWithLogitsLoss


class Predictor(Builder):

    def __init__(self, build_method):
        build_options = {
            'FC': Predictor.fc,
            'CNN1D-FC': Predictor.conv1d_fc,
            'LSTM-FC': Predictor.lstm_fc,
        }
        super(Predictor, self).__init__(build_method, build_options)

    @staticmethod
    def fc(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        if temporal_dim is not None:
            flatten = Reshape(in_size=(feature_dim, temporal_dim),
                              out_size=feature_dim*temporal_dim)
            fc = FC(in_size=temporal_dim*feature_dim, out_size=1,
                    hidden_dims=[64, 16],
                    hidden_activation='relu',
                    out_activation=None if paras.model.out_activation == 'sigmoid' else paras.model.out_activation)
            model = Block(net_list=[flatten, fc])
        else:
            model = FC(in_size=feature_dim, out_size=1,
                       hidden_dims=[16, 8],
                       hidden_activation='relu',
                       out_activation=None if paras.model.out_activation == 'sigmoid'
                       else paras.model.out_activation)
        loss_func = BCEWithLogitsLoss() if paras.model.out_activation == 'sigmoid' else BCELoss()
        return model, [loss_func], [1]

    @staticmethod
    def conv1d_fc(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        conv1d = Conv1d(in_size=(feature_dim, temporal_dim), out_size=(64, temporal_dim),
                        hidden_channels=[64, 64], kernel_sizes=[3, 3, 3],
                        strides=None, same_length=True, paddings=None,
                        hidden_activation='relu', out_activation='relu')
        flatten = Reshape(in_size=(64, temporal_dim), out_size=64*temporal_dim)
        fc = FC(in_size=64*temporal_dim, out_size=1,
                hidden_dims=[128, 32],
                hidden_activation='relu',
                out_activation=None if paras.model.out_activation == 'sigmoid' else paras.model.out_activation)
        model = Block(net_list=[conv1d, flatten, fc])
        loss_func = BCEWithLogitsLoss() if paras.model.out_activation == 'sigmoid' else BCELoss()
        return model, [loss_func], [1]

    @staticmethod
    def lstm_fc(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        lstm = LSTM(in_size=(feature_dim, temporal_dim), out_size=(64, temporal_dim),
                    hidden_sizes=[], dropouts=0.2, bidirectional=False,
                    hidden_activation='relu', out_activation='relu')
        flatten = Reshape(in_size=(64, temporal_dim), out_size=64*temporal_dim)
        fc = FC(in_size=64*temporal_dim, out_size=1,
                hidden_dims=[128, 32],
                hidden_activation='relu',
                out_activation=None if paras.model.out_activation == 'sigmoid' else paras.model.out_activation)
        model = Block(net_list=[lstm, flatten, fc])
        loss_func = BCEWithLogitsLoss() if paras.model.out_activation == 'sigmoid' else BCELoss()
        return model, [loss_func], [1]



