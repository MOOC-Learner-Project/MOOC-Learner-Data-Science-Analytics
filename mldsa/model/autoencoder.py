from ..core.builder import Builder
from ..nn.basic import *
from ..nn.abstract import *
from ..nn.loss import MSELoss


class AutoEncoder(Builder):

    def __init__(self, build_method):
        build_options = {
            'PCA': AutoEncoder.pca,
            'CNN1D-AE': AutoEncoder.conv1d_convtrans1d,
            'LSTM-AE': AutoEncoder.lstm,
        }
        super(AutoEncoder, self).__init__(build_method, build_options)

    @staticmethod
    def pca(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        flatten1 = Reshape(in_size=(feature_dim, temporal_dim),
                           out_size=feature_dim*temporal_dim)
        fc1 = FC(in_size=temporal_dim*feature_dim,
                 out_size=2*temporal_dim, hidden_dims=[],
                 hidden_activation=None, out_activation=None)
        reshape1 = Reshape(in_size=2*temporal_dim, out_size=(2, temporal_dim))
        repeat = Repeat(in_size=reshape1.out_size, num_of_copies=2)
        identical = Identical(in_size=reshape1.out_size)
        flatten2 = Reshape(in_size=(2, temporal_dim),
                           out_size=2*temporal_dim)
        fc2 = FC(in_size=fc1.out_size,
                 out_size=temporal_dim*feature_dim, hidden_dims=[],
                 hidden_activation=None, out_activation=None)
        reshape2 = Reshape(in_size=temporal_dim*feature_dim,
                          out_size=(feature_dim, temporal_dim))
        model = Block(net_list=[flatten1, fc1, reshape1, repeat, (Block(net_list=[flatten2, fc2, reshape2]), identical)])
        return model, [MSELoss(), None], [1, None]

    @staticmethod
    def conv1d_convtrans1d(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        conv1d = Conv1d(in_size=(feature_dim, temporal_dim), out_size=(2, temporal_dim),
                        hidden_channels=[32, 16], kernel_sizes=[3, 3, 3],
                        strides=None, same_length=True, paddings=None,
                        hidden_activation='relu', out_activation='relu')
        repeat = Repeat(in_size=conv1d.out_size, num_of_copies=2)
        identical = Identical(in_size=conv1d.out_size)
        convtrans1d = ConvTranspose1d(in_size=(2, temporal_dim), out_size=(feature_dim, temporal_dim),
                                      hidden_channels=[16, 32], kernel_sizes=[3, 3, 3],
                                      strides=None, same_length=True, paddings=None,
                                      hidden_activation='relu', out_activation='sigmoid')
        model = Block(net_list=[conv1d, repeat, (convtrans1d, identical)])
        return model, [MSELoss(), None], [1, None]

    @staticmethod
    def lstm(paras):
        temporal_dim = paras.data.temporal_dim
        feature_dim = paras.data.feature_dim
        lstm1 = LSTM(in_size=(feature_dim, temporal_dim), out_size=(2, temporal_dim),
                     hidden_sizes=[], dropouts=0.2, bidirectional=False,
                     hidden_activation='relu', out_activation='relu')
        repeat = Repeat(in_size=lstm1.out_size, num_of_copies=2)
        identical = Identical(in_size=lstm1.out_size)
        lstm2 = LSTM(in_size=(2, temporal_dim), out_size=(feature_dim, temporal_dim),
                     hidden_sizes=[], dropouts=0.2, bidirectional=False,
                     hidden_activation='relu', out_activation='relu')
        model = Block(net_list=[lstm1, repeat, (lstm2, identical)])
        return model, [MSELoss(), None], [1, None]
