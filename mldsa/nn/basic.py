import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from .util import *
from .abstract import Abstract


class FC(Abstract):
    def __init__(self, in_size, out_size, hidden_dims, hidden_activation='relu', out_activation='relu'):
        super(FC, self).__init__(in_size, out_size, True)
        check_list_of_positive_ints(self.name, hidden_dims, 'hidden_dims')
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        dims = [self.in_size.size[0][-1]] + hidden_dims + [self.out_size.size[0][-1]]
        for i in range(len(hidden_dims)+1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            # TODO: A better way of applying initialization to models
            xavier_normal_(self.layers[-1].weight.data)

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        for l in list(self.layers)[:-1]:
            x = self.apply_layer(l(x), self.hidden_activation)
        x = self.apply_layer(self.layers[-1](x), self.out_activation)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class Conv1d(Abstract):
    def __init__(self, in_size, out_size, hidden_channels, kernel_sizes, strides=None, same_length=True, paddings=None,
                 hidden_activation='relu', out_activation=None):
        super(Conv1d, self).__init__(in_size, out_size, True)
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        check_list_of_positive_ints(self.name, hidden_channels, 'hidden_channels')
        check_list_of_positive_ints(self.name, kernel_sizes, 'kernel_sizes')
        if strides is None:
            strides = [1] * len(kernel_sizes)
        if same_length:
            if paddings is not None:
                raise ValueError("{}: same_length mode cannot be used when paddings are indicated.".format(self.name))
            if not all([s % 2 == 1 for s in kernel_sizes]):
                raise ValueError("{}: same_length mode only support odd kernel sizes.".format(self.name))
            paddings = [s // 2 for s in kernel_sizes]
        else:
            if not paddings:
                paddings = [0] * len(kernel_sizes)
        check_list_of_positive_ints(self.name, strides, 'strides')
        check_list_of_non_negative_ints(self.name, paddings, 'paddings')
        if not (len(hidden_channels) == len(kernel_sizes)-1 and len(kernel_sizes) == len(strides)
                and len(strides) == len(paddings)):
            raise ValueError("{}: kernel_sizes, strides, and paddings should have the same length, "
                             "hidden_channels should be 1 element shorter.".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        channels = [self.in_size.size[0][0]] + hidden_channels + [self.out_size.size[0][0]]
        for i in range(len(kernel_sizes)):
            self.layers.append(torch.nn.Conv1d(channels[i], channels[i+1], kernel_sizes[i], strides[i], paddings[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        for l in list(self.layers)[:-1]:
            x = self.apply_layer(l(x), self.hidden_activation)
        x = self.apply_layer(self.layers[-1](x), self.out_activation)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class Conv2d(Abstract):
    """
    Conv2d
    Arch: [In(>=3)]---[Conv1d] x A---[Out(>=3)]
    """
    def __init__(self, in_size, out_size, hidden_channels, kernel_sizes, strides=None, same_length=True, paddings=None,
                 hidden_activation='relu', out_activation=None):
        super(Conv2d, self).__init__(in_size, out_size, True)
        if not (self.in_size.dim() == 3 and self.out_size.dim() == 3):
            raise ValueError("{}: expect input and output to be of shape (C, W, H).".format(self.name))
        check_list_of_positive_ints(self.name, hidden_channels, 'hidden_channels')
        check_list_of_positive_bi_tuples(self.name, kernel_sizes, 'kernel_sizes')
        if strides is None:
            strides = [(1, 1)] * len(kernel_sizes)
        if same_length:
            if paddings is not None:
                raise ValueError("{}: same_length mode cannot be used when paddings is indicated.".format(self.name))
            if not all([s[0] % 2 == 1 and s[1] % 2 == 1 for s in kernel_sizes]):
                raise ValueError("{}: same_length mode only support odd kernel sizes.".format(self.name))
            paddings = [(s[0] // 2, s[1] // 2) for s in kernel_sizes]
        else:
            if not paddings:
                paddings = [(0, 0)] * len(kernel_sizes)
        check_list_of_positive_bi_tuples(self.name, strides, 'strides')
        check_list_of_non_negative_bi_tuples(self.name, paddings, 'paddings')
        if not (len(hidden_channels) == len(kernel_sizes)-1 and len(kernel_sizes) == len(strides)
                and len(strides) == len(paddings)):
            raise ValueError("{}: kernel_sizes, strides, and paddings should have the same length, "
                             "hidden_channels should be 1 element shorter.".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        channels = [self.in_size[0][0]] + hidden_channels + [self.out_size[0][0]]
        for i in range(len(kernel_sizes)):
            self.layers.append(torch.nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], strides[i], paddings[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        for l in list(self.layers)[:-1]:
            x = self.apply_layer(l(x), self.hidden_activation)
        x = self.apply_layer(self.layers[-1](x), self.out_activation)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class ConvTranspose1d(Abstract):
    """
    ConvTranspose1d
    Arch: [In(>=2)]---[Conv1d] x A---[Out(>=2)]
    """
    def __init__(self, in_size, out_size, hidden_channels, kernel_sizes, strides=None, same_length=True, paddings=None,
                 hidden_activation='relu', out_activation=None):
        super(ConvTranspose1d, self).__init__(in_size, out_size, True)
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        check_list_of_positive_ints(self.name, hidden_channels, 'hidden_channels')
        check_list_of_positive_ints(self.name, kernel_sizes, 'kernel_sizes')
        if strides is None:
            strides = [1] * len(kernel_sizes)
        if same_length:
            if paddings is not None:
                raise ValueError("{}: same_length mode cannot be used when paddings is indicated.".format(self.name))
            if not all([s % 2 == 1 for s in kernel_sizes]):
                raise ValueError("{}: same_length mode only support odd kernel sizes.".format(self.name))
            paddings = [s // 2 for s in kernel_sizes]
        else:
            if not paddings:
                paddings = [0] * len(kernel_sizes)
        check_list_of_positive_ints(self.name, strides, 'strides')
        check_list_of_non_negative_ints(self.name, paddings, 'paddings')
        if not (len(hidden_channels) == len(kernel_sizes)-1 and len(kernel_sizes) == len(strides)
                and len(strides) == len(paddings)):
            raise ValueError("{}: kernel_sizes, strides, and paddings should have the same length, "
                             "hidden_channels should be 1 element shorter.".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        channels = [self.in_size.size[0][0]] + hidden_channels + [self.out_size.size[0][0]]
        for i in range(len(kernel_sizes)):
            self.layers.append(torch.nn.ConvTranspose1d(channels[i], channels[i+1], kernel_sizes[i], strides[i], paddings[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        for l in list(self.layers)[:-1]:
            x = self.apply_layer(l(x), self.hidden_activation)
        x = self.apply_layer(self.layers[-1](x), self.out_activation)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class ConvTranspose2d(Abstract):
    """
    ConvTranspose1d
    Arch: [In(>=2)]---[Conv1d] x A---[Out(>=2)]
    """
    def __init__(self, in_size, out_size, hidden_channels, kernel_sizes, strides=None, same_length=True, paddings=None,
                 hidden_activation='relu', out_activation=None):
        super(ConvTranspose2d, self).__init__(in_size, out_size, True)
        if not (self.in_size.dim() == 3 and self.out_size.dim() == 3):
            raise ValueError("{}: expect input and output to be of shape (C, H, W).".format(self.name))
        check_list_of_positive_ints(self.name, hidden_channels, 'hidden_channels')
        check_list_of_positive_bi_tuples(self.name, kernel_sizes, 'kernel_sizes')
        if strides is None:
            strides = [(1, 1)] * len(kernel_sizes)
        if same_length:
            if paddings is not None:
                raise ValueError("{}: same_length mode cannot be used when paddings is indicated.".format(self.name))
            if not all([s[0] % 2 == 1 and s[1] % 2 == 1 for s in kernel_sizes]):
                raise ValueError("{}: same_length mode only support odd kernel sizes.".format(self.name))
            paddings = [(s[0] // 2, s[1] // 2) for s in kernel_sizes]
        else:
            if not paddings:
                paddings = [(0, 0)] * len(kernel_sizes)
        check_list_of_positive_bi_tuples(self.name, strides, 'strides')
        check_list_of_non_negative_bi_tuples(self.name, paddings, 'paddings')
        if not (len(hidden_channels) == len(kernel_sizes)-1 and len(kernel_sizes) == len(strides)
                and len(strides) == len(paddings)):
            raise ValueError("{}: kernel_sizes, strides, and paddings should have the same length, "
                             "hidden_channels should be 1 element shorter.".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        channels = [self.in_size[0][0]] + hidden_channels + [self.out_size[0][0]]
        for i in range(len(kernel_sizes)):
            self.layers.append(torch.nn.ConvTranspose2d(channels[i], channels[i+1], kernel_sizes[i], strides[i], paddings[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        for l in list(self.layers)[:-1]:
            x = self.apply_layer(l(x), self.hidden_activation)
        x = self.apply_layer(self.layers[-1](x), self.out_activation)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class MaxPool1d(Abstract):
    def __init__(self, in_size, out_size, kernel_size=None, stride=2, padding=0):
        super(MaxPool1d, self).__init__(in_size, out_size, True)
        check_positive_int(self.name, stride, "stride")
        if kernel_size is None:
            kernel_size = stride
        else:
            check_positive_int(self.name, kernel_size, "kernel_size")
        check_non_negative_int(self.name, padding, "padding")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.layers.append(torch.nn.MaxPool1d(kernel_size, stride, padding=padding))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class MaxPool2d(Abstract):
    def __init__(self, in_size, out_size, kernel_size, stride=(2, 2), padding=(0, 0)):
        super(MaxPool2d, self).__init__(in_size, out_size, True)
        check_positive_bi_tuple(self.name, stride, "stride")
        if kernel_size is None:
            kernel_size = stride
        else:
            check_positive_bi_tuple(self.name, kernel_size, "kernel_size")
        check_non_negative_bi_tuple(self.name, padding, "padding")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, H, W).".format(self.name))
        self.layers.append(torch.nn.MaxPool2d(kernel_size, stride, padding=padding))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class AvgPool1d(Abstract):
    def __init__(self, in_size, out_size, kernel_size, stride=None, padding=0):
        super(AvgPool1d, self).__init__(in_size, out_size, True)
        check_positive_int(self.name, stride, "stride")
        if kernel_size is None:
            kernel_size = stride
        else:
            check_positive_int(self.name, kernel_size, "kernel_size")
        check_non_negative_int(self.name, padding, "padding")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.layers.append(torch.nn.AvgPool1d(kernel_size, stride, padding=padding))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class AvgPool2d(Abstract):
    def __init__(self, in_size, out_size, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__(in_size, out_size, True)
        check_positive_bi_tuple(self.name, stride, "stride")
        if kernel_size is None:
            kernel_size = stride
        else:
            check_positive_bi_tuple(self.name, kernel_size, "kernel_size")
        check_non_negative_bi_tuple(self.name, padding, "padding")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, H, W).".format(self.name))
        self.layers.append(torch.nn.AvgPool2d(kernel_size, stride, padding=padding))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class ConstantPad1d(Abstract):
    def __init__(self, in_size, out_size, padding, value=0):
        super(ConstantPad1d, self).__init__(in_size, out_size, True)
        check_positive_int(self.name, padding, "padding")
        check_int(self.name, value, "value")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.layers.append(torch.nn.ConstantPad1d(padding, value))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class ConstantPad2d(Abstract):
    def __init__(self, in_size, out_size, padding, value=0):
        super(ConstantPad2d, self).__init__(in_size, out_size, True)
        check_positive_bi_tuple(self.name, padding, "padding")
        check_int(self.name, value, "value")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, H, W).".format(self.name))
        self.layers.append(torch.nn.ConstantPad2d(padding, value))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class BatchNorm1d(Abstract):
    def __init__(self, in_size, out_size):
        super(BatchNorm1d, self).__init__(in_size, out_size, True)
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.layers.append(torch.nn.BatchNorm1d(in_size[0][0]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class LSTM(Abstract):
    def __init__(self, in_size, out_size, hidden_sizes, dropouts, bidirectional=False,
                 hidden_activation='relu', out_activation=None):
        super(LSTM, self).__init__(in_size, out_size, True)
        check_list_of_positive_ints(self.name, hidden_sizes, "hidden_sizes")
        if isinstance(dropouts, float) or isinstance(dropouts, int):
            dropouts = [dropouts] * (len(hidden_sizes)+1)
        check_list_of_weights(self.name, dropouts, "dropouts")
        self.dropouts = dropouts
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        sizes = [self.in_size.size[0][0]] + hidden_sizes + [self.out_size.size[0][0]]
        self.hs = [None] * len(dropouts)
        self.cs = [None] * len(dropouts)
        for i in range(len(dropouts)):
            self.layers.append(torch.nn.LSTM(sizes[i], sizes[i+1], num_layers=1, bias=True, batch_first=True,
                                             dropout=0, bidirectional=bidirectional))
            self.layers.append(torch.nn.Dropout(dropouts[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = x.transpose(1, 2)
        for i, l in enumerate(list(self.layers)):
            if i % 2 == 0:
                if self.hs[i//2] is None and self.cs[i//2] is None:
                    x, (h, c) = l(x)
                else:
                    x, (h, c) = l(x, (self.hs[i//2], self.cs[i//2]))
                self.hs[i//2] = h
                self.cs[i//2] = c
            else:
                x = self.apply_layer(l(x), self.hidden_activation if i < (len(self.layers)-1) else self.out_activation)
        x = x.transpose(1, 2)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class GRU(Abstract):
    def __init__(self, in_size, out_size, hidden_sizes, dropouts, bidirectional=False,
                 hidden_activation='relu', out_activation=None):
        super(GRU, self).__init__(in_size, out_size, True)
        check_list_of_positive_ints(self.name, hidden_sizes, "hidden_sizes")
        if isinstance(dropouts, float) or isinstance(dropouts, int):
            dropouts = [dropouts] * (len(hidden_sizes)+1)
        check_list_of_weights(self.name, dropouts, "dropouts")
        self.dropouts = dropouts
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        sizes = [self.in_size.size[0][0]] + hidden_sizes + [self.out_size.size[0][0]]
        self.hs = [None] * len(dropouts)
        for i in range(len(dropouts)):
            self.layers.append(torch.nn.GRU(sizes[i], sizes[i+1], num_layers=1, bias=True, batch_first=True,
                                            dropout=0, bidirectional=bidirectional))
            self.layers.append(torch.nn.Dropout(dropouts[i]))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = x.transpose(1, 2)
        for i, l in enumerate(list(self.layers)):
            if i % 2 == 0:
                if self.hs[i//2] is None:
                    x, h = l(x)
                else:
                    x, h = l(x, self.hs[i//2])
                self.hs[i//2] = h
            else:
                x = self.apply_layer(l(x), self.hidden_activation if i < (len(self.layers)-1) else self.out_activation)
        x = x.transpose(1, 2)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class Dropout1d(Abstract):
    def __init__(self, in_size, out_size, p):
        super(Dropout1d, self).__init__(in_size, out_size, True)
        check_weight(self.name, p, "dropout")
        if not (self.in_size.dim() == 2 and self.out_size.dim() == 2):
            raise ValueError("{}: expect input and output to be of shape (C, L).".format(self.name))
        self.layers.append(torch.nn.Dropout(p))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class Dropout2d(Abstract):
    def __init__(self, in_size, out_size, p):
        super(Dropout2d, self).__init__(in_size, out_size, True)
        check_weight(self.name, p, "dropout")
        if not (self.in_size.dim() == 3 and self.out_size.dim() == 3):
            raise ValueError("{}: expect input and output to be of shape (C, H, W).".format(self.name))
        self.layers.append(torch.nn.Dropout2d(p))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        x = self.apply_layer(list(self.layers)[0](x), None)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs
