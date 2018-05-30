import torch
from torch.autograd import Variable
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from collections import OrderedDict
from sys import stdout
from graphviz import Digraph
from itertools import accumulate
import inspect

activations = {
    None: lambda x: x,
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'leaky_relu': F.leaky_relu,
    'logsigmoid': F.logsigmoid,
    'elu': F.elu,
    'selu': F.selu,
    'glu': F.glu,
    'softshrink': F.softshrink,
    'hardshrink': F.hardshrink,
    'tanhshrink': F.tanhshrink,
    'softplus': F.softplus,
    'prelu': F.prelu,
    'softsign': F.softsign,
    'rrelu': F.rrelu,
    'hardtanh': F.hardtanh,
    'softmin': F.softmin,
    'softmax': F.softmax,
    'log_softmax': F.log_softmax,
}


class ModelError(ValueError):
    def __init__(self, message, expected=None, received=None):
        self.msg = "{}.{}: {}\nExpected:{}\nReceived:{}".format(
            inspect.stack()[2][0].f_locals["self"].__class__.__name__,
            inspect.stack()[2][0].f_code.co_name,
            message, expected, received)

    def __str__(self):
        return self.msg


class Size:
    def __init__(self, size, name, is_single=True):
        if not isinstance(name, str):
            raise ModelError("size name should be a string", received=name)
        if not isinstance(is_single, bool):
            raise ModelError("size is_single flag should be a bool", received=is_single)
        self.name = name
        self.is_single = is_single
        self.size = self.validate(size)

    def validate(self, size):
        format_warning = "{} must be a list of tuples of ints and ints, " \
                         "or a tuple of ints, or an int.".format(self.name)
        type_warning = "all ints inside {} mush be positive.".format(self.name)
        sample_warning = "there can only be one single tensor size in {}".format(self.name)

        def validate_tuple(t):
            if isinstance(t, int):
                return tuple([validate_int(t)])
            elif isinstance(t, tuple):
                return tuple([validate_int(i) for i in t])
            else:
                raise ModelError(format_warning, expected="list of tuples, tuple, or int", received=size)

        def validate_int(i):
            if not isinstance(i, int):
                raise ModelError(format_warning, expected="int", received="{} in {}".format(i, size))
            if i <= 0:
                raise ModelError(type_warning, expected="non-negative int", received="{} in {}".format(i, size))
            return i

        if isinstance(size, Size):
            return size.size
        if not isinstance(size, (list, tuple, int)):
            raise ModelError(format_warning, expected="list of tuples, tuple, or int", received=size)
        if isinstance(size, list):
            if len(size) > 1 and self.is_single:
                raise ModelError(sample_warning, expected="single size (one tuple)", received=size)
            return [validate_tuple(t) for t in size]
        else:
            return [validate_tuple(size)]

    def equal(self, size):
        if not len(self.size) == len(size.size):
            return False
        for t1, t2 in zip(self.size, size.size):
            if not (len(t1) == len(t2) and all([n1 == n2 for n1, n2 in zip(t1, t2)])):
                return False
        return True

    @classmethod
    def merge(cls, list_of_sizes):
        return Size([t for size in list_of_sizes for t in size.size],
                    "+".join([size.name for size in list_of_sizes]),
                    is_single=False)

    def check(self, xs):
        if not (len(xs) == len(self.size) and all([x.size()[1:] == s for x, s in zip(xs, self.size)])):
            raise ModelError("encounter tensor whose shape does not match {}".format(self.name),
                             [tuple(x.size()[1:]) for x in xs], self.size)

    def len(self):
        return len(self.size)

    def dim(self):
        if self.is_single:
            return len(self.size[0])
        else:
            return [len(s) for s in self.size]


class Abstract(Module):
    __count = 0

    def __init__(self, in_size, out_size, single_io=False):
        super(Abstract, self).__init__()
        self._count()
        self.name = "{}_{}".format(self.__class__.__name__, self.__count)
        self.in_size = Size(in_size, "{}.in_size".format(self.name), single_io)
        self.out_size = Size(out_size, "{}.out_size".format(self.name), single_io)
        self.layers = ModuleList()
        # TODO: Design the attribute structure and co-operate with the parameter object
        self.attributes = {}

    def forward(self, *xs):
        return xs

    def apply_layer(self, x, activation):
        if activation not in activations:
            raise ValueError("{}: Invalid activation type {}.".format(self.net_type, activation))
        return activations[activation](x)

    @classmethod
    def _count(cls):
        cls.__count += 1
        return cls.__count

    def _gen_input(self):
        return tuple([Variable(torch.randn(1, *t)) for t in self.in_size.size])

    def summary(self, raw=False, buffer=stdout):
        def register_hook(module):
            def hook(mod, x, y):
                if mod._modules:  # only want base layers
                    return
                class_name = str(mod.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(x[0].size())
                summary[m_key]['input_shape'][0] = None
                if y.__class__.__name__ == 'tuple':
                    summary[m_key]['output_shape'] = list(y[0].size())
                else:
                    summary[m_key]['output_shape'] = list(y.size())
                summary[m_key]['output_shape'][0] = None

                params = 0
                # iterate through parameters and count num params
                for name, p in mod._parameters.items():
                    params += torch.numel(p.data)
                    summary[m_key]['trainable'] = p.requires_grad

                summary[m_key]['nb_params'] = params

            if not isinstance(module, torch.nn.Sequential) and \
               not isinstance(module, torch.nn.ModuleList) and \
               not (module == self):
                hooks.append(module.register_forward_hook(hook))

        def get_names(module, name, acc):
            if not module._modules:
                acc.append(name)
            else:
                for key in module._modules.keys():
                    p_name = key if name == "" else name + "." + key
                    get_names(module._modules[key], p_name, acc)

        def crop(s):
            return s[:col_width] if len(s) > col_width else s

        if raw:
            print(self, file=buffer)
        else:
            summary = OrderedDict()
            hooks = []
            self.apply(register_hook)
            self(*self._gen_input())
            for h in hooks:
                h.remove()
            names = []
            get_names(self, "", names)
            col_width = 25  # should be >= 12
            summary_width = 61
            print('_' * summary_width, file=buffer)
            print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
                'Layer (type)', 'Output Shape', 'Param #', col_width), file=buffer)
            print('=' * summary_width, file=buffer)
            total_params = 0
            trainable_params = 0
            for (i, l_type), l_name in zip(enumerate(summary), names):
                d = summary[l_type]
                total_params += d['nb_params']
                if 'trainable' in d and d['trainable']:
                    trainable_params += d['nb_params']
                print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
                    crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
                    crop(str(d['nb_params'])), col_width), file=buffer)
                if i < len(summary) - 1:
                    print('_' * summary_width, file=buffer)
            print('=' * summary_width, file=buffer)
            print('Total params: ' + str(total_params), file=buffer)
            print('Trainable params: ' + str(trainable_params), file=buffer)
            print('Non-trainable params: ' + str((total_params - trainable_params)), file=buffer)
            print('_' * summary_width, file=buffer)

    def dot(self):
        def size_to_str(size):
            return '(' + ', '.join(['%d' % v for v in size]) + ')'

        def add_nodes(fn):
            if fn not in seen:
                if torch.is_tensor(fn):
                    dot.node(str(id(fn)), size_to_str(fn.size()), fillcolor='orange')
                elif hasattr(fn, 'variable'):
                    u = fn.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(fn)), node_name, fillcolor='lightblue')
                else:
                    dot.node(str(id(fn)), str(type(fn).__name__))
                seen.add(fn)
                if hasattr(fn, 'next_functions'):
                    for u in fn.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(fn)))
                            add_nodes(u[0])
                if hasattr(fn, 'saved_tensors'):
                    for t in fn.saved_tensors:
                        dot.edge(str(id(t)), str(id(fn)))
                        add_nodes(t)

        ys = self(*self._gen_input())
        params = dict(self.named_parameters())
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()
        for y in ys:
            add_nodes(y.grad_fn)
        return dot


class Block(Abstract):
    def __init__(self, net_list):
        self.net_list = Block.validate_net_list(net_list)
        super(Block, self).__init__(Size.merge([n.in_size for n in self.net_list[0]]),
                                    Size.merge([n.out_size for n in self.net_list[-1]]))
        self.validate_size_alignment()
        for t in self.net_list:
            for n in t:
                self.layers.extend(list(n.layers))

    def forward(self, *xs):
        for t in self.net_list:
            Size.merge([n.in_size for n in t]).check(xs)
            xs = tuple([ys for xs_chunk, n in Block.data_split(xs, t) for ys in n(*xs_chunk)])
            Size.merge([n.out_size for n in t]).check(xs)
        return xs
    
    @staticmethod
    def data_split(xs, net_tuple):
        split_idx = [0] + list(accumulate([n.in_size.len() for n in net_tuple]))
        xs_chunks = tuple([xs[split_idx[i]:split_idx[i+1]] for i in range(len(net_tuple))])
        return zip(xs_chunks, net_tuple)
        
    @staticmethod
    def validate_net_list(net_list):
        format_warning = "net_list must be a list of tuples of nets " \
                         "and nets, or a tuple of nets."
        type_warning = "All nets inside net_list must be an " \
                       "instance of a subclass of AbstractNet."

        def validate_tuple(t):
            if not isinstance(t, tuple):
                return tuple([validate_net(t)])
            else:
                return tuple([validate_net(n) for n in t])

        def validate_net(n):
            if not issubclass(type(n), Abstract):
                raise ModelError(type_warning, expected="subclass of Abstract", received=type(n))
            return n

        if not isinstance(net_list, (list, tuple)):
            raise ModelError(format_warning, expected="list of tuple", received=net_list)
        elif isinstance(net_list, list):
            return [validate_tuple(t) for t in net_list]
        else:
            return [validate_tuple(net_list)]

    def validate_size_alignment(self):
        for i in range(len(self.net_list) - 1):
            out_size = Size.merge([n.out_size for n in self.net_list[i]])
            in_size = Size.merge([n.in_size for n in self.net_list[i+1]])
            if not out_size.equal(in_size):
                raise ModelError("{} does not match {}".format(in_size.name, out_size.name),
                                 out_size.size, in_size.size)


class Reshape(Abstract):
    def __init__(self, in_size, out_size):
        super(Reshape, self).__init__(in_size, out_size, True)

    def forward(self, *xs):
        self.in_size.check(xs)
        xs = tuple([xs[0].view(-1, *(self.out_size.size[0]))])
        self.out_size.check(xs)
        return xs


class Split(Abstract):
    def __init__(self, in_size, axis, split_size_or_sections):
        if not (isinstance(axis, int) and axis >= 1):
            raise ValueError("{}: axis must be a positive integer.".format(self.__class__.__name__))
        self.axis = axis
        in_size = Abstract.validate_size(in_size, 'in_size', False)
        if not len(in_size) == 1:
            raise ValueError("{}: only singe input is supported.".format(self.__class__.__name__))
        in_size = in_size[0]
        if axis > len(in_size):
            raise ValueError("{}: input does not have indicated axis.".format(self.__class__.__name__))
        if isinstance(split_size_or_sections, int):
            if in_size[axis-1] % split_size_or_sections != 0:
                raise ValueError("{}: indicated number of chunks does not divide "
                                 "the input length on that axis.".format(self.__class__.__name__))
            self.split_idx = [i*split_size_or_sections for i in range(in_size[axis-1]//split_size_or_sections+1)]
            out_size = [tuple([s if a != (axis-1) else split_size_or_sections for a, s in enumerate(in_size)])] * \
                       (in_size[axis-1]//split_size_or_sections)
            super(Split, self).__init__(self.__class__.__name__, in_size, out_size)
        elif isinstance(split_size_or_sections, list) and \
                all([isinstance(s, int) for s in split_size_or_sections]):
            if in_size[axis-1] != sum(split_size_or_sections):
                raise ValueError("{}: sum of indicated chuck sizes does not match "
                                 "the input length on that axis.".format(self.__class__.__name__))
            self.split_idx = [0] + list(accumulate(split_size_or_sections))
            out_size = [tuple([s if a != (axis-1) else ss for a, s in enumerate(in_size)]) for ss in split_size_or_sections]
            super(Split, self).__init__(in_size, out_size)
        else:
            raise ValueError("{}: unsupported type of split_size_or_sections, "
                             "it must be an int of a list of ints.".format(self.__class__.__name__))

    def forward(self, *xs):
        self.in_size.check(xs)
        x = xs[0]
        xs = tuple([x.narrow(self.axis, self.split_idx[i], self.split_idx[i+1]-self.split_idx[i])
                    for i in range(len(self.out_size.size))])
        self.out_size.check(xs)
        return xs


class Cat(Abstract):
    def __init__(self, in_size, axis):
        if not (isinstance(axis, int) and axis > 0):
            raise ValueError("{}: axis must be a positive integer.".format(self.__class__.__name__))
        self.axis = axis
        in_size = Abstract.validate_size(in_size, 'in_size', False)
        if not len(in_size) > 1:
            raise ValueError("{}: only multiple inputs are supported.".format(self.__class__.__name__))
        if not all([len(ins) == len(in_size[0]) and
                    all([ins[i] == in_size[0][i] for i in range(len(in_size[0]))
                         if i != (axis-1)]) for ins in in_size]):
            raise ValueError("{}: input sizes must match on axes except "
                             "from the indicated axis.".format(self.__class__.__name__))
        out_size = [tuple([s if a != (axis-1) else sum([ins[axis-1] for ins in in_size])
                           for a, s in enumerate(in_size[0])])]
        super(Cat, self).__init__(in_size, out_size)

    def forward(self, *xs):
        self.in_size.check(xs)
        x = torch.cat(tuple([x.contiguous() for x in xs]), dim=self.axis)
        xs = tuple([x])
        self.out_size.check(xs)
        return xs


class Identical(Abstract):
    def __init__(self, in_size):
        super(Identical, self).__init__(in_size, in_size)


class Repeat(Abstract):
    def __init__(self, in_size, num_of_copies):
        if not (isinstance(num_of_copies, int) and num_of_copies > 0):
            raise ValueError("{}: num_of_copies must be a positive integer.".format(self.__class__.__name__))
        self.num_of_copies = num_of_copies
        super(Repeat, self).__init__(in_size, in_size)
        if self.in_size.len() > 1:
            raise ValueError("{}: only single input is supported.".format(self.__class__.__name__))
        self.out_size = Size.merge([self.in_size] * num_of_copies)

    def forward(self, *xs):
        self.in_size.check(xs)
        xs = xs * self.num_of_copies
        self.out_size.check(xs)
        return xs


class SwapAxes(Abstract):
    def __init__(self, in_size, dim0, dim1):
        if not (dim0 != dim1 and dim0 >= 0 and dim1 >= 0):
            raise ModelError("dim0 and dim1 should be two different non-negative integers.",
                             "dim0:{}, dim1:{}".format(dim0, dim1))
        self.dim0 = dim0
        self.dim1 = dim1
        in_size = Abstract.validate_size(in_size, 'in_size', False)
        if len(in_size) > 1:
            raise ValueError("{}: only single input is supported.".format(self.__class__.__name__))
        if len(in_size[0]) <= max(dim0, dim1):
            raise ValueError("{}:\n"
                             "Expected: dimension of in_size should be larger than the two axes to swap.\n"
                             "Received: dimension of in_size: {}".format(self.__class__.__name__, len(in_size[0])))
        out_size = list(in_size[0])
        tmp = out_size[dim0]
        out_size[dim0] = out_size[dim1]
        out_size[dim1] = tmp
        out_size = [tuple(out_size)]
        super(SwapAxes, self).__init__(in_size, out_size)

    def forward(self, *xs):
        self.in_size.check(xs)
        xs = tuple([xs[0].transpose(self.dim0+1, self.dim1+1).contiguous()])
        self.out_size.check(xs)
        return xs
