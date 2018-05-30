def check_int(net_name, x, name):
    if not isinstance(x, int):
        raise ValueError("{}: {} should be an int".format(net_name, name))


def check_weight(net_name, x, name):
    if not (0 <= x <= 1):
        raise ValueError("{}: {} should be an int".format(net_name, name))


def check_positive_int(net_name, x, name):
    if not (isinstance(x, int) and x > 0):
        raise ValueError("{}: {} should be a positive int".format(net_name, name))


def check_non_negative_int(net_name, x, name):
    if not (isinstance(x, int) and x >= 0):
        raise ValueError("{}: {} should be a non-negative int".format(net_name, name))


def check_positive_bi_tuple(net_name, t, name):
    if not (isinstance(t, tuple) and len(t) == 2 and all([isinstance(x, int) and x > 0 for x in t])):
        raise ValueError("{}: {} should be a bi-tuple of positive ints".format(net_name, name))


def check_non_negative_bi_tuple(net_name, t, name):
    if not (isinstance(t, tuple) and len(t) == 2 and all([isinstance(x, int) and x >= 0 for x in t])):
        raise ValueError("{}: {} should be a bi-tuple of non-negative ints".format(net_name, name))


def check_list_of_positive_ints(net_name, l, name):
    if not (isinstance(l, list) and all([isinstance(x, int) and x > 0 for x in l])):
        raise ValueError("{}: {} should be a list of positive ints".format(net_name, name))


def check_list_of_non_negative_ints(net_name, l, name):
    if not (isinstance(l, list) and all([isinstance(x, int) and x >= 0 for x in l])):
        raise ValueError("{}: {} should be a list of non-negative ints".format(net_name, name))


def check_list_of_weights(net_name, l, name):
    if not (isinstance(l, list) and all([w >=0 and w <= 1 for w in l])):
        raise ValueError("{}: {} should be a list of weights between 0 and 1".format(net_name, name))


def check_list_of_positive_bi_tuples(net_name, l, name):
    if not (isinstance(l, list) and
            all([(isinstance(xs, tuple) and len(xs) == 2 and all([isinstance(x, int) and x > 0 for x in xs]))
                 for xs in l])):
        raise ValueError("{}: {} should be a list of bi-tuples of positive ints".format(net_name, name))


def check_list_of_non_negative_bi_tuples(net_name, l, name):
    if not (isinstance(l, list) and
            all([(isinstance(xs, tuple) and len(xs) == 2 and all([isinstance(x, int) and x >= 0 for x in xs]))
                 for xs in l])):
        raise ValueError("{}: {} should be a list of bi-tuples of positive ints".format(net_name, name))
