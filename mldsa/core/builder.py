

class Builder:
    def __init__(self, build_method, build_options):
        assert build_method in build_options, \
            "Builder: invalid build method {}, valid options are {}." \
            "".format(build_method, build_options)
        self.build_method = build_method
        self.build_options = build_options
        print("Builder: {} model {} is constructed.".format(self.__class__.__name__, build_method))

    def __call__(self, paras):
        return self.build_options[self.build_method](paras)
