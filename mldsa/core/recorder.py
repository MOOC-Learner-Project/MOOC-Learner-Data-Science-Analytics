import os
import pickle
import torch


class Recorder:
    def __init__(self, paras):
        self.paras = paras
        self.iter_keys = paras.get_iter_keys()
        self.iter_lists = paras.get_iter_lists()
        self.iter_types = Recorder.get_parameter_type(self.iter_lists)
        self.cur_para_list = []
        self.result_list = []

    @staticmethod
    def check_numerical_variable(v):
        import numbers
        return isinstance(v, numbers.Number)

    @staticmethod
    def get_parameter_type(para_list):
        return {k: all([Recorder.check_numerical_variable(v) for v in l]) for k, l in para_list.items()}

    def __call__(self, paras, results):
        self.cur_para_list.append(paras.get_cur_iter_paras())
        self.result_list.append(results)

    @staticmethod
    def save(recorder):
        directory = recorder.paras.config.result_dir + recorder.paras.config.task_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for cur_para, results in zip(recorder.cur_para_list, recorder.result_list):
            torch.save(results.pop_model().state_dict(),
                       directory + '_'.join([recorder.paras.model.type, recorder.paras.model.architecture]
                                            + [k + ':' + str(v) for k, v in cur_para.items()]) + '.model')
        recorder.paras.destroy()
        with open(directory + 'main.record', 'wb') as handle:
            pickle.dump(recorder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Recoder: results and model weights saved to {}.".format(directory))

    @staticmethod
    def load(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)
