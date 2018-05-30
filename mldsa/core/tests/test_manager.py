import pytest
from ..manager import Manager

test_paras_dict = {
    'config': {
        'task_name': 'test',
        'data_dir': None,
        'result_dir': None,
        'suffix': '_test',
    },
    'data': {
        'temporal_dim': 1,
        'contextual_dim': None,
        'feature_dim': 2,
        'label_dim': 1,
        'process_method': 'temporal',
        'split_method': 'cross_validation',
        'feed_method': 'predict',
    },
    'model': {
        'type': 'predictor',
        'architecture': 'fc',
        'fc_hidden_dims': [],
        'fc_activation': 'relu',
        'out_activation': 'sigmoid',
        'regularization': 0,
        'dropout_prob': 0,
    },
    'train': {
        'nepochs': 1,
        'optimizer': 'Adam',
        'batch_size': 1,
        'learning_rate': 0.001,
        'decay': 0,
        'trial': iter(range(2)),
    },
    'log': {
        'metric': {},
        'callback': [],
        'verbosity': 0,
    },
}


class TestManager:
    @pytest.fixture(scope='class', autouse=True)
    def setup(self, tmpdir_factory):
        import numpy as np
        x = np.random.rand(2, 2, 2)
        y = np.random.randint(2, size=(2, 2))
        data_dir = tmpdir_factory.mktemp('data')
        result_dir = tmpdir_factory.mktemp('result')
        x_path = data_dir.join('x_test.npy')
        y_path = data_dir.join('y_test.npy')
        np.save(str(x_path), x)
        np.save(str(y_path), y)
        test_paras_dict['config']['data_dir'] = str(data_dir)+'/'
        test_paras_dict['config']['result_dir'] = str(result_dir)+'/'

    def test_manager_stdout(self, capsys):
        manager = Manager(test_paras_dict)
        manager()
        captured = capsys.readouterr()
        assert captured.out == "Builder: Predictor model fc is constructed.\n" \
                               "Loader: Data loaded:\n" \
                               " - suffix: _test - feature shape: (2, 2, 2) - label shape: (2, 2)\n" \
                               "Logger: selected verbosity 0.\n" \
                               "Trainer: Training started.\n" \
                               "Trainer: Training finished.\n" \
                               "Recoder: results and model weights saved to {}.\n" \
                               "".format(test_paras_dict['config']['result_dir'] + 'test/')
        assert captured.err == ""

    def test_manager_result(self):
        import os
        result_dir = test_paras_dict['config']['result_dir']
        assert os.path.isdir(result_dir + 'test/')
        assert os.path.exists(result_dir + 'test/' + 'main.record')
        assert os.path.exists(result_dir + 'test/' + 'predictor_fc_train:trial:0.model')
        assert os.path.exists(result_dir + 'test/' + 'predictor_fc_train:trial:1.model')
