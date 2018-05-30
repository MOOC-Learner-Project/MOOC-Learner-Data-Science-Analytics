import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA


class CallBack:
    def __init__(self, callback):
        self.callback_options = {
            'learning_curve': CallBack.learning_curve,
            'pca2d': CallBack.pca2d,
        }
        assert callback in self.callback_options, \
            "{}: invalid callback {}, valid options are {}." \
            "".format(self.__class__.__name__, callback, self.callback_options)
        self.callback = callback

    def __call__(self, paras, results):
        self.callback_options[self.callback](paras, results)

    @staticmethod
    def learning_curve(paras, results):
        loss = results.get_all_loss()
        metrics = results.get_all_metric()
        xrange = range(1, len(loss)+1)
        nrow = len(metrics)//2+1
        ncol = 2
        fig = plt.figure(figsize=(15, 4*(len(metrics)//2+1)))
        ax = fig.add_subplot(nrow, ncol, 1)
        ax.plot(xrange, [l[0] for l in loss])
        ax.plot(xrange, [l[1] for l in loss])
        ax.set_title('loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'tests'], loc='upper left')
        for i, k in enumerate(metrics):
            ax = fig.add_subplot(nrow, ncol, i+2)
            ax.plot(xrange, [m[0] for m in metrics[k]])
            ax.plot(xrange, [m[1] for m in metrics[k]])
            ax.set_title(k)
            ax.set_ylabel(k)
            ax.set_xlabel('epoch')
            ax.legend(['train', 'tests'], loc='upper left')
        plt.show()

    @staticmethod
    def pca2d(paras, results):
        if paras.data.feed_method not in ['autoencoder', 'combined']:
            return
        os_train, os_test = results.get_output()
        ebd_train = os_train[1].data.cpu().numpy().reshape(os_train[1].shape[0], -1)
        ebd_test = os_test[1].data.cpu().numpy().reshape(os_test[1].shape[0], -1)
        pca = PCA(n_components=2)
        pca.fit(ebd_train)
        ebd_2d = pca.transform(ebd_test)
        plt.figure(figsize=(10, 6))
        if paras.data.feed_method == 'combined':
            plt.scatter(ebd_2d[:, 0], ebd_2d[:, 1], s=20,
                        c=np.mean(os_test[2].data.cpu().numpy(), axis=1).reshape(-1),
                        alpha=0.5, cmap=cm.rainbow)
            plt.colorbar()
        else:
            plt.scatter(ebd_2d[:, 0], ebd_2d[:, 1], s=20)
        plt.title("2D PCA Projection of Embedding Space (Colored by predicted labels)")
        plt.axis([np.min(ebd_2d[:, 0]), np.max(ebd_2d[:, 0]), np.min(ebd_2d[:, 1]), np.max(ebd_2d[:, 1])])
        plt.show()
