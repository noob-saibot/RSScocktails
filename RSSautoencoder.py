import pickle
import pandas as pd
import tensorflow as tf
import numpy as np


class RSSautoencoder:
    def __init__(self):
        self.data = self._load_test()

    @property
    def config(self):
        return {'learning_rate': 0.001,
                'num_steps': 2000,
                'batch_size': 10,
                'num_input': 62110, }

    def _load_test(self):
        with open('data_encoder', 'rb') as file:
            return pickle.load(file)

    def create_network(self):
        pass


if __name__ == '__main__':
    R = RSSautoencoder()
    cos = R.data['cosine']
    euc = R.data['euclidean']
    tdm = np.stack([cos.values, euc.values], axis=1)
    print(tdm.shape)