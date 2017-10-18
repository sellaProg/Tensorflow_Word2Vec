import tensorflow as tf
import numpy as np



# we define the class where we can get the hyperparameters

class Config(object):
    def __init__(self):
        # embedding dimebtiopn
        self.emb_dim=300
        # trainging data path
        self.train_data=""
        # number of negative samples per training example
        self.neg_samples_num=100
        #learning rate
        self.learn_rate=1e-4
        # number of epochs
        self.epoch_num=20
        # Number of examples for one training step.
        self.batch_size =16
        # prediction window size
        self.window_size=10
        # minimum number of occurences of a word to be counted
        self.min_occ=5
        # this one is to down sample word that appears too often like {the, a, an ....}
        self.subsample=1e-3

