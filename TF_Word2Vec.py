import tensorflow as tf
import numpy as np



# we define the class where we can get the hyperparameters

class Configurations(object):
    def __init__(self):
        # embedding dimebtiopn
        self.emb_dim=300
        # trainging data path
        self.train_data=None
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
        # print statistics and save summery every stat_time seconds
        self.stat_time=5
        #time in seconds when we save the state of the model (checkpoint)
        self.checkpoint=600
        # Directory to write the model and training summaries.
        self.save_path=None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # path evaluation data
        self.eval_data = None

class SkipGram(object):

    # this is going to be our model
    #we need to specify the session at initialization time

    def __init__(self,session):
        #import the hyper-parameters
        self.config=Configurations()
        #define the working session
        self.sess=session
        self._word2id = {}
        self._id2word = []
        model_graph()

    def model_graph(self):

        # Build the graph for the model
        

