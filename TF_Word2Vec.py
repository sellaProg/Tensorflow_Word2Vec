import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
from Data_Manager import Load_data


# we define the class where we can get the hyperparameters

class Configurations(object):
    def __init__(self):
        #hidden layer size
        self.h_layer_size=50
        # embedding dimebtiopn
        self.emb_dim=300
        # trainging data path
        self.train_data="text8.txt"
        # number of negative samples per training example
        self.neg_samples_num=100
        #learning rate
        self.learn_rate=1e-4
        # number of epochs
        self.epoch_num=10
        # Number of examples for one training step.
        self.batch_size =772
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
        #self.save_path=None
        #if not os.path.exists(self.save_path):
         #   os.makedirs(self.save_path)
        # path evaluation data
        self.eval_data = None

class SkipGram(object):

    # this is going to be our model
    #we need to specify the session at initialization time

    def __init__(self):
        #import the hyper-parameters
        self.config = Configurations()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.config.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.config.batch_size,1])
        loader=Load_data(self.config.train_data)
        #self.test_data ,self.test_labels=loader.test_data(loader.words_list)
        self.words_list=loader.words_list
        self.words_per_epoch=loader.words_per_epoch
        self.vocab_counts=loader.vocab_counts
        self.id2word=loader.vocab_words
        self.word2id={}
        for i, w in enumerate(self.id2word):
            self.word2id[w] = i
        train_data,train_label=self.get_enumerated_data(loader)
        embeddings=tf.Variable(tf.random_uniform([len(self.id2word),self.config.emb_dim],-1,1))
        weights=tf.Variable(tf.random_uniform([len(self.id2word),self.config.emb_dim],-1,1))
        biases=tf.Variable(tf.zeros([len(self.id2word)]))
        lookup_table=tf.nn.embedding_lookup(embeddings,self.train_inputs)
        loss=tf.reduce_mean(tf.nn.nce_loss(weights=weights,biases=biases,labels=self.train_labels,inputs=lookup_table,num_sampled=self.config.neg_samples_num,num_classes=len(self.id2word)))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # we need to normalize the embeddings before visualization

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        with tf.Session() as sess:

            tf.global_variables_initializer().run()
            print(len(train_data),len(train_label))
            assert len(train_data)%self.config.batch_size==0
            for i in range(self.config.epoch_num):
                epoch_oss=0
                for j in range(int(len(train_data)/self.config.batch_size)):
                    batch_inputs=train_data[j*self.config.batch_size:j*self.config.batch_size+self.config.batch_size]
                    batch_labels=train_label[j*self.config.batch_size:j*self.config.batch_size+self.config.batch_size]
                    batch_labels=np.reshape(batch_labels,[self.config.batch_size,1])
                    feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                    epoch_oss+=loss_val
                    if j%2000==0:
                        print("average loss after",j,"batches is : ",epoch_oss/j)
                print("the loss at the epoch :",i," is : ",epoch_oss)
            final_embeddings = normalized_embeddings.eval()

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 1000
        low_dim_embs = tsne.fit_transform(final_embeddings[-plot_only:-1, :])
        labels = [self.id2word[i] for i in range(-plot_only,-1)]
        self.plot_with_labels(low_dim_embs, labels)
    def get_enumerated_data(self,loader):
        enumerated_data=[]
        enumerated_labels=[]
        data,labels=loader.train_data(self.words_list)
        for i in range(len(data)):
            enumerated_data.append(self.word2id[data[i]])
        for i in range(len(labels)):
            enumerated_labels.append(self.word2id[labels[i]])

        return enumerated_data,enumerated_labels

    def plot_with_labels(self,low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(20, 20))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)
if __name__ == '__main__':
    model=SkipGram()