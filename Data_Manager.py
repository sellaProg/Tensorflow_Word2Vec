import nltk
import numpy as np
import tensorflow as tf
from collections import Counter
class Load_data(object):
    def __init__(self,path):
        with open(path, "r", encoding="UTF-8") as f:
            self.words_list = []
            for line in f:
                self.words = nltk.wordpunct_tokenize(line)
                self.words_list.extend(list(self.words))

        self.words_per_epoch = len(self.words_list)
        self.vocab_counts=Counter(self.words_list)
        self.vocab_words=[w for w in self.vocab_counts]



    def train_data(self,data,window_size=10):
        data_size=len(data)
        train_x=[]
        train_y=[]
        window=window_size*2+1
        assert data_size > window * 2
        possible_windows= data_size-window_size*2
        train_windows=divmod((possible_windows*80),100)[0]
        for i in range(train_windows):
            tw=data[i:i+window]
            cw=tw[window_size]
            for j in range(window):
                if tw[j] is not cw:
                    train_x.append(cw)
                    train_y.append(tw[j])

        return train_x,train_y

    def test_data(self,data,window_size=10):
        data_size=len(data)

        test_x=[]
        test_y=[]
        window=window_size*2+1
        assert data_size >window*2
        possible_windows= data_size-window_size*2
        train_windows=divmod((possible_windows*80),100)[0]
        test_windows=possible_windows-train_windows
        start=window_size+train_windows+1

        for i in range(start,start+test_windows):
            tw=data[i-window_size-1:i+window_size+1]
            cw=tw[window_size]
            for j in range(window):
                if tw[j] is not cw:
                    test_x.append(cw)
                    test_y.append(tw[j])

        return test_x,test_y
if __name__ == '__main__':
    loader=Load_data("text8.txt")

    loader.test_data(loader.words_list)