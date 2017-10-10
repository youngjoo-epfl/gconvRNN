import tensorflow as tf
import numpy as np
import math
import pickle
import os
import json
from datetime import datetime
from IPython import embed
import tensorflow.contrib.slim as slim
from scipy.sparse import coo_matrix

def save_config(model_dir, config):
    '''
    save config params in a form of param.json in model directory
    '''
    param_path = os.path.join(model_dir, "params.json")

    print("[*] PARAM path: %s" %param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def prepare_dirs(config):
    if config.load_path:
        config.model_name = "{}_{}".format(config.task, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.task, get_time())

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory '%s' created" %path)
            

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
        
def convert_to_one_hot(a, max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())
            
class BatchLoader(object):
    def __init__(self, data_dir, dataset_name, batch_size, seq_length):
        train_fname = os.path.join(data_dir, dataset_name, 'ptb.char.train.txt')
        valid_fname = os.path.join(data_dir, dataset_name, 'ptb.char.valid.txt')
        test_fname = os.path.join(data_dir, dataset_name, 'ptb.char.test.txt')
        input_fnames = [train_fname, valid_fname, test_fname]

        vocab_fname = os.path.join(data_dir, dataset_name, 'vocab_char.pkl')
        tensor_fname = os.path.join(data_dir, dataset_name, 'data_char.pkl')
        Adj_fname = os.path.join(data_dir, dataset_name, 'adj_char.pkl')

        if not os.path.exists(vocab_fname) or not os.path.exists(tensor_fname) or not os.path.exists(Adj_fname):
            print("Creating vocab...")
            self.text_to_tensor(input_fnames, vocab_fname, tensor_fname, Adj_fname)

        print("Loading vocab...")
        adj = pklLoad(Adj_fname)
        all_data = pklLoad(tensor_fname)
        self.idx2char, self.char2idx = pklLoad(vocab_fname)
        vocab_size = len(self.idx2char)

        print("Char vocab size: %d" % (len(self.idx2char)))
        self.sizes = []
        self.all_batches = []
        self.all_data = all_data
        self.adj = adj

        print("Reshaping tensors...")
        for split, data in enumerate(all_data):  # split = 0:train, 1:valid, 2:test
            #Cutting training sample for check profile fast..(Temporal)
            #if split==0:
            #    #Only for training set
            #    length = data.shape[0]
            #    data = data[:int(length/4)]
            
            length = data.shape[0]
            data = data[: batch_size * seq_length * int(math.floor(length / (batch_size * seq_length)))]
            ydata = np.zeros_like(data)
            ydata[:-1] = data[1:].copy()
            ydata[-1] = data[0].copy()

            if split < 2:
                x_batches = list(data.reshape([-1, batch_size, seq_length]))
                y_batches = list(ydata.reshape([-1, batch_size, seq_length]))
                self.sizes.append(len(x_batches))
            else:
                x_batches = list(data.reshape([-1, batch_size, seq_length]))
                y_batches = list(ydata.reshape([-1, batch_size, seq_length]))
                self.sizes.append(len(x_batches))

            self.all_batches.append([x_batches, y_batches])

        self.batch_idx = [0, 0, 0]
        print("data load done. Number of batches in train: %d, val: %d, test: %d" \
              % (self.sizes[0], self.sizes[1], self.sizes[2]))

    def next_batch(self, split_idx):
        # cycle around to beginning
        if self.batch_idx[split_idx] >= self.sizes[split_idx]:
            self.batch_idx[split_idx] = 0
        idx = self.batch_idx[split_idx]
        self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
        return self.all_batches[split_idx][0][idx], \
               self.all_batches[split_idx][1][idx]

    def reset_batch_pointer(self, split_idx, batch_idx=None):
        if batch_idx == None:
            batch_idx = 0
        self.batch_idx[split_idx] = batch_idx

    def text_to_tensor(self, input_files, vocab_fname, tensor_fname, Adj_fname):
        counts = []
        char2idx = {}
        idx2char = []

        output = []

        for input_file in input_files:
            count = 0
            output_chars = []
            with open(input_file) as f:
                for line in f:
                    line = ''.join(line.split())
                    chars_in_line = list(line)
                    chars_in_line.append('|')
                    for char in chars_in_line:
                        if char not in char2idx:
                            idx2char.append(char)
                            #                 print("idx: %d, char: %s" %(len(idx2char), char))
                            char2idx[char] = len(idx2char) - 1
                        output_chars.append(char2idx[char])
                        count += 1
            counts.append(count)
            output.append(np.array(output_chars))

        train_data = output[0]
        train_data_shift = np.zeros_like(train_data)
        train_data_shift[:-1] = train_data[1:].copy()
        train_data_shift[-1] = train_data[0].copy()

        # Co-occurance
        Adj = np.zeros([len(idx2char), len(idx2char)])
        for x, y in zip(train_data, train_data_shift):
            Adj[x, y] += 1

        # Make Adj symmetric & visualize it


        print("Number of chars : train %d, val %d, test %d" % (counts[0], counts[1], counts[2]))
        pklSave(vocab_fname, [idx2char, char2idx])
        pklSave(tensor_fname, output)
        pklSave(Adj_fname, Adj)
