import os

import numpy as np
import tensorflow as tf
from tqdm import trange

#from buffer import Buffer
import scipy
import graph
from model import Model
from utils import BatchLoader, convert_to_one_hot
from six.moves import reduce, xrange


"""
Trainer: 

1. Initializes model
2. Train
3. Test
"""

class Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.task = config.task
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.checkpoint_secs = config.checkpoint_secs
        self.log_step = config.log_step
        self.num_epoch = config.num_epochs
        
        ## import data Loader ##
        data_dir = config.data_dir
        dataset_name = config.task
        batch_size = config.batch_size
        num_time_steps = config.num_time_steps
        self.data_loader = BatchLoader(data_dir, dataset_name,
                        batch_size, num_time_steps)
        
        ## Need to think about how we construct adj matrix(W)
        W = self.data_loader.adj        
        laplacian = W/W.max()
        laplacian = scipy.sparse.csr_matrix(laplacian, dtype=np.float32)
        lmax = graph.lmax(laplacian)      
        
        
        #idx2char = batchLoader_.idx2char
        #char2idx = batchLoader_.char2idx
        #batch_x, batch_y = batchLoader_.next_batch(0) 0:train 1:valid 2:test
        #batchLoader_.reset_batch_pointer(0)
        
        ## define model ##
        self.model = Model(config, laplacian, lmax)
        
        ## model saver / summary writer ##
        self.saver = tf.train.Saver()
        self.model_saver = tf.train.Saver(self.model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        
    def train(self):
        print("[*] Checking if previous run exists in {}"
              "".format(self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if tf.train.latest_checkpoint(self.model_dir) is not None:
            print("[*] Saved result exists! loading...")
            self.saver.restore(
                self.sess,
                latest_checkpoint
            )
            print("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            print("[*] No previous result")
            self.b_pretrain_loaded = False
            
        print("[*] Training starts...")
        self.model_summary_writer = None
        
        ##Training
        for n_epoch in trange(self.num_epoch, desc="Training[epoch]"):
            self.data_loader.reset_batch_pointer(0)
            for k in trange(self.data_loader.sizes[0], desc="[per_batch]"):
                # Fetch training data
                batch_x, batch_y = self.data_loader.next_batch(0)
                batch_x_onehot = convert_to_one_hot(batch_x, self.config.num_node)
                if self.config.model_type == 'lstm':
                    reshaped = batch_x_onehot.reshape([self.config.batch_size, 
                                                   self.config.num_node,
                                                   self.config.num_time_steps])
                    batch_x = reshaped
                elif self.config.model_type == 'glstm':
                    reshaped = batch_x_onehot.reshape([self.config.batch_size, 
                                                   self.config.num_time_steps,
                                                   1,self.config.num_node])
                    batch_x = np.transpose(reshaped,(0, 3, 2, 1))
                

                feed_dict = {
                    self.model.rnn_input: batch_x,
                    self.model.rnn_output: batch_y
                }
                res = self.model.train(self.sess, feed_dict, self.model_summary_writer,
                                       with_output=True)
                self.model_summary_writer = self._get_summary_writer(res)

            if n_epoch % 10 == 0:
                self.saver.save(self.sess, self.model_dir)
                
        ##Testing
        #for n_sample in trange(self.data_loader.size[2], desc="Testing"):
        #    batch_x, batch_y = self.data_loader.next_batch(2)
        #    batch_x_onehot = convert_to_one_hot(batch_x, self.config.num_node)
        #    reshaped = batch_x_onehot.reshape([self.config.batch_size, 
        #                                           self.config.num_time_steps,
        #                                           1,self.config.num_node])
        #    batch_x = np.transpose(reshaped,(0, 3, 2, 1))

        #    feed_dict = {
        #            self.model.rnn_input: batch_x,
        #            self.model.rnn_output: batch_y
        #        }
        #    res = self.model.test(self.sess, feed_dict, self.model_summary_writer,
        #                               with_output=True)
        #    self.model_summary_writer = self._get_summary_writer(res)
            
                
    def _get_summary_writer(self, result):
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None
        
        
