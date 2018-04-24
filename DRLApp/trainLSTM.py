from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This part is modified from:
#   https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html

class LSTMRNN(object):
    def __init__(self,sess,train_X, test_X, train_y, test_y,
                 input_size,
                 num_steps,
                 result_start,
                 result_end,
                 lstm_size=128,
                 num_layers=1,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"
                 ):
        self.sess = sess
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y
        self.input_size = input_size
        self.num_steps = num_steps
        self.result_start = result_start
        self.result_end = result_end
        self.lstm_size = lstm_size
        self.num_layers = num_layers

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.build_graph()

    def build_graph(self):
        """
        The model asks for five things to be trained:
        - learning_rate
        - keep_prob: 1 - dropout rate
        - symbols: a list of stock symbols associated with each sample
        - input: training data X
        - targets: training label y
        """
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # Stock symbols are mapped to integers.
        #self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        self.inputs_with_embed = tf.identity(self.inputs)
        self.embed_matrix_summ = None

        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32, scope="dynamic_rnn")
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(last, ws) + bias
        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")
        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)
        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y


    def train(self,config):
        tf.global_variables_initializer().run()
        self.merged_sum = tf.summary.merge_all()
        test_data_feed = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: self.test_X,
            self.targets: self.test_y,
        }
        global_step = 0
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )
            for batch_X, batch_y in self.generate_one_epoch(config.batch_size):
                global_step += 1
                epoch_step += 1
                train_data_feed = {
                    self.learning_rate: learning_rate,
                    self.keep_prob: config.keep_prob,
                    self.inputs: batch_X,
                    self.targets: batch_y,
                }
                train_loss, _, __ = self.sess.run(
                    [self.loss, self.optim, self.merged_sum], train_data_feed)
                if np.mod(global_step,200 / config.input_size) == 1:
                    test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)
                    print ("Step: "+str(global_step)+' Epoch: '+str(epoch)+' Learning rate:'+
                           str(learning_rate)+' train_loss:'+str(train_loss)+' test_loss:'+str(test_loss))
        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        print ("***********_________________**************")
        # What we really need is original window/2 ~
        a = self.result_start - len(self.train_X)-1
        b = self.result_end - len(self.train_X) -1
        return final_pred[a:b]