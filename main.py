from __future__ import absolute_import
import os
import time
from argparse import ArgumentParser
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DRLApp.generate import generate_data
from DRLApp.preProcess import generate_whole_set
from DRLApp.preProcess import generate_individual_set
from DRLApp.trainLSTM import LSTMRNN
from DRLApp.plotResult import plot_result
from DRLApp.weightsLabelTrain import weights_label_training
import numpy as np

'''
The LSTM training part is modified from:
    https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html

These flags are parameters of LSTM network
'''

class FLAG(object):
    def __init__(self):
	self.input_size = 1
	self.num_steps = 15
	self.num_layers = 1
	self.lstm_size = 128
	self.batch_size = 32
	self.keep_prob = 0.8
	self.init_learning_rate = 0.001
	self.learning_rate_decay = 0.99
	self.init_epoch = 5
	self.max_epoch = 50
FLAGS = FLAG()
'''
flags = tf.app.flags
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 15, "Num of steps [15]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
FLAGS = flags.FLAGS
'''
def build_parser():
    '''
    Command:
     python --mode=generate --choice=1
    '''
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="train, generate, plot",
                        metavar="MODE", default="train")
    parser.add_argument("--choice", dest="choice",
                        help="choice the index of training subfolder",
                        default="1")
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists('./'+'train_package'):
        print('Please copy Zhengyao\'s train_package to this directory first!')
        exit()

    if not os.path.exists("./" + "database"):
        os.makedirs("./" + "database")

    if not os.path.exists("./" + "trend_result"):
        os.makedirs("./" + "trend_result")

    if options.mode == "train":
        '''
        train MODE:
        1. LSTM for predicting Google trends data and store results into
            trend_result/ directory named [*Coin Name*]_result_normalized.txt
            and [*Coin Name*]_result_original.txt
        2. Label JiangZhengyao's final training weights as Y, Google trends data
            as X, use Linear Regression to predict weights based on 1's LSTM result.
        '''
        print ("------------- Start Training ----------------")
        if not os.listdir("./database/"+options.choice):
            print ("You need generate the data first, please try:")
            print ("\'python main.py --mode=generate --choice=" + str(options.choice)+'\'')
        else:

            print ("Training Step 1: generate training and testing set")
            input_size = 1
            num_steps = 15
            newDict, coin_list,index_result_start, index_result_end,training_number = generate_whole_set(options.choice)
            print ("Training Step 2: Learning LSTM")
            if not os.path.exists("./trend_result/" + options.choice):
                os.makedirs("./trend_result/" + options.choice)
            if not os.listdir("./trend_result/" + options.choice):
                # Here we should generate two kinds of data,
                # one for original, another for normalized.
                LSTM_start = time.time()
                print ('------ Start training in original way ------')
                for each_coin in coin_list:
                    print ('Start training on coin:', each_coin)
                    normalized = 0
                    train_X, test_X, train_y, test_y = generate_individual_set(each_coin, newDict, training_number,
                                                                               input_size,num_steps,normalized)
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        rnn_model = LSTMRNN(sess, train_X, test_X, train_y, test_y, input_size, num_steps,
                                            index_result_start,
                                            index_result_end, lstm_size=128, num_layers=1)
                        result = rnn_model.train(FLAGS)
                        np.savetxt('./trend_result/'+ options.choice+'/'+
                                   each_coin+'_result_original.txt',result,delimiter=',')
                        sess.close()
                print ('------ Start training in normalized way ------')
                for each_coin in coin_list:
                    print ('Start training on coin:', each_coin)
                    normalized = 1
                    train_X, test_X, train_y, test_y = generate_individual_set(each_coin, newDict, training_number,
                                                                               input_size,num_steps,normalized)
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        rnn_model = LSTMRNN(sess, train_X, test_X, train_y, test_y, input_size, num_steps,
                                            index_result_start,
                                            index_result_end, lstm_size=128, num_layers=1)
                        result = rnn_model.train(FLAGS)
                        np.savetxt('./trend_result/'+ options.choice+'/'+
                                   each_coin+'_result_normalized.txt',result,delimiter=',')
                LSTM_end = time.time()
                print ('LSTM training cost:',LSTM_end - LSTM_start)
                print ("Training Step 3: Weights Training")
                weights_label_training(options.choice, 0)
                weights_label_training(options.choice, 1)
                print ('Weights label training cost:',time.time() - LSTM_end,'seconds')
            else:
                print ("LSTM Training already Done! Please use plot!")

    elif options.mode == "generate":
        '''
        generate MODE:
            Grab the Google Trends data online and store
            into the database/ directory
        '''
        if not os.path.exists("./database/"+options.choice):
            os.makedirs("./database/"+options.choice)
        # if folder is empty
        if not os.listdir("./database/"+options.choice):
            print ('Generating dataset...')
            start_generate_time = time.time()
            generate_data(options.choice)
            print ('Cost: ' + str(time.time() - start_generate_time) + ' seconds on Google Trends Generation')
        else:
            print ("dataset already exist, please try:")
            print ("\'python main.py --mode=train --choice=" + str(options.choice) + '\'')

    elif options.mode == "plot":
        """
        plot MODE:
            Get the results from trend_result/ directory and
            plot them

        """
        if not os.path.exists("./trend_result/"+options.choice):
            os.makedirs("./trend_result/"+options.choice)
        # if folder is empty
        if not os.listdir("./trend_result/"+options.choice):
            print ("You need train the data first, please try:")
            print ("\'python main.py --mode=train --choice=" + str(options.choice) + '\'')
        else:
            print ("Ready to plot!")
            plot_result(options.choice)

if __name__ == "__main__":
    main()
