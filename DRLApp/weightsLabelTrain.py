from __future__ import absolute_import
import numpy as np
import os
import glob
import re
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DRLApp.preProcess import generate_whole_set

def weights_label_training(choice,normalized):
    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/programlog") as myfile:
        data = myfile.read().replace('\n', '')
    regex = r"The training Omega is \[(.*?)\]INFO:"
    result = re.findall(regex, data, re.DOTALL)
    train_weights = []
    for j in range(len(result)):
        haha = [float(i) for i in result[j].split()]
        train_weights.append(haha)
    train_weights = np.array(train_weights)

    train_number = train_weights.shape[0]
    train_weights = np.transpose(train_weights)

    #print (train_weights.shape)
    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/programlog") as myfile:
        data = myfile.read().replace('\n', '')


    regex = r"the raw omega is \[(.*?)\]DEBUG:root:"
    result = re.findall(regex, data, re.DOTALL)
    test_weights = []
    for j in range(len(result)):
        haha = [float(i) for i in result[j].split()]
        test_weights.append(haha)

    test_weights = np.array(test_weights)
    test_number = test_weights.shape[0]
    test_weights = np.transpose(test_weights)
    #print (train_number)
    #print (test_number)

    '''
    Firstly, we focus on the original style of data
    train_X: google trends
    test_X: train_weights

    train_Y: LSTM Result from trend_result
    test_Y: test_weights
    '''
    newDict, coin_list, _,__, ___= generate_whole_set(choice)

    result_all_coin = [[] for _ in range(len(coin_list))]

    # Grab the train_X, the number should be same and
    if normalized:
        print ('Training based on normalized way')
    else:
        print ('Training based on original way')
    index = 0
    for each_coin in coin_list:
        quotient,remainder = divmod(train_number,2)
        train_each_X = newDict[each_coin][:quotient]
        print ('Training weights on coin:',each_coin)

        index_1 = 0
        for each in train_each_X:
            if each == 0 and index_1 == 0:
                train_each_X[index_1] = np.mean(train_each_X)
            if each == 0 and index_1 != 0:
                train_each_X[index_1] = train_each_X[index_1 - 1]
            index_1+=1

        if normalized:
            # Dealing with train_each_X, one / one
            first = train_each_X[0]
            train_each_X = [train_each_X[i]/train_each_X[i-1] for i in range(1,quotient)]
            train_each_X.insert(0,first)

        train_each_X = np.array(train_each_X)
        train_each_X = np.repeat(train_each_X,2)
        if remainder != 0 :
            train_each_X = np.insert(train_each_X,quotient,
                                     newDict[each_coin][quotient])
        #print (len(train_each_X))
        train_each_Y = train_weights[index]

        # train_each_Y , Original Style
        path = os.path.dirname(__file__) + "/../" + "trend_result/" + str(choice)
        filename = glob.glob(os.path.join(path, str(each_coin)+'_result_original.txt'))
        pred_LSTM = np.loadtxt(filename[0])

        test_each_X = np.array(pred_LSTM)
        test_each_X = np.repeat(test_each_X,2)

        quotient2, remainder2 = divmod(test_number, 2)
        if remainder2 != 0:
            test_each_X = np.insert(test_each_X,quotient2-1,
                                    test_each_X[quotient2-1])
        test_each_Y = test_weights[index]
        # Parameters
        '''
        This part is modified from:
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master
        /examples/2_BasicModels/linear_regression.py
        '''
        learning_rate = 0.05
        training_epochs = 50
        display_step = 10
        # tf Graph Input
        X = tf.placeholder("float")
        Y = tf.placeholder("float")
        # Set model weights
        W = tf.Variable(np.random.randn(), name="weight")
        b = tf.Variable(np.random.randn(), name="bias")
        # Construct a linear model
        pred_linear = tf.add(tf.multiply(X, W), b)
        # Mean squared error
        cost = tf.reduce_sum(tf.pow(pred_linear - Y, 2)) / (2 * train_number)
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        # Start training
        with tf.Session() as sess:
            sess.run(init)
            # Fit all training data
            for epoch in range(training_epochs):
                for (x, y) in zip(train_each_X, train_each_Y):
                    sess.run(optimizer, feed_dict={X: x, Y: y})
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    c = sess.run(cost, feed_dict={X: train_each_X, Y: train_each_Y})
                    print ('Epoch:'+str(epoch+1)+' cost='+str(c))

            result_each_coin = sess.run(pred_linear,feed_dict={X:test_each_X})
            result_all_coin[index]=result_each_coin
            sess.close()

        index += 1
    result_all_coin = np.array(result_all_coin)
    if normalized:
        np.savetxt('./trend_result/' + choice + '/' +
                   'train_weights_result_normalized.txt', result_all_coin)
    else:
        np.savetxt('./trend_result/' + choice + '/' +
                'train_weights_result_original.txt', result_all_coin)
    return 0