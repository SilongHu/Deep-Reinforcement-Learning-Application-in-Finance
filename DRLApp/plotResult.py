from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import glob
import re
import datetime
import time
import json

"""
This part is modified from :
https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/resultprocess/plot.py

"""
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def extract_result(path, typeList, typeIndex, coinList, choose):
    # Type is 0: *_original.txt, 1:*_normalized.txt
    # 2:train_weights_result_normalized.txt
    # 3:train_weights_result_original.txt
    mat_result = [[] for _ in range(len(coinList))]
    if typeIndex <= 1:
        for filename in glob.glob(os.path.join(path, typeList[typeIndex])):
            pred = np.loadtxt(filename)
            if filename.split('/' + choose + '/')[1] in typeList:
                continue
            coin_name = filename.split('/' + choose + '/')[1].split('_result_')[0]
            index_of_coin = coinList.index(coin_name)
            mat_result[index_of_coin] = pred

        mat_result = np.array(mat_result)
        weight_result_original = mat_result / np.sum(mat_result, axis=0)
        weight_result_original = weight_result_original.transpose()
        weight_result_original = np.insert(weight_result_original, 0, 0.0, axis=1)

    else:
        filename = glob.glob(os.path.join(path, typeList[typeIndex]))
        pred = np.loadtxt(filename[0])
        if typeIndex == 2:
            weight_result_original = pred/np.sum(pred,axis=0)
            weight_result_original = weight_result_original.transpose()
            weight_result_original = np.insert(weight_result_original, 0, 0.0, axis=1)
        else:

            result = softmax(pred)
            weight_result_original = np.transpose(result)
            weight_result_original = np.insert(weight_result_original, 0, 0.0, axis=1)

    return weight_result_original

def plot_result(choice):

    # We need weights in the programlog
    # Adjusting weights should be same.
    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/programlog") as file:
        # We want the last 2 elements of time_line
        time_line = file.readline().strip('\n').split(' ')
        coin_line = file.readline().strip('\n').split('\'')
        # We want the odd index of coin_line
        coin_list = []
        index = 0
        for item in coin_line:
            if index % 2 == 0:
                index += 1
            else:
                index += 1
                coin_list.append(item)
    #print (coin_list)

    path = os.path.dirname(__file__) + "/../" + "trend_result/" + str(choice)
    type_list = ['*_original.txt', '*_normalized.txt',
                 'train_weights_result_original.txt',
                 'train_weights_result_normalized.txt'
                 ]
    # for i in range(2,4):
    weight_result_original = extract_result(path, type_list, 0, coin_list, choice)
    weight_result_normalized = extract_result(path, type_list, 1, coin_list, choice)
    train_weight_original = extract_result(path, type_list, 2, coin_list, choice)
    train_weight_normalized = extract_result(path, type_list, 3, coin_list, choice)

    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/programlog") as myfile:
        data = myfile.read().replace('\n', '')


    regex = r"the raw omega is \[(.*?)\]DEBUG:root:"
    result = re.findall(regex, data, re.DOTALL)
    ori_weights = []
    for j in range(len(result)):
        haha = [float(i) for i in result[j].split()]
        ori_weights.append(haha)

    ori_weights = np.array(ori_weights)
    trend_weights_original = np.repeat(weight_result_original, 2, axis=0)
    trend_weights_normalized = np.repeat(weight_result_normalized, 2, axis=0)
    ori_length = ori_weights.shape[0]
    trend_length = trend_weights_original.shape[0]

    if ori_length == trend_length + 1:
        trend_weights_original = np.insert(trend_weights_original, trend_length - 1,
                                           trend_weights_original[trend_length - 1], axis=0)
        trend_weights_normalized = np.insert(trend_weights_normalized, trend_length - 1,
                                             trend_weights_normalized[trend_length - 1], axis=0)
    elif ori_length == trend_length - 1:
        trend_weights_original = np.delete(trend_weights_original, trend_length - 1, 0)
        trend_weights_normalized = np.delete(trend_weights_normalized, trend_length - 1, 0)

    elif ori_length == trend_length:
        print ('Exactly same number')
    else:
        print ('Original testing set number can not match trends test set number!')
        exit()

    regg = r"Silonghu Added: the future price is \[(.*?)\]DEBUG:root:"
    price_result = re.findall(regg, data, re.DOTALL)
    future_price = []
    for j in range(len(price_result)):
        haha = [float(i) for i in price_result[j].split()]
        future_price.append(haha)
    future_price = np.array(future_price)

    alpha_list = [0, 1, 2, 3, 4]
    weights_list = [ori_weights,
                    trend_weights_original,
                    trend_weights_normalized,
                    train_weight_original,
                    train_weight_normalized]

    weights_name = ['Original weights','trends_original_weights',
                    'trends_normalized_weights',
                    'train_weights_original',
                    'train_weights_normalized']
    final_result = []
    coin_number = len(coin_list)
    end_value = []
    for i in alpha_list:
        number_result = back_test(coin_number).main(ori_length,
                                                    weights_list[i],
                                                    ori_weights,
                                                    1.0,future_price)
        end_value.append(number_result[ori_length-1])
        final_result.append(number_result)
    print (end_value)

    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/net_config.json") as file:
        config = json.load(file)
    start,end = _extract_test(config)
    timestamps = np.linspace(start,end,len(ori_weights))
    dates = [datetime.datetime.fromtimestamp(int(ts)-int(ts)%config['input']['global_period'])
             for ts in timestamps]

    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator()

    #rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"],
                  #"size": 8})
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    for i, pvs in enumerate(final_result):
        label = "type: "+str(weights_name[i])+"\n"+"end_value = "+str(end_value[i])
        if max(end_value)>5:
            ax.semilogy(dates, pvs, linewidth=1, label=label)
        else:
            ax.plot(dates, pvs, linewidth=1, label=label)

    plt.ylabel("portfolio value $p_t/p_0$", fontsize=12)
    plt.xlabel("time", fontsize=12)
    xfmt = mdates.DateFormatter("%m-%d %H:%M")
    # Judge the length, set_major_locator or set_minor_locator
    #print ((end-start)/86400.0)
    day_interval = (end-start)/86400.0
    if day_interval > 15:
        ax.xaxis.set_major_locator(weeks)
        ax.xaxis.set_minor_locator(days)
    else:
        ax.xaxis.set_major_locator(days)
        #ax.xaxis.set_minor_locator(days)

    datemin = dates[0]
    datemax = dates[-1]
    ax.set_xlim(datemin, datemax)

    ax.xaxis.set_major_formatter(xfmt)

    plt.grid(True)
    ax.legend(loc="upper left", prop={"size":9})
    fig.autofmt_xdate()
    file_name = "result_"+choice+'.png'
    plt.savefig(file_name)
    plt.show()
    return 0

def _extract_test(config):
    global_start = parse_time(config['input']['start_date'])
    global_end = parse_time(config['input']['end_date'])
    span = global_end - global_start
    start = global_end - config['input']['test_portion']*span
    end = global_end
    return start,end

def parse_time(time_string):
    return time.mktime(datetime.datetime.strptime(time_string, "%Y/%m/%d").timetuple())

class back_test(object):
    def __init__(self,coin_number):
        self.commission_rate = 0.0025
        self.__test_pc_vector = []
        self.coin_number = coin_number
        self._last_omega = np.zeros((self.coin_number+1,))
        self._last_omega[0] = 1
        self.step = 0

    def trade_by_stratedy(self,omega,future_price):
        #future_price = []# We could gather from text
        pv_after_commission = self.calculate_pv_after_commission(omega, self._last_omega, self.commission_rate)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._last_omega = pv_after_commission * omega * \
                           future_price / \
                           portfolio_change
        self.__test_pc_vector.append(portfolio_change)

    def calculate_pv_after_commission(self,w1, w0, commission_rate):
        """
        @:param w1: target portfolio vector, first element is btc
        @:param w0: rebalanced last period portfolio vector, first element is btc
        @:param commission_rate: rate of commission fee, proportional to the transaction cost
        """
        mu0 = 1
        mu1 = 1 - 2 * commission_rate + commission_rate ** 2
        while abs(mu1 - mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - commission_rate * w0[0] -
                   (2 * commission_rate - commission_rate ** 2) *
                   np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
                  (1 - commission_rate * w1[0])
        return mu1

    def main(self,total_step,trends,original,alpha,future):
        while(self.step < total_step):
            new_omega = alpha * trends[self.step] + (1 - alpha) * original[self.step]
            self.trade_by_stratedy(new_omega, future[self.step])
            self.step += 1
        return np.cumprod(self.__test_pc_vector)