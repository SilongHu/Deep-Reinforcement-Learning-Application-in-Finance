from __future__ import absolute_import
import json
import os.path
import datetime as dt
from datetime import timedelta
from pytrends.request import TrendReq
import pandas as pd
import time

def load_config_data(choice):
    # Two files we want:
    # 1. net_config.json
    # 2. programlog
    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/net_config.json") as file:
        config = json.load(file)
    #we want the 'input' layer: coin_number,start_date,end_date,window_size
    input_info = config['input']
    coin_number = input_info['coin_number']
    start_date = input_info['start_date']
    end_date = input_info['end_date']
    window_size = input_info['window_size']
    return [coin_number,start_date,end_date,window_size]

def load_log_data(choice,window):

    with open(os.path.dirname(__file__) +"/../" + "train_package/"+str(choice)+"/programlog") as file:
        # We want the last 2 elements of time_line
        time_line = file.readline().strip('\n').split(' ')
        train_end = time_line[len(time_line)-2] +' '+ time_line[len(time_line)-1]+':00'
        fake_train_end = dt.datetime.strptime(train_end,'%Y-%m-%d %H:%M:%S')
        datetime_train_end = dt.datetime.strptime(train_end, '%Y-%m-%d %H:%M:%S') - timedelta(hours=(window+1)/2)
        coin_line = file.readline().strip('\n').split('\'')
        # We want the odd index of coin_line
        coin_list = []
        index = 0
        for item in coin_line:
            if index %2 == 0:
                index += 1
            else:
                index+=1
                coin_list.append(item)

    return fake_train_end,datetime_train_end,coin_list


def generate_data(choice):
    global name_dict
    #time_start = time.time()
    # Abbreviation: Real Name, if ignored, please added by hand
    name_dict = {'ETH':'ethereum','LTC':'litecoin','XRP':'ripple','ETC':'ethereum classic',\
                 'DASH':'dash coin','XMR':'monero','XEM':'NEM','FCT':'factom','GNT':'golem',\
                 'ZEC':'zcash','reversed_USDT':'tether','BTN':'bitcoin','STR':'stellar',\
                 'BCH':'bitcoin cash','LSK':'lisk','SC':'siacoin','ZRX':'0x','VTC':'vertcoin',\
                 'BTS':'bitshares','STRAT':'Stratis','DGB':'digibyte','OMG':'omisego',\
                 'DOGE':'dogecoin','NXT':'nxt','EMC2':'einsteinium','AMP':'synereo amp',\
                 'MAID':'maidsafecoin','REP':'augur','DCR':'decred','SYS':'syscoin',\
                 'ARDR':'ardor','CVC':'civic','LBC':'lbry credits',\
                 'STEEM':'steem','OMNI':'omni','BCN':'bytecoin','NAV':'navcoin',\
                 'PASC':'pascalcoin','POT':'potcoin','XCP':'counterparty','VIA':'viacoin',\
                 'XPM':'primecoin','RIC':'riecoin','XBC':'Bitcoinplus','GNO':'gnosis',\
		 'CLAM':'CLAMS'}

    # First We should gather the configure information from train_package
    config_info = load_config_data(choice)
    _,train_end,coin_list = load_log_data(choice,config_info[3])
    # Now generating trend_list

    # Split Coins into Groups with size n
    n = 3
    upper_coin_list = [coin_list[i:i + n] for i in range(0, len(coin_list), n)]
    lower_coin_list = []
    for upper_list in upper_coin_list:
        upper_list.append('BTN')
        temp_lower_list= []
        for item in upper_list:
            temp_lower_list.append(name_dict[item])
        lower_coin_list.append(temp_lower_list)
    # Here we base on date to choose the data
    start_date = str(config_info[1]).replace('/','-',2)
    end_date = str(config_info[2]).replace('/','-',2)

    pytrend = TrendReq(hl='en-US', tz=360)
    # Let us store uppercase trend first
    length = len(upper_coin_list)
    for i in range(length):
        time.sleep(0.4)
        coin_list = upper_coin_list[i]
        newDF = pd.DataFrame()
        start_timedate = dt.datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_timedate = dt.datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        print ('Generating trends for upper coin list '+str(i+1)+'/'+str(length))

        while start_timedate < end_timedate:
            #time.sleep(0.7)
            next_date = start_timedate + timedelta(days=3)
            frame_time = str(start_timedate.strftime("%Y-%m-%d")) + 'T00' + ' ' + \
                 str(next_date.strftime("%Y-%m-%d")) + 'T00'
            pytrend.build_payload(kw_list=coin_list, timeframe=frame_time)
            interest_over_time_df = pytrend.interest_over_time()
            newDF = newDF.append(interest_over_time_df)
            start_timedate = next_date
        path_file = os.path.dirname(__file__)+"/../" + "database/"+str(choice)+'/'+'upperdata'+str(i)+'.csv'
        newDF.to_csv(path_file)

    # Let us store lowercase
    for i in range(length):
        time.sleep(0.4)
        coin_list = lower_coin_list[i]
        newDF = pd.DataFrame()

        start_timedate = dt.datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_timedate = dt.datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        print ('Generating trends for lower coin list '+ str(i + 1) + '/' +str(length))
        while start_timedate < end_timedate:
            #time.sleep(0.7)
            next_date = start_timedate + timedelta(days=3)
            frame_time = str(start_timedate.strftime("%Y-%m-%d")) + 'T00' + ' ' + \
                 str(next_date.strftime("%Y-%m-%d")) + 'T00'
            pytrend.build_payload(kw_list=coin_list, timeframe=frame_time)
            interest_over_time_df = pytrend.interest_over_time()
            newDF = newDF.append(interest_over_time_df)
            start_timedate = next_date
        path_file = os.path.dirname(__file__)+"/../" + "database/"+str(choice)+'/'+'lowerdata'+str(i)+'.csv'
        newDF.to_csv(path_file)
    print ("-------------*********** Generating Done ***********------------------")
