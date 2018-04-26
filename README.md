# Deep Reinforcement Learning Application in Finance

USC 2018 Spring Directed Research ----- Deep Reinforcement Learning Application in Finance

In this project, we are using [Google Trends](https://medium.com/google-news-lab/what-is-google-trends-data-and-what-does-it-mean-b48f07342ee8) as our source data and algorithm LSTM (Long Short Time Memory) for predicting portfolio's weights, based on [ZhengyaoJiang's paper](https://arxiv.org/pdf/1706.10059.pdf) and [his github](https://github.com/ZhengyaoJiang/PGPortfolio).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Before we grab the Google Trends data, [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio) should be downloaded first. 

```
$ git clone https://github.com/ZhengyaoJiang/PGPortfolio.git
```
After that, you should modified his repo slightly. The 4 different parts are following:

![Different Graph](https://github.com/SilongHu/Deep-Reinforcement-Learning-Application-in-Finance/blob/master/four_little_modified.png)

Adding the GREEN contents and removing the RED contents for 4 different places in 3 different files.

Then run ZhengyaoJiang's code based on his [USERGUIDE](https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/user_guide.md).

You could have as many training packages as you want via his project. When you finished all the processes in [USERGUIDE](https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/user_guide.md). Replace my train_package/ directory by his train_package. The case 1 in my train_package is an example.

### Generating

After getting the train_package, now choosing anyone file in train_package directory to generate the Google Trends data via:

```
$ python main.py --mode=generate --choice=1
```
The result would be stored in database/ directory

NOTICE:

The CHOICE is the name of directory in train_package.

Maybe blocked or banned by Google if you crawl too much data.


## Training and Plotting

Using the following command to train the Google Trends data.

```
$python main.py --mode=train --choice=1

```

The result would be stored in trend_result/ directory.

Then using command to plot the results.
```
$python main.py --mode=plot --choice=1

```

![Result Graph](https://github.com/SilongHu/Deep-Reinforcement-Learning-Application-in-Finance/blob/master/result_1.png)


## Built With

* [pytrends](https://github.com/GeneralMills/pytrends) - The Unofficial API for Google Trends
* [tensorflow](https://www.tensorflow.org) - An Open Source Machine Learning Framework


## Authors

* **SilongHu** - *Initial work* -

## Acknowledgments

* [Professor Bhaskar Krishnamachari](http://ceng.usc.edu/~bkrishna/)
* [Zhengyao Jiang](https://github.com/ZhengyaoJiang)

