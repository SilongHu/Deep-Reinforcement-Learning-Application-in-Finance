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
Move to the [USERGUIDE](https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/user_guide.md) for details. 
And /cd that directory

```
$./configure
$[sudo] make && make install
```

### Generating

After installing the MPICH, then install MPI4PY.

```
$[sudo] pip install mpi4py
```


End with an example of getting some data out of the system, download [helloworld example](https://github.com/jbornschein/mpi4py-examples/blob/master/01-hello-world) and look at the following result

```
$ mpiexec -n 4 python ./helloworld
Hello! I'm rank 0 from 4 running in total...
Hello! I'm rank 1 from 4 running in total...
Hello! I'm rank 2 from 4 running in total...
Hello! I'm rank 3 from 4 running in total...
```

## Running the tests

In our project, we predefine a task graph, it's a DAG (Directed Acyclic Graph)

![Task Graph](https://github.com/SilongHu/USC-Directed-Research/blob/master/taskGraph.png)

Task A: sqaure root of an input

Task B: power of 5 from output A

Task C: power of 3 from output A

Task D: result_B + result_C + 1


Using the following command to assign task to each device. (Assign task A on device 0, B on 1, C on 1 and D on 2)

```
$python main.py 0 1 1 2

task_A should be finished on processor 0
task_B should be finished on processor 1
task_C should be finished on processor 1
The result is 271.0
task_D should be finished on processor 2
Execute time :0.217902898788 seconds

```

The input number is defined in the main.py, for our main.py, it creates the execute.py and then uses mpiexec to run it.

## Deployment

For this project, we just use the simple 'send' and 'recv' communication, for further usage, please go to [more MPI4PY usage](http://pythonhosted.org/mpi4py/usrman/index.html)

Also, the [MPI could run on server cluster in LAN](http://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/) and communicate with each other. If interested, please try it.


## Built With

* [MPI4PY](http://pythonhosted.org/mpi4py/) - The distributed computing tool
* [Argparse](https://docs.python.org/3/library/argparse.html) - Modified input task-device assignment


## Authors

* **SilongHu** - *Initial work* -

## Acknowledgments

* [Professor Bhaskar Krishnamachari](http://ceng.usc.edu/~bkrishna/)
* [Ph.D Pranav Sakulkar](http://www-scf.usc.edu/~sakulkar/)
* [Ph.D Kwame-Lante Wright](http://www-scf.usc.edu/~kwamelaw/)

