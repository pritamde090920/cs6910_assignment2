# cs6910_assignment2

## Getting the code files
You need to first clone the github repository containing the files.
```
git clone https://github.com/pritamde090920/cs6910_assignment2.git
```
Then change into the code directory.
```
cd cs6910_assignment2
```
Make sure you are in the correct directory before proceeding further.


## Setting up the platform and environment
- ### Local machine
  If you are running the code on a local machine, then you need to have ython installed in the machine and pip command added in the environemnt variables.
  You can execute the following command to setup the environment and install all the required packages
  ```
  pip install -r requirements.txt
  ```
- ### Google colab/Kaggle
  If you are using google colab platform or kaggle, then you need to execute the follwoing code
  ```
  pip install wandb torch lightning pytorch_lightning matplotlib torchvision torchmetrics
  ```
This step will setup the environment required before proceeding.


## Project
The project deals with working with Convolutional Neural Networks(CNN). It is divided into two parts:
- ### Part A
  This part has a CNN model trained from scratch and corresponding train and test files implemented. The codes to run this part is present in the Part_A directory.
- ### Part B
  This part has a pre trained model called GoogLeNEt, which is fine tuned to work for the given dataset. The codes to run this part is present in the Part_B directory.


## Loading the dataset
The dataset needs to be placed in the home directory, i.e. in the cs6910_assignment2 directory.
If the directory is placed somewhere else, then to execute the files related to this project, you need to specify the absolute path of the root of the dataset.
For example, if you want to run train_parta.py with the dataset located somewhere else other than the home directory then you need to do
```
python train_parta.py --root <absoulte_path_of_dataset>
```
Same needs to be done for train_partb.py

#### Note
Do not give the paths for the train and val folders seperately. Just pass the absolute path of the root directory of the dataset, i.e. the directory ```inaturalist_12K``` as the argument. The train and val folders will be seperately handled inside the code itself.


## Part A
Make sure to change the directory by the command
```
cd Part_A
```

To train the model, you need to compile and execute the [train.py](https://github.com/pritamde090920/cs6910_assignment2/blob/main/Part_A/train_parta.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train_parta.py
```
By the above command, the model will run with the default configuration.\
To customize the run, you need to specify the parameters like ```python train_parta.py <*args>```\
For example,
```
python train_parta.py -e 20 -b 64 --filter 256 
```
The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Pritam CS6910 - Assignment 2 | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     cs23m051    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|     `-r`,`--root`        |../inaturalist_12K |Absolute path of the dataset")`                                         |
|     `-e`, `--epochs`     |       10      | Number of epochs to train neural network.                                 |
|   `-b`, `--batch`        |       8       | Batch size used to train neural network.                                  |
|   `-f`,`--filter`        |       64      | Number of filters in the first convolution layer                          |
|   `-fs`,`--filter_size`  |        7      | Dimension of the filters                                                  |
|    `-n`,`--neurons`      |     1024      | Number of neurons in the fully connected layer                            |
|    `-bn`,`--batch_normal`|     Yes       | choices: ['Yes','No']                                                     |
|   `-da`,`--data_augment` |      No       | choices: ['Yes','No']                                                     |
|   `-d`,`--dropout`       |       0       | Percentage of dropout in the network                                      |
|   `-a`,`--activation`    |     Mish      | Activation function in the activation layers                              |
|   `-fo`,`--filter_org`   |      same     | Organization of the filters across the layers                             |
|   `-t`,`--test`          |       0       | choices: [0,1]                                                            |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the key in the interactive command prompt.

## Testing the model
To test the model, you need to specify the test argument as 1. For example
```
python train_parta.py -t 1
```
This will run the model with default parameters and print the test accuracy.


## Part B
Make sure to change the directory by the command
```
cd Part_B
```

To train the model, you need to compile and execute the [train.py](https://github.com/pritamde090920/cs6910_assignment2/blob/main/Part_B/train_partb.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train_partb.py
```
By the above command, the model will run with the default configuration.\
To customize the run, you need to specify the parameters like ```python train_partb.py <*args>```\
For example,
```
python train_partb.py -e 20 -b 64 --freezed 10 
```

The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
|     `-r`,`--root`        |../inaturalist_12K |Absolute path of the dataset")`                                         |
|     `-e`, `--epochs`     |       10      | Number of epochs to train neural network.                                 |
|   `-b`, `--batch`        |       8       | Batch size used to train neural network.                                  |
|   `-lr`,`--learning`     |    1e-3       | Learning rate to train the model                                          |
|   `-fr`,`--freezed`      |      8        | Number of layers freezed from the beginning                               |
|   `-t`,`--test`          |       0       | choices: [0,1]                                                            |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the key in the interactive command prompt.

## Testing the model
To test the model, you need to specify the test argument as 1. For example
```
python train_partb.py -t 1
```
This will run the model with default parameters and print the test accuracy.


## Additional features
The following features are also supported
  - If you need some clarification on the arguments to be passed, then you can do
    ```
    python train_parta.py --help
    ```

## Links
[Wandb Report](https://wandb.ai/cs23m051/Pritam%20CS6910%20-%20Assignment%202/reports/CS6910-Assignment-2-Pritam-De-CS23M051--Vmlldzo3MzU5ODY3)
