import wandb
from RunTrainPartA import Train
import warnings
warnings.filterwarnings("ignore")
import argparse

'''login to wandb to generate plot'''
wandb.login()

def arguments():
    '''
      Parameters:
        None
      Returns :
        A parser object
      Function:
        Does command line argument parsing and returns the arguments passed
    '''
    commandLineArgument=argparse.ArgumentParser(description='Model Parameters')
    commandLineArgument.add_argument('-wp','--wandb_project',help="Project name used to track experiments in Weights & Biases dashboard")
    commandLineArgument.add_argument('-we','--wandb_entity',help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    commandLineArgument.add_argument('-r','--root',help="Absolute path of the dataset")
    commandLineArgument.add_argument('-e','--epochs',type = int,help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch',type=int,help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-f','--filter',type=int,help="Number of filters in the first convolution layer")
    commandLineArgument.add_argument('-fs','--filter_size',type=int,help="Dimension of the filters")
    commandLineArgument.add_argument('-n','--neurons',type=int,help="Number of neurons in the fully connected layer")
    commandLineArgument.add_argument('-bn','--batch_normal',help="choices: ['Yes','No']")
    commandLineArgument.add_argument('-da','--data_augment',help="choices: ['Yes','No']")
    commandLineArgument.add_argument('-d','--dropout',type=float,help="Percentage of dropout in the network")
    commandLineArgument.add_argument('-a','--activation',help="Activation function in the activation layers")
    commandLineArgument.add_argument('-fo','--filter_org',help="Organization of the filters across the layers")
    commandLineArgument.add_argument('-t','--test',type = int,help="choices: [0,1]")

    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    project_name='Pritam CS6910 - Assignment 2'
    entity_name='cs23m051'
    epochs=10
    batch_size=8
    numOfFilters=64
    sizeFilter=7
    sizeDenseUnits=1024
    batchNormalization="Yes"
    dataAugmentation="No"
    dropoutProb=0
    activation="Mish"
    filterOrganization="same"
    test=0
    root='../inaturalist_12K'

    '''call to argument function to get the arguments'''
    argumentsPassed=arguments()

    '''checking if a particular argument is passed thorugh commadn line or not and updating the values accordingly'''
    if argumentsPassed.wandb_project is not None:
        project_name=argumentsPassed.wandb_project
    if argumentsPassed.wandb_entity is not None:
        entity_name=argumentsPassed.wandb_entity
    if argumentsPassed.root is not None:
        root=argumentsPassed.root
    if argumentsPassed.epochs is not None:
        epochs=argumentsPassed.epochs
    if argumentsPassed.batch is not None:
        batch_size=argumentsPassed.batch
    if argumentsPassed.filter is not None:
        numOfFilters=argumentsPassed.filter
    if argumentsPassed.filter_size is not None:
        sizeFilter=argumentsPassed.filter_size
    if argumentsPassed.neurons is not None:
        sizeDenseUnits=argumentsPassed.neurons
    if argumentsPassed.batch_normal is not None:
        batchNormalization=argumentsPassed.batch_normal
    if argumentsPassed.data_augment is not None:
        dataAugmentation=argumentsPassed.data_augment
    if argumentsPassed.dropout is not None:
        dropoutProb=argumentsPassed.dropout
    if argumentsPassed.activation is not None:
        activation=argumentsPassed.activation
    if argumentsPassed.filter_org is not None:
        filterOrganization=argumentsPassed.filter_org
    if argumentsPassed.test is not None:
        test=argumentsPassed.test

    '''initializing to the project'''
    wandb.init(project=project_name,entity=entity_name)

    '''calling the functions with the parameters'''
    run="EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}".format(epochs,numOfFilters,sizeFilter,sizeDenseUnits,filterOrganization,activation,dropoutProb)
    print("run name = {}".format(run))
    wandb.run.name=run
    Train.runTrain(project_name,root,epochs,batch_size,numOfFilters,sizeFilter,sizeDenseUnits,batchNormalization,dataAugmentation,dropoutProb,activation,filterOrganization,test)
    wandb.finish()

if __name__ == '__main__':
    main()