import wandb
from Part_B.RunTrainPartB import PreTrained
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
    commandLineArgument.add_argument('-e','--epochs',type = int,help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch',type=int,help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-lr','--learning',type=int,help="Learning rate to train the model")
    commandLineArgument.add_argument('-fr','--freezed',type=int,help="Number of layers unfreezed from the last")
    commandLineArgument.add_argument('-t','--test',type=int,help="choices: [0,1]")
    
    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    epochs=10
    batch_size=8
    learning_rate=1e-3
    freezed=8
    test=0

    '''call to argument function to get the arguments'''
    argumentsPassed=arguments()

    '''checking if a particular argument is passed thorugh commadn line or not and updating the values accordingly'''
    if argumentsPassed.epochs is not None:
        epochs=argumentsPassed.epochs
    if argumentsPassed.batch is not None:
        batch_size=argumentsPassed.batch
    if argumentsPassed.learning is not None:
        learning_rate=argumentsPassed.learning
    if argumentsPassed.freezed is not None:
        freezed=argumentsPassed.freezed
    if argumentsPassed.test is not None:
        test=argumentsPassed.test


    '''calling the functions with the parameters'''
    PreTrained.run(epochs,batch_size,learning_rate,freezed,test)
    wandb.finish()

if __name__ == '__main__':
    main()