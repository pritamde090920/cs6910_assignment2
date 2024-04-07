import torch
import LoadDataset
import ModelPreTrained
import lightning as pl

'''setting the device to gpu if avaiable'''
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''helper class to execeute train_partb.py'''
class PreTrained:
    def run(root,epochs,batch_size,learning_rate,freezed,test):
        '''
        Parameters:
            root : absolute path of the dataset
            epochs : number of epochs to run
            batch_size : batch size to split the dataset
            learning_rate : learning rate used to train the model
            freezed : number of layers freezed startinf from the input layer
            test : boolean variable denoting whether or not to test the model 
        Returns :
            None
        Function:
            Executes the Fine Tuning on the model
        '''

        '''loads dataset'''
        dataLoader=LoadDataset.DatasetLoader(root=root,batch_size=batch_size)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()
        
        '''setting number of output classes to 10'''
        numOfOutputClasses=10

        '''creating the object of the class and a trainer on it'''
        preTrainedModel=ModelPreTrained.FineTuningModel(numOfOutputClasses,freezed,learning_rate)

        trainer=pl.Trainer(max_epochs=epochs)

        '''executing train and validation steps'''
        trainer.fit(preTrainedModel,trainLoader,valLoader)

        '''if prompted then executing test step'''
        if test==1:
            trainer.test(preTrainedModel,testLoader)
            print("Test Accuracy : ",preTrainedModel.test_accuracy/len(testLoader))
