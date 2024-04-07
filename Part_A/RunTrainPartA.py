import Question1
import LoadDataset
import torch
import torch.nn as nn
from torchvision import transforms
from LightliningModel import FastRunning
from pytorch_lightning.loggers import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb

'''setting the device to gpu if it is available'''
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''class to support the train_parta.py execution'''
class Train:
    def runTrain(project_name,root,epochs,batch_size,numOfFilters,sizeFilter,sizeDenseUnits,batchNormalization,dataAugmentation,dropoutProb,activation,filterOrganization,test):
        '''
        Parameters:
            project_name : naem of the wandb project
            root : absolute path of the dataset
            epochs : number of epochs to run
            batch_size : size of batch to divide the dataset into
            numOfFilters : number of filters in the input layer
            sizeFilter : size(dimension) of each filter
            sizeDenseUnits : Number of neurons in the dense layer
            bacthNormalization : boolean value indicating wheteher to apply batch normalization or not
            dataAugmentation : boolean value indicating wheteher to apply data augmentation or not
            dropoutProb : probability of dropout
            activation : activation fucntion that is to be applied
            filterOrganization : organization of the filters across the layers
            test : boolean value indicating wheteher to do testing or not
        Returns :
            None
        Function:
            Supports the execution of train_parta.py
        '''

        '''get the data loaders'''
        dataLoader=LoadDataset.DatasetLoader(root=root,batch_size=batch_size)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

        '''if needed then apply data augmentation'''
        if dataAugmentation=="Yes":
            transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4747786223888397,0.4644955098628998,0.3964916169643402],std=[0.2389, 0.2289, 0.2422]),
            ])
            trainLoader.dataset.transform=transform
            valLoader.dataset.transform=transform
            testLoader.dataset.transform=transform

        '''set wandb run name'''
        run="EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}_BS_{}".format(epochs,numOfFilters,sizeFilter,sizeDenseUnits,filterOrganization,activation,dropoutProb,batch_size)
        wandb.run.name=run

        '''define and apply early stopping'''
        early_stop_callback = EarlyStopping(monitor="validation_accuracy", min_delta=0.01, patience=3, verbose=False, mode="max")

        '''create and object of the CNN class that will have the model defined in it'''
        model=Question1.CNN(inputDepth=3,numOfFilters=numOfFilters,sizeFilter=sizeFilter,stride=1,padding=2,sizeDenseUnits=sizeDenseUnits,filterOrganization=filterOrganization,activation=activation, batchNormalization=batchNormalization, dropoutProb=dropoutProb)

        '''define an object of the lightning class adn pass the CNN class model into it to run it'''
        lightningModel=FastRunning(model)

        '''set a wandb logger'''
        wandb_logger=WandbLogger(project=project_name,log_model='all')

        '''define a trainer object to run the lightning object'''
        trainer=pl.Trainer(max_epochs=epochs,logger=wandb_logger)
        if device!=torch.device('cpu'):
            trainer=pl.Trainer(max_epochs=epochs,devices=-1,logger=wandb_logger)

        '''training and validation step'''
        trainer.fit(lightningModel,trainLoader,valLoader)

        '''if needed then run test and print the accuracy'''
        if(test==1):
            correct=0
            total=0

            for image,label in testLoader:
                with torch.no_grad():
                    y_hat=lightningModel.model(image)
                    _,predicted=torch.max(y_hat,1)
                    total+=label.size(0)
                    correct+=(predicted==label).sum().item()

            print("Test Accuracy : ",correct/total)