import Question1
import LoadDataset
import torch
import torch.nn as nn
from torchvision import transforms
import wandb
import lightning as pl
from pytorch_lightning.loggers import WandbLogger
from LightliningModel import FastRunning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

'''login to wandb to generate plot'''
wandb.login()

'''setting the device to gpu if it is available'''
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    '''initialize to project and create a config'''
    wandb.init(project="Pritam CS6910 - Assignment 2")
    config=wandb.config

    '''get the data loader'''
    dataLoader=LoadDataset.DatasetLoader(root='../inaturalist_12K',batch_size=config.batch_size)
    trainLoader,valLoader,testLoader=dataLoader.data_loaders()

    '''if data augmentation is needed then apply it'''
    dataAugmentation=config.data_augment
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
    run="EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}_BS_{}".format(config.epochs,config.number_of_filters,config.size_of_filter,config.neurons_in_fc,config.filter_organization,config.activation_function,config.dropout,config.batch_size)
    wandb.run.name=run

    '''define and apply early stopping'''
    early_stop_callback = EarlyStopping(monitor="validation_accuracy", min_delta=0.01, patience=3, verbose=False, mode="max")
    
    '''create and object of the CNN class that will have the model defined in it'''
    model=Question1.CNN(inputDepth=config.input_depth,numOfFilters=config.number_of_filters,sizeFilter=config.size_of_filter,stride=1,padding=2,sizeDenseUnits=config.neurons_in_fc,filterOrganization=config.filter_organization,activation=config.activation_function, batchNormalization=config.batch_normalization, dropoutProb=config.dropout)
    
    '''define an object of the lightning class adn pass the CNN class model into it to run it'''
    lightningModel=FastRunning(model)

    '''set a wandb logger'''
    wandb_logger=WandbLogger(project='Pritam CS6910 - Assignment 2',log_model='all')
    
    '''define a trainer object to run the lightning object'''
    trainer=pl.Trainer(max_epochs=config.epochs,logger=wandb_logger)
    if device!=torch.device('cpu'):
        trainer=pl.Trainer(max_epochs=config.epochs,devices=-1,logger=wandb_logger)

    '''training and validation step'''
    trainer.fit(lightningModel,trainLoader,valLoader)


'''setting the configuration values'''
configuration_values={
    'method': 'bayes',
    'name': 'ACCURACY AND LOSS',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
    },
    'parameters': {
        'batch_size' : {'values' : [8,16,32,64,128]},
        'epochs' : {'values' : [5,10,15,20]},
        'input_depth' : {'values' : [3]},
        'number_of_filters' : {'values' : [32,64,128,256,512]},
        'size_of_filter' : {'values' : [5,7,11]},
        'neurons_in_fc' : {'values' : [32,64,128,256,512,1024]},
        'batch_normalization' : {'values' : ['Yes','No']},
        'data_augment' : {'values' : ['Yes','No']},
        'dropout' : {'values' : [0,0.2,0.4]},
        'activation_function' : {'values' : ['ReLU','GELU','SiLU','Mish']},
        'filter_organization' : {'values' : ['same','half','double']},
    }
}

'''running the sweep by creating an agent'''
sweep_id=wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 2')
wandb.agent(sweep_id,function=main,count=150)
wandb.finish()