import Model
import LoadDataset
import torch
import torch.nn as nn
from torchvision import transforms
import wandb
from TrainModel import TrainModel

wandb.login()

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    '''initialize to project and create a config'''
    wandb.init(project="Pritam CS6910 - Assignment 2")
    config=wandb.config

    dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=config.batch_size)
    trainLoader,valLoader,testLoader=dataLoader.data_loaders()

    dataAugmentation=config.data_augment
    if dataAugmentation=="Yes":
        transform=transforms.Compose([
            transforms.Resize((64,64)),#224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])
        trainLoader.dataset.transform=transform
        valLoader.dataset.transform=transform
        testLoader.dataset.transform=transform
    
    run="EP_{}_FIL_{}_FILSIZE_{}_FCSIZE_{}_FILORG_{}_AC_{}_DRP_{}".format(config.epochs,config.number_of_filters,config.size_of_filter,config.neurons_in_fc,config.filter_organization,config.activation_function,config.dropout)
    print("run name = {}".format(run))
    wandb.run.name=run
    model=Model.CNN(inputDepth=config.input_depth,numOfFilters=config.number_of_filters,sizeFilter=config.size_of_filter,stride=1,padding=2,sizeDenseUnits=config.neurons_in_fc,filterOrganization=config.filter_organization,activation=config.activation_function, batchNormalization=config.batch_normalization, dropoutProb=config.dropout)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    model.to(device)
    TrainModel.accuracyAndLoss(model,device,trainLoader,valLoader,dataLoader,epochs=config.epochs)



configuration_values={
    'method': 'bayes',
    'name': 'ACCURACY AND LOSS',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
    },
    'parameters': {
        'batch_size' : {'values' : [32,64,128]},
        'epochs' : {'values' : [5,10]},
        'input_depth' : {'values' : [3]},
        'number_of_filters' : {'values' : [32,64,128,256,512]},
        'size_of_filter' : {'values' : [5,7,11]},
        'neurons_in_fc' : {'values' : [32,64,128]},
        'batch_normalization' : {'values' : ['Yes','No']},
        'data_augment' : {'values' : ['Yes','No']},
        'dropout' : {'values' : [0,0.2,0.4]},
        'activation_function' : {'values' : ['ReLU','GELU','SiLU','Mish']},
        'filter_organization' : {'values' : ['same','half','double']}
    }
}

sweep_id=wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 2')
wandb.agent(sweep_id,function=main,count=150)
wandb.finish()