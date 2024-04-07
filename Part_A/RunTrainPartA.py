import Model
import LoadDataset
import torch
import torch.nn as nn
from torchvision import transforms
from TrainModel import TrainModel
from LightliningModel import FastRunning
from pytorch_lightning.callbacks import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Train:
    def runTrain(project_name,epochs,batch_size,numOfFilters,sizeFilter,sizeDenseUnits,batchNormalization,dataAugmentation,dropoutProb,activation,filterOrganization,test):
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=batch_size)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

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

        early_stop_callback = EarlyStopping(monitor="validation_accuracy", min_delta=0.01, patience=3, verbose=False, mode="max")
        model=Model.CNN(inputDepth=3,numOfFilters=numOfFilters,sizeFilter=sizeFilter,stride=1,padding=2,sizeDenseUnits=sizeDenseUnits,filterOrganization=filterOrganization,activation=activation, batchNormalization=batchNormalization, dropoutProb=dropoutProb)
        lightningModel=FastRunning(model)
        wandb_logger=WandbLogger(project=project_name,log_model='all')
        trainer=pl.Trainer(max_epochs=epochs,devices=-1,logger=wandb_logger,callbacks=[early_stop_callback])

        trainer.fit(lightningModel,trainLoader,valLoader)

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