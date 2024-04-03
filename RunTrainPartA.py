import Model
import LoadDataset
import torch
import torch.nn as nn
from torchvision import transforms
from TrainModel import TrainModel

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Train:
    def runTrain(epochs,batch_size,numOfFilters,sizeFilter,sizeDenseUnits,batchNormalization,dataAugmentation,dropoutProb,activation,filterOrganization,test):
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=batch_size)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

        if dataAugmentation=="Yes":
            transform=transforms.Compose([
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
            trainLoader.dataset.transform=transform
            valLoader.dataset.transform=transform
            testLoader.dataset.transform=transform

        model=Model.CNN(inputDepth=3,numOfFilters=numOfFilters,sizeFilter=sizeFilter,stride=1,padding=2,sizeDenseUnits=sizeDenseUnits,filterOrganization=filterOrganization,activation=activation, batchNormalization=batchNormalization, dropoutProb=dropoutProb)
        if torch.cuda.device_count()>1:
            model=nn.DataParallel(model)
        model.to(device)

        TrainModel.accuracyAndLoss(model,device,trainLoader,valLoader,dataLoader,epochs=epochs)

        if(test==1):
            total=0
            correct=0
            with torch.no_grad():
                for data in testLoader:
                    inputs,labels=data[0].to(device), data[1].to(device)
                    outputs=model(inputs)
                    _,predicted=torch.max(outputs.data,1)
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum().item()

            print("\n=======================================================================\n\nTest Accuracy :",correct/total)