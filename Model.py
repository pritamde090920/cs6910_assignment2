import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, inputDepth, numOfFilters, sizeFilter, stride, padding, sizeDenseUnits, filterOrganization, activation, batchNormalization, dropoutProb):
        super(CNN, self).__init__()

        self.inputDepth=inputDepth
        self.outputDepth=numOfFilters
        self.stride=stride
        self.padding=padding
        self.neruonsInDenseLayer=sizeDenseUnits
        self.filterSize=sizeFilter
        self.filterOrganization=filterOrganization
        self.activation=activation
        self.batchNormalization=batchNormalization
        self.dropoutProb=dropoutProb

        self.widhtOfImage=64

        self.convLayer1=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        if self.batchNormalization=="Yes":
            self.bacthNormal1=nn.BatchNorm2d(self.outputDepth)
        else:
            self.bacthNormal1=nn.Identity()
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer1=self.activationFunction(self.activation)
        self.maxPool1=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        self.convLayer2=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        if self.batchNormalization=="Yes":
            self.bacthNormal2=nn.BatchNorm2d(self.outputDepth)
        else:
            self.bacthNormal2=nn.Identity()
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer2=self.activationFunction(self.activation)
        self.maxPool2=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        self.convLayer3=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        if self.batchNormalization=="Yes":
            self.bacthNormal3=nn.BatchNorm2d(self.outputDepth)
        else:
            self.bacthNormal3=nn.Identity()
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer3=self.activationFunction(self.activation)
        self.maxPool3=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        self.convLayer4=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        if self.batchNormalization=="Yes":
            self.bacthNormal4=nn.BatchNorm2d(self.outputDepth)
        else:
            self.bacthNormal4=nn.Identity()
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer4=self.activationFunction(self.activation)
        self.maxPool4=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        self.convLayer5=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        if self.batchNormalization=="Yes":
            self.bacthNormal5=nn.BatchNorm2d(self.outputDepth)
        else:
            self.bacthNormal5=nn.Identity()
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer5=self.activationFunction(self.activation)
        self.maxPool5=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        self.flatten=nn.Flatten()

        self.dropout=nn.Dropout(self.dropoutProb)

        self.fullyConnected=nn.Linear(self.widhtOfImage*self.widhtOfImage*self.outputDepth,self.neruonsInDenseLayer)
        self.activationLayer6=self.activationFunction(self.activation)
        self.outputLayer=nn.Linear(self.neruonsInDenseLayer,10)
    
    def filterSizeCalculator(self,filterSize,filterOrganization):
        if(filterOrganization=="same"):
            return filterSize
        if(filterOrganization=="half"):
            return filterSize//2
        if(filterOrganization=="double"):
            return filterSize*2
    
    def activationFunction(self,activation):
        if activation=="ReLU":
            return nn.ReLU()
        if activation=="GELU":
            return nn.GELU()
        if activation=="SiLU":
            return nn.SiLU()
        if activation=="Mish":
            return nn.Mish()

    def forward(self,x):
        x=self.maxPool1(self.activationLayer1(self.bacthNormal1(self.convLayer1(x))))
        x=self.maxPool2(self.activationLayer2(self.bacthNormal2(self.convLayer2(x))))
        x=self.maxPool3(self.activationLayer3(self.bacthNormal3(self.convLayer3(x))))
        x=self.maxPool4(self.activationLayer4(self.bacthNormal4(self.convLayer4(x))))
        x=self.maxPool5(self.activationLayer5(self.bacthNormal5(self.convLayer5(x))))
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.fullyConnected(x)
        x=self.activationLayer6(x)
        x=self.outputLayer(x)
        x=F.softmax(x,dim=1)
        return x