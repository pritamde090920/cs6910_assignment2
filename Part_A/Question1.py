import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

'''class to define the CNN model'''
class CNN(nn.Module):
    def __init__(self, inputDepth, numOfFilters, sizeFilter, stride, padding, sizeDenseUnits, filterOrganization, activation, batchNormalization, dropoutProb):
        '''constructor to set all the class parameters'''
        super(CNN, self).__init__()

        '''setting the hyperparameters'''
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

        '''setting image size to 112'''
        self.widhtOfImage=112

        '''first convolution-activation-maxpool block'''
        self.convLayer1=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.bacthNormal1=nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer1=self.activationFunction(self.activation)
        self.maxPool1=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        '''setting input depth and calculating filter size for next layer'''
        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        '''second convolution-activation-maxpool block'''
        self.convLayer2=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.bacthNormal2=nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer2=self.activationFunction(self.activation)
        self.maxPool2=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        '''setting input depth and calculating filter size for next layer'''
        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        '''third convolution-activation-maxpool block'''
        self.convLayer3=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.bacthNormal3=nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer3=self.activationFunction(self.activation)
        self.maxPool3=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        '''setting input depth and calculating filter size for next layer'''
        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        '''fourth convolution-activation-maxpool block'''
        self.convLayer4=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.bacthNormal4=nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer4=self.activationFunction(self.activation)
        self.maxPool4=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        '''setting input depth and calculating filter size for next layer'''
        self.inputDepth=self.outputDepth
        self.outputDepth=self.filterSizeCalculator(self.inputDepth,self.filterOrganization)

        '''fifth convolution-activation-maxpool block'''
        self.convLayer5=nn.Conv2d(in_channels=self.inputDepth,out_channels=self.outputDepth,kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.bacthNormal5=nn.BatchNorm2d(self.outputDepth)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1
        self.activationLayer5=self.activationFunction(self.activation)
        self.maxPool5=nn.MaxPool2d(kernel_size=self.filterSize,stride=self.stride,padding=self.padding)
        self.widhtOfImage=((self.widhtOfImage-self.filterSize+2*self.padding)//self.stride)+1

        '''flattening the output of the last maxpool layer'''
        self.flatten=nn.Flatten()

        '''applying dropout'''
        self.dropout=nn.Dropout(self.dropoutProb)

        '''defining the dense layer'''
        self.fullyConnected=nn.Linear(self.widhtOfImage*self.widhtOfImage*self.outputDepth,self.neruonsInDenseLayer)

        '''applying activation function on the dense layer'''
        self.activationLayer6=self.activationFunction(self.activation)

        '''defining the output layer'''
        self.outputLayer=nn.Linear(self.neruonsInDenseLayer,10)
    
    def filterSizeCalculator(self,filterSize,filterOrganization):
        '''
        Parameters:
            filterSize : current size of the filter
            filterOrganization : what type of organization to apply of the filter size
        Returns :
            filterSize : size of filter after applying the update rule
        Function:
            Applies filter organization
        '''
        if(filterOrganization=="same"):
            return filterSize
        if(filterOrganization=="half" and filterSize>1):
            return filterSize//2
        if(filterOrganization=="double" and filterSize<=512):
            return filterSize*2
        return filterSize
    
    def activationFunction(self,activation):
        '''
        Parameters:
            activation : what type of activation function is to be applied
        Returns :
            object of the activation function
        Function:
            Creates and returns an object of the activation function
        '''
        if activation=="ReLU":
            return nn.ReLU()
        if activation=="GELU":
            return nn.GELU()
        if activation=="SiLU":
            return nn.SiLU()
        if activation=="Mish":
            return nn.Mish()

    def forward(self,x):
        '''
        Parameters:
            x : object to apply forward propagation on
        Returns :
            x : the same object after application of forward propagation
        Function:
            Applies forward propagation
        '''

        '''if batchnormalization is needed then apply it else do not apply it'''
        if(self.batchNormalization=="Yes"):
            x=self.maxPool1(self.activationLayer1(self.bacthNormal1(self.convLayer1(x))))
        else:
            x=self.maxPool1(self.activationLayer1(self.convLayer1(x)))
        if(self.batchNormalization=="Yes"):
            x=self.maxPool2(self.activationLayer2(self.bacthNormal2(self.convLayer2(x))))
        else:
            x=self.maxPool2(self.activationLayer2(self.convLayer2(x)))
        if(self.batchNormalization=="Yes"):
            x=self.maxPool3(self.activationLayer3(self.bacthNormal3(self.convLayer3(x))))
        else:
            x=self.maxPool3(self.activationLayer3(self.convLayer3(x)))
        if(self.batchNormalization=="Yes"):
            x=self.maxPool4(self.activationLayer4(self.bacthNormal4(self.convLayer4(x))))
        else:
            x=self.maxPool4(self.activationLayer4(self.convLayer4(x)))
        if(self.batchNormalization=="Yes"):
            x=self.maxPool5(self.activationLayer5(self.bacthNormal5(self.convLayer5(x))))
        else:
            x=self.maxPool5(self.activationLayer5(self.convLayer5(x)))
        
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.fullyConnected(x)
        x=self.activationLayer6(x)
        x=self.outputLayer(x)
        return x
