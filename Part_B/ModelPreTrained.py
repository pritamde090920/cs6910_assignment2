import torch
import torchvision
import torch.nn as nn
import lightning as pl
import re
import torchmetrics
import torch.optim as optim

'''class to perform fine tuning'''
class FineTuningModel(pl.LightningModule):

    '''constructor to set all the class parameters'''
    def __init__(self, numClasses, numOfFreezedLayers, learning_rate, aux_logits=True):
        super(FineTuningModel, self).__init__()
        
        '''loads the GoogLeNet model'''
        self.model=torchvision.models.googlenet(pretrained=True)
        self.numOfFreezedLayers=numOfFreezedLayers
        self.learning_rate=learning_rate

        '''freezes the layers'''
        for n,p in self.model.named_parameters():
            match=re.search(r'\d+',n.split('.')[0])
            if match and int(match.group())<self.numOfFreezedLayers:
                p.requires_grad=False
        
        numOfLayers=list(self.model.children())[:-1]
        self.feature_extractor=nn.Sequential(*numOfLayers)
        self.feature_extractor.eval()

        '''attaches a layer to comply with the output layers of iNaturalist dataset'''
        inFeatures=self.model.fc.in_features
        self.outputLayer=nn.Linear(inFeatures,numClasses)

        '''defining the loss'''
        self.criterion=nn.CrossEntropyLoss()

        '''metrics to store the accuracies'''
        self.training_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=numClasses)
        self.validation_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=numClasses)
        self.test_accuracy=0
        
    def forward(self, x):
        '''
        Parameters:
            x : object to apply forward propagation on
        Returns :
            x : the same object after application of forward propagation
        Function:
            Applies forward propagation
        '''
        flattened=self.feature_extractor(x).flatten(1)
        x=self.outputLayer(flattened)
        return x
    
    def configure_optimizers(self):
        '''
        Parameters:
            None
        Returns :
            optimizer : the optimizer object that will be applied on the network
        Function:
            Creates an object of the optimizer and returns it
        '''
        optimizer=optim.Adam(self.model.parameters(),lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        '''
        Parameters:
            batch : images in batches
            batch_idx : batch id of the corresponding batch
        Returns :
            loss : loss obtained after backpropagation
        Function:
            Does the training step
        '''
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat,y)
        
        self.training_accuracy(y_hat,y)
        self.log("training_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        '''
        Parameters:
            None
        Returns :
            training accuracy after reseting the metric to 0
        Function:
            Logs training accuracy after end of epoch
        '''
        self.log('training_accuracy',self.training_accuracy.compute(),prog_bar=True,logger=True)
        return self.training_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        '''
        Parameters:
            batch : images in batches
            batch_idx : batch id of the corresponding batch
        Returns :
            loss : loss obtained after backpropagation
        Function:
            Does the validation step
        '''
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat, y)

        self.validation_accuracy(y_hat,y)
        self.log("validation_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        '''
        Parameters:
            None
        Returns :
            validation accuracy after reseting the metric to 0
        Function:
            Logs validation accuracy after end of epoch
        '''
        self.log("validation_accuracy",self.validation_accuracy.compute(),prog_bar=True,logger=True)
        return self.validation_accuracy.reset()
    
    def test_step(self, batch, batch_idx):
        '''
        Parameters:
            batch : images in batches
            batch_idx : batch id of the corresponding batch
        Returns :
            accuracy : test accuracy accumulated after all the batches till the current batch
        Function:
            Calculates and returns the test accuracy for all the batches
        '''
        x, y=batch
        y_hat=self(x)
        predicted=torch.argmax(y_hat, dim=1)
        correct_points=(predicted==y).sum().item()
        total_points=len(y)
        accuracy=correct_points/total_points
        self.test_accuracy+=accuracy
        return accuracy