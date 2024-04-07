import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as pl
import torch.optim as optim

'''class to run pytorch lightning'''
class FastRunning(pl.LightningModule):

    '''constructor to set all the class parameters'''
    def __init__(self,model):
        super(FastRunning,self).__init__()
        '''setting the model defined by the CNN class'''
        self.model=model

        '''setting the loss type'''
        self.criterion=nn.CrossEntropyLoss()

        '''metrics to store the accuracies'''
        self.training_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=10)
        self.validation_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=10)
    
    def configure_optimizers(self):
        '''
        Parameters:
            None
        Returns :
            optimizer : the optimizer object that will be applied on the network
        Function:
            Creates an object of the optimizer and returns it
        '''
        optimizer=optim.SGD(self.model.parameters(),lr=0.0001,momentum=0.8,nesterov=True)
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
        y_hat=self.model(x)
        loss=self.criterion(y_hat,y)
        predicted=torch.argmax(y_hat,dim=1)
        self.training_accuracy(predicted,y)
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
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        predicted = torch.argmax(y_hat,dim=1)
        self.validation_accuracy(predicted,y)
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