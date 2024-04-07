import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as pl
import torch.optim as optim

class FastRunning(pl.LightningModule):
    def __init__(self,model):
        super(FastRunning,self).__init__()
        self.model=model
        self.criterion=nn.CrossEntropyLoss()

        self.training_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=10)
        self.validation_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=10)
    
    def configure_optimizers(self):
        optimizer=optim.SGD(self.model.parameters(),lr=0.0001,momentum=0.8,nesterov=True)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        y_hat=self.model(x)
        loss=self.criterion(y_hat,y)
        
        # Calculate training accuracy
        predicted = torch.argmax(y_hat,dim=1)
        self.training_accuracy(predicted,y)
        self.log("training_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log('training_accuracy',self.training_accuracy.compute(),prog_bar=True,logger=True)
        return self.training_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        # Calculate validation accuracy
        predicted = torch.argmax(y_hat,dim=1)
        self.validation_accuracy(predicted,y)
        self.log("validation_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log("validation_accuracy",self.validation_accuracy.compute(),prog_bar=True,logger=True)
        return self.validation_accuracy.reset()
    
    def test_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self.model(x)
        preds = torch.argmax(y_hat, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        test_accuracy=correct/total
        return test_accuracy