import torch
import torchvision
import torch.nn as nn
import lightning as pl
import re
import torchmetrics
import torch.optim as optim

class FineTuningModel(pl.LightningModule):
    def __init__(self, numClasses, numOfFreezedLayers, learning_rate, aux_logits=True):
        super(FineTuningModel, self).__init__()
        
        self.model=torchvision.models.googlenet(pretrained=True)
        self.numOfFreezedLayers=numOfFreezedLayers
        self.learning_rate=learning_rate

        for name,param in self.model.named_parameters():
            match=re.search(r'\d+',name.split('.')[0])
            if match and int(match.group())<self.numOfFreezedLayers:
                param.requires_grad=False
        
        layers=list(self.model.children())[:-1]
        self.feature_extractor=nn.Sequential(*layers)
        self.feature_extractor.eval()
        inFeatures=self.model.fc.in_features
        self.classifier=nn.Linear(inFeatures,numClasses)
        self.criterion=nn.CrossEntropyLoss()

        self.training_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=numClasses)
        self.validation_accuracy=torchmetrics.Accuracy(task="multiclass",num_classes=numClasses)
        self.test_accuracy=0
        
    def forward(self, x):
        representation=self.feature_extractor(x).flatten(1)
        x=self.classifier(representation)
        return x
    
    def configure_optimizers(self):
        optimizer=optim.Adam(self.model.parameters(),lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat,y)
        
        self.training_accuracy(y_hat,y)
        self.log("training_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log('training_accuracy',self.training_accuracy.compute(),prog_bar=True,logger=True)
        return self.training_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat, y)

        self.validation_accuracy(y_hat,y)
        self.log("validation_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log("validation_accuracy",self.validation_accuracy.compute(),prog_bar=True,logger=True)
        return self.validation_accuracy.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred_labels = torch.argmax(y_hat, dim=1)
        correct = (pred_labels == y).sum().item()
        total = len(y)
        accuracy = correct / total
        self.test_accuracy+=accuracy
        return accuracy