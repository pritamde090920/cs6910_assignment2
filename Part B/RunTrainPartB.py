import torch
import LoadDataset
import ModelPreTrained
import lightning as pl

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PreTrained:
    def run(epochs,batch_size,learning_rate,freezed,test):
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=batch_size)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()
        
        numOfOutputClasses=10

        preTrainedModel=ModelPreTrained.FineTuningModel(numOfOutputClasses,freezed,learning_rate)

        trainer=pl.Trainer(max_epochs=epochs)
        trainer.fit(preTrainedModel,trainLoader,valLoader)

        if test==1:
            trainer.test(preTrainedModel,testLoader)
