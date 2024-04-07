import ModelPreTrained
import LoadDataset
import lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.login()

def main():
    wandb.init(project="Pritam CS6910 - Assignment 2")
    config=wandb.config
    
    dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=config.batch_size)
    trainLoader,valLoader,testLoader=dataLoader.data_loaders()
    
    numOfOutputClasses=10

    preTrainedModel=ModelPreTrained.FineTuningModel(numOfOutputClasses,config.freezed_layers,config.learning_rate)
    
    run="EP_{}_BS_{}_LR_{}_FREEZED_{}".format(config.epochs,config.batch_size,config.learning_rate,config.freezed_layers)
    wandb.run.name=run
    wandb_logger=WandbLogger(project='Pritam CS6910 - Assignment 2',log_model='all')
    trainer=pl.Trainer(max_epochs=config.epochs,logger=wandb_logger)

    trainer.fit(preTrainedModel,trainLoader,valLoader)
    trainer.test(preTrainedModel,testLoader)
    print("Test Accuracy : ",preTrainedModel.test_accuracy)

configuration_values={
    'method': 'bayes',
    'name': 'ACCURACY AND LOSS',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
    },
    'parameters': {
        'batch_size' : {'values' : [8,16,32]},
        'epochs' : {'values' : [5,10]},
        'learning_rate' : {'values' : [1e-3,1e-4]},
        'freezed_layers' : {'values' : [8,9,10]},
    }
}

sweep_id=wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 2')
wandb.agent(sweep_id,function=main,count=20)
wandb.finish()