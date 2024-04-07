import Question1
import LoadDataset
import torch
import wandb
import torchvision
import matplotlib.pyplot as plt
import lightning as pl
from LightliningModel import FastRunning
from torchvision import transforms

'''setting the device to gpu if it is available'''
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''class to run test on my best model'''
class TestBestModel:
    def testAccuracy():
        '''
        Parameters:
            None
        Returns :
            None
        Function:
            Runs test on my best model
        '''

        '''login to the wandb project'''
        wandb.login()
        wandb.init(project="Pritam CS6910 - Assignment 2",name="Part A Test plot")

        '''setting the device if it is available'''
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        '''get the dataset loaders'''
        dataLoader=LoadDataset.DatasetLoader(root='../inaturalist_12K',batch_size=8)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

        '''running training and validation'''
        model=Question1.CNN(inputDepth=3,numOfFilters=64,sizeFilter=7,stride=1,padding=2,sizeDenseUnits=1024,filterOrganization="same",activation="Mish", batchNormalization="Yes", dropoutProb=0)
        lightningModel=FastRunning(model)
        trainer=pl.Trainer(max_epochs=10)
        if device!=torch.device('cpu'):
            trainer=pl.Trainer(max_epochs=10,devices=-1)
        trainer.fit(lightningModel,trainLoader,valLoader)

        '''loading the test loader with a batch size of 1 on a shuffled test dataset to get the 30 random images'''
        images=list()
        predictClass=list()
        trueClass=list()
        class_names=testLoader.dataset.classes
        correct=0
        total=0

        transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
        ])
        test_dataset=torchvision.datasets.ImageFolder(root='../inaturalist_12K',transform=transform)
        testLoader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)

        '''running test and printing the accuracy'''
        for image,label in testLoader:
            with torch.no_grad():
                y_hat=lightningModel.model(image)
            _,predict=torch.max(y_hat,dim=1)
            if(predict==label):
              correct+=1
            total+=1

            if total<=30:
                images.append(image.squeeze(0))
                predictClass.append(class_names[predict])
                trueClass.append(class_names[label])

        print("Test Accuracy : ",correct/total)

        '''plotting the image and logging it to wandb'''
        _,axs=plt.subplots(10,3,figsize=(15,30))
        for i, ax in enumerate(axs.flat):
            ax.imshow(transforms.ToPILImage()(images[i]))
            ax.axis('off')
            ax.text(0.5,-0.1,f'True: {trueClass[i]}',transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')
            ax.text(-0.1,0.5,f'Predicted: {predictClass[i]}',transform=ax.transAxes,horizontalalignment='center',verticalalignment='center',rotation=90)

        wandb.log({"Part A plot": wandb.Image(plt)})

        plt.close()
        wandb.finish()

TestBestModel.testAccuracy()