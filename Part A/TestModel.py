import Model
import LoadDataset
import torch
import wandb
import torchvision
import matplotlib.pyplot as plt
import lightning as pl
from LightliningModel import FastRunning
from torchvision import transforms


class TestBestModel:
    def testAccuracy():
        wandb.login()
        wandb.init(project="Pritam CS6910 - Assignment 2",name="Part A Test plot")

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=8)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

        model=Model.CNN(inputDepth=3,numOfFilters=64,sizeFilter=7,stride=1,padding=2,sizeDenseUnits=1024,filterOrganization="same",activation="Mish", batchNormalization="Yes", dropoutProb=0)
        lightningModel=FastRunning(model)
        trainer=pl.Trainer(max_epochs=10,devices=-1)

        trainer.fit(lightningModel,trainLoader,valLoader)

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
        test_dataset=torchvision.datasets.ImageFolder(root='./inaturalist_12K',transform=transform)
        testLoader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)

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