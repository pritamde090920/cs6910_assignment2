import Model
import LoadDataset
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import torch.optim as optim

class TestBestModel:
    def testAccuracy():
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=128)
        trainLoader,valLoader,testLoader=dataLoader.data_loaders()

        model=Model.CNN(inputDepth=3,numOfFilters=256,sizeFilter=11,stride=1,padding=2,sizeDenseUnits=128,filterOrganization="half",activation="Mish", batchNormalization="Yes", dropoutProb=0.4)
        if torch.cuda.device_count()>1:
            model=nn.DataParallel(model)
        model.to(device)

        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(),lr=0.001)
        for _ in range(20):
            for data in trainLoader:
                inputs,labels=data[0].to(device),data[1].to(device)
                optimizer.zero_grad()
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
        
        for _ in range(20):
            for data in valLoader:
                inputs,labels=data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()

        total=0
        correct=0
        with torch.no_grad():
            for data in testLoader:
                inputs,labels=data[0].to(device), data[1].to(device)
                outputs=model(inputs)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()

        print("\nTest Accuracy :",correct/total)

        return model
    
    def plot(model):
        wandb.login()
        wandb.init(project="Pritam CS6910 - Assignment 2",name="Part A Test plot")

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K',batch_size=128)
        _,_,testLoader=dataLoader.data_loaders()

        if torch.cuda.device_count()>1:
            model=nn.DataParallel(model)
        model.to(device)

        class_names=testLoader.dataset.classes
        true_prediction_indices=[]
        true_prediction_data=[]
        predicted_labels=[]

        with torch.no_grad():
            for i,(data, label) in enumerate(testLoader):
                outputs=model(data)
                _, predicted=torch.max(outputs, 1)
                for j in range(len(label)):
                    if predicted[j]==label[j]:
                        true_prediction_indices.append(i*testLoader.batch_size + j)
                        true_prediction_data.append(data[j])
                        predicted_labels.append(predicted[j])

        selected_true_prediction_data=torch.stack(true_prediction_data)

        num_plots=min(len(true_prediction_indices),10*3)
        fig, axs=plt.subplots(10, 3, figsize=(15, 30))
        for i in range(num_plots):
            idx=true_prediction_indices[i]
            image=selected_true_prediction_data[i].numpy().transpose((1,2,0))
            true_class=testLoader.dataset[idx][1]
            predicted_class=predicted_labels[i].item()
            axs[i//3,i%3].imshow(image)
            axs[i//3,i%3].set_title(f"True: {class_names[true_class]}, Predicted: {class_names[predicted_class]}")
            axs[i//3,i%3].axis('off')

        wandb.log({"Part A plot": wandb.Image(plt)})

        plt.close()
        wandb.finish()

bestModel=TestBestModel.testAccuracy()
TestBestModel.plot(bestModel)