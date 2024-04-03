import torch
import torch.optim as optim
import torch.nn as nn
import wandb

class TrainModel:
    def accuracyAndLoss(model,device,trainLoader,valLoader,dataLoader,epochs):
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(),lr=0.001)
        trainAccuracyPerEpoch=list()
        valAccuracyPerEpoch=list()
        trainLossPerEpoch=list()
        valLossPerEpoch=list()

        for epoch in range(epochs):
            print("\nEpoch :",epoch+1)
            train_loss=0.0
            correct=0
            total=0
            for data in trainLoader:
                inputs,labels=data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                # print statistics
                train_loss+=loss.item()
                # the class with the highest energy is what we choose as prediction
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
            
            trainLossPerEpoch.append(train_loss/len(dataLoader.train_dataset))
            trainAccuracyPerEpoch.append(correct/total)
            print("Training loss :",trainLossPerEpoch[-1])
            print("Training Accuracy :",trainAccuracyPerEpoch[-1])

            val_loss=0.0
            correct=0
            total=0
            for data in valLoader:
                inputs,labels=data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                # print statistics
                val_loss+=loss.item()
                # the class with the highest energy is what we choose as prediction
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
            
            valLossPerEpoch.append(val_loss/len(dataLoader.val_dataset))
            valAccuracyPerEpoch.append(correct/total)
            print("Validation loss :",valLossPerEpoch[-1])
            print("Validation Accuracy :",valAccuracyPerEpoch[-1])

            wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(epoch+1)})