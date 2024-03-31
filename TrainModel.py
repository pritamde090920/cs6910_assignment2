import Model
import LoadDataset
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms

dataLoader=LoadDataset.DatasetLoader(root='./inaturalist_12K')
trainLoader,valLoader,testLoader=dataLoader.data_loaders()

model=Model.CNN(inputDepth=3,numOfFilters=32,sizeFilter=5,stride=1,padding=2,sizeDenseUnits=32,filterOrganization="same",activation="ReLU", batchNormalization="Yes", dropoutProb=0.2)
dataAugmentation="Yes"

if dataAugmentation=="Yes":
    transform=transforms.Compose([
        transforms.Resize((64,64)),#224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    trainLoader.dataset.transform=transform
    valLoader.dataset.transform=transform
    testLoader.dataset.transform=transform

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    print("\n\nEpoch :",epoch)
    running_loss=0.0
    for i,data in enumerate(trainLoader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs,labels=data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss+=loss.item()

        print("Done Image",i+1)
    
    print("Training loss :",running_loss/len(dataLoader.train_dataset))

    correct=0
    total=0
    with torch.no_grad():
        for data in trainLoader:
            images,labels=data
            # calculate outputs by running images through the network
            outputs=model(images)
            # the class with the highest energy is what we choose as prediction
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print("Training Accuracy :",correct/total)

    correct=0
    total=0
    with torch.no_grad():
        for data in valLoader:
            images,labels=data
            # calculate outputs by running images through the network
            outputs=model(images)
            # the class with the highest energy is what we choose as prediction
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print("Validation Accuracy :",correct/total)