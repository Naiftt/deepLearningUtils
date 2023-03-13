import torch
def train_test_model(model, optimizer, criterion, 
    num_epochs, trainloader, testloader, train = True, test = True):
    """
    Train a pytorch model and testing it 
    
    Args: 
        model (Pytorch model): an instance of a pytorch model 
        optimizer: torch optimizer 
        criterion: torch loss function
        num_epochs: the num of epochs to train for
        trainloader: pytorch data loader for training
        testloader: pytorch data loader for testing
        test: boolean to test or not
        train: boolean to train or not

    Returns: 
    trained model
    
    """


    if torch.cuda.is_available():
         device = 'cuda'

    logs = {'training_loss': [], 'testing_loss':[], 'training_acc': [], 'testing_acc':[]}
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if train:
            running_loss = 0.0
            correct = 0
            total = 0
            print(f"{epoch+1} / {num_epochs}")
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct // total
            train_loss = running_loss / len(trainloader)
            print(f"Training Acc: {train_acc} %")
            print(f"Training Loss: {train_loss}")
            logs['training_acc'].append(train_acc)
            logs['training_loss'].append(train_loss)
        running_loss = 0
        correct = 0 
        total = 0
        if test:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    running_loss += loss.item()
            
            test_acc = 100 * correct // total
            test_loss = running_loss / len(testloader)
            print(f'Testing Acc: {test_acc} %')
            print(f"Testing Loss: {test_loss}\n")
            logs['testing_acc'].append(test_acc)
            logs['testing_loss'].append(test_loss)
        if test == True and train == False: 
            break
    print('Finished Training')
    return model, logs