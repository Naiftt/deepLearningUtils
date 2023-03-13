import torch
def distribute_data(numOfClients, train_dataset, batch_size):
    """
    numOfClients: int 
    train_dataset: train_dataset (torchvision.datasets class)
    return distributed dataloaders for each client
    """
    # distribution list to fill the number of samples in each entry for each client
    distribution = []
    # rounding the number to get the number of dataset each client will get
    p = round(1/numOfClients * len(train_dataset))
    
    # the remainder data that won't be able to split if it's not an even number
    remainder_data = len(train_dataset) - numOfClients * p 
    # if the remainder data is 0 ---> all clients will get the same number of dataset
    if remainder_data == 0: 
        distribution = [p for i in range(numOfClients)]
    else:
        distribution = [p for i in range(numOfClients-1)]
        distribution.append(p+remainder_data)

    # splitting the data to different dataloaders
    data_split = torch.utils.data.random_split(train_dataset, distribution)
    # CLIENTS DATALOADERS
    ClIENTS_DATALOADERS = [torch.utils.data.DataLoader(data_split[i], batch_size=batch_size,shuffle=True, num_workers=32) for i in range(numOfClients)]
    
    print(f"Length of the training dataset: {len(train_dataset)} sample")
    return ClIENTS_DATALOADERS
