from torchvision import transforms,datasets
from torch.utils.data import DataLoader



def create_transformer():
    transformer=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
    return transformer





def create_dataset(transformer,dataset_path):
    
    train_dataset=datasets.MNIST(root=dataset_path,
                                        train=True,
                                        download=True,
                                        transform=transformer)


    test_dataset=datasets.MNIST(root=dataset_path,
                    train=False,
                    download=True,
                    transform=transformer)

    return train_dataset,test_dataset




def Create_Dataloader(BATCH_SIZE,train_dataset,test_dataset):
    
    train_dataloader=DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True)

    test_dataloader=DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                drop_last=True)
    
    return train_dataloader,test_dataloader