import matplotlib.pyplot as plt 
import torch
from torchvision.utils import make_grid

def visualize_data(Subplot_range,dataloader):
    for i in range(Subplot_range):
        plt.subplot(4,4,i+1)
        plt.imshow(torch.transpose(torch.transpose(dataloader.dataset[i][0],0,2),0,1))
        plt.xticks([])
        plt.yticks([])
    plt.show()
