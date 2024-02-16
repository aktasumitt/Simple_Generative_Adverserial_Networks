import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def Test_Model(BATCH_SIZE,devices,Generator_model,noise_dim,tensorboard):
    
    test_images=[]
    for i in range(10):
        noise_test=torch.randn(BATCH_SIZE,noise_dim).to(devices)

        generated_test=Generator_model(noise_test)   
        generated_image_test=generated_test.cpu().detach()
        test_images.append(generated_image_test)
        
        tensorboard.add_image("Test Images",make_grid(generated_image_test,nrow=10),global_step=i+1)

    for i in range(len(test_images)):
        plt.subplot(5,2,i+1)
        plt.imshow(torch.permute(test_images[i][i],(1,2,0)),cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    