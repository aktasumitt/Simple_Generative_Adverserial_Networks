import torch
import torch.nn as nn

class Discriminator(nn.Module):
    
    def __init__(self,img_size,hidden_dim) -> None:
        super(Discriminator,self).__init__()

        self.img_size=img_size
        
        self.linear_disc=nn.Sequential(nn.Linear(img_size,hidden_dim*4),
                                       nn.ReLU(0.1),
                                       nn.Dropout(0.2),
                                       
                                       nn.Linear(hidden_dim*4,hidden_dim*2),
                                       nn.ReLU(0.1),
                                       nn.Dropout(0.2),
                                       
                                       nn.Linear(hidden_dim*2,hidden_dim),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout(0.2),
                                       
                                       nn.Linear(hidden_dim,1),
                                       nn.Sigmoid())
        
    
    def forward(self,data):
        
        x=data.view(-1,self.img_size)
        out=self.linear_disc(x)
        
        return out



class Generator(nn.Module):
    def __init__(self,img_size,hidden_dim,noise_dim):
        super(Generator,self).__init__()
        
        
        self.linear_gen=nn.Sequential(nn.Linear(noise_dim,hidden_dim),
                                      nn.ReLU(),
                                      
                                      nn.Linear(hidden_dim,hidden_dim*2),
                                      nn.ReLU(),
                                      
                                      nn.Linear(hidden_dim*2,hidden_dim*4),
                                      nn.ReLU(),
                                      
                                      nn.Linear(hidden_dim*4,img_size),
                                      nn.Tanh())
        
                                      
        
    def forward(self,noise):
        
        x=self.linear_gen(noise)
        x=x.view(-1,1,28,28)
        
        return x        

