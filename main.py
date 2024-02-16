import torch
import Checkpoint,visualization,model,train,dataset,Config,testing
from torch.utils.tensorboard import SummaryWriter


# Control cuda
devices=("cuda" if torch.cuda.is_available()==True else "cpu")


# Create Tensorboard
tensorboard=SummaryWriter(Config.TENSORBOARD_PATH,filename_suffix="Gans_mnist")


# Create transformer, dataset and dataloader
transformer=dataset.create_transformer()
train_dataset,test_dataset=dataset.create_dataset(transformer=transformer,dataset_path=Config.DATASET_PATH)
train_dataloader,test_dataloader=dataset.Create_Dataloader(BATCH_SIZE=Config.BATCH_SIZE,
                                                           train_dataset=train_dataset,
                                                           test_dataset=test_dataset)


# Visualization
if Config.VISUALIZE_DATA==True:
    visualization.visualize_data(Subplot_range=16,dataloader=test_dataloader)
 
 
    
# Generator Model
Generator_Model=model.Generator(img_size=Config.IMG_SIZE,
                                hidden_dim=Config.HIDDEN_DIM,
                                noise_dim=Config.NOISE_DIM).to(devices)


# Discriminator Model
Discriminator_Model=model.Discriminator(img_size=Config.IMG_SIZE,
                                        hidden_dim=Config.HIDDEN_DIM).to(devices)


# Create Optimizers and Loss Function
Generator_optimizer=torch.optim.Adam(Generator_Model.parameters(),lr=Config.LEARNING_RATE)
Discriminator_optimizer=torch.optim.Adam(Discriminator_Model.parameters(),lr=Config.LEARNING_RATE)
loss_fn=torch.nn.BCELoss()



# Load Checkpoint
if Config.LOAD_CHECKPOINT==True:
    checkpoint=torch.load(Config.CHECKPOINT_PATH)
    start_epoch=Checkpoint.load_checkpoints(checkpoint=checkpoint,
                                 optimizer_disc=Discriminator_optimizer,
                                 optimizer_gen=Generator_optimizer,
                                 model_generator=Generator_Model,
                                 model_discriminator=Discriminator_Model)
      
else:
    start_epoch=0
    print("Training Is Starting From Scratch...")


# Training
if Config.TRAIN==True:
     train.Training(start_epoch=start_epoch+1,
                    EPOCHS=Config.EPOCHS,
                    train_dataloader=train_dataloader,
                    z_dim=Config.NOISE_DIM,
                    devices=devices,
                    BATCH_SIZE=Config.BATCH_SIZE,
                    Generator_model=Generator_Model,
                    Discriminator_model=Discriminator_Model,
                    optimizer_discriminator=Discriminator_optimizer,
                    loss_fn=loss_fn,
                    CHECKPOINT_PATH=Config.CHECKPOINT_PATH,
                    optimizer_generator=Generator_optimizer,
                    save_checkpoints=Checkpoint.save_checkpoints,
                    tensorboard=tensorboard)

# Test Model
if Config.TEST_MODEL==True:
    testing.Test_Model(BATCH_SIZE=Config.BATCH_SIZE,
                       devices=devices,
                       Generator_model=Generator_Model,
                       noise_dim=Config.NOISE_DIM,
                       tensorboard=tensorboard)




    
