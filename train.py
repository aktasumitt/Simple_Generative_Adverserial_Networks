import torch
from tqdm import tqdm
from torchvision.utils import make_grid


def Training(start_epoch,EPOCHS,z_dim,train_dataloader,devices,BATCH_SIZE,
             Generator_model,Discriminator_model,optimizer_discriminator,
             loss_fn,optimizer_generator,CHECKPOINT_PATH,save_checkpoints,tensorboard):
    
    print(f"{start_epoch}.Epoch Is Starting...")

    for epoch in range(start_epoch,EPOCHS):
        loss_disc=0.0
        loss_gen=0.0
        progress_bar=tqdm(range(len(train_dataloader)),desc="Training Progress:",position=0,leave=True)
        
        for batch,(data,_labels_) in enumerate(train_dataloader,0): # we use real image labels with 1 so we dont need real data labels
            
            real_image=data.to(devices)
            random_noise=torch.randn(BATCH_SIZE,z_dim).to(devices)
            fake_image=Generator_model(random_noise)

 
            # discriminator
            Discriminator_model.zero_grad()
            real_discriminator=Discriminator_model(real_image)
            fake_discriminator=Discriminator_model(fake_image)
            
            loss_discriminator_fake=loss_fn(fake_discriminator,torch.zeros_like(fake_discriminator))
            loss_discriminator_real=loss_fn(real_discriminator,torch.ones_like(real_discriminator))
            loss_discriminator=(loss_discriminator_fake+loss_discriminator_real)/2
            
            loss_discriminator.backward(retain_graph=True)
            optimizer_discriminator.step()
            
            loss_disc+=loss_discriminator.item()
            
            
            # generator
            Generator_model.zero_grad()
            
            out_genrated_disc=Discriminator_model(fake_image)
            loss_generator=loss_fn(out_genrated_disc,torch.ones_like(out_genrated_disc))
            loss_generator.backward()
            optimizer_generator.step()

            loss_gen+=loss_generator.item()
        
            progress_bar.set_postfix({"Epoch":(epoch) ,  "Loss Discriminator" : (loss_disc/(batch+1)),  "Loss Generator" : (loss_gen/(batch+1))})
            progress_bar.update(1)
            
        progress_bar.close()      
        
   
        
        # Giving real images and generated images to tensorboard each epochs
        grid_real=make_grid(real_image,nrow=10)
        grid_fake=make_grid(fake_image,nrow=10)    
        tensorboard.add_image("Real Images",grid_real,global_step=epoch+1)
        tensorboard.add_image("Generated Image",grid_fake,global_step=epoch+1)
        
        # Giving losses to tensorboard each epochs
        tensorboard.add_scalar("Discriminator loss", loss_disc/(batch+1),global_step=epoch+1)
        tensorboard.add_scalar("Generator loss", loss_gen/(batch+1),global_step=epoch+1)

        save_checkpoints(optimizer_discriminator=optimizer_discriminator,
                        optimizer_generator=optimizer_generator,
                        model_dicriminator=Discriminator_model,
                        model_generator=Generator_model,
                        epoch=epoch,
                        checkpoint_path=CHECKPOINT_PATH
                        )
