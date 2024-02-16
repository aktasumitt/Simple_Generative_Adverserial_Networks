import torch

def save_checkpoints(optimizer_generator,optimizer_discriminator,model_generator,model_dicriminator,epoch,checkpoint_path):
    print("Checkpoints are saving...\n")
    
    Checkpoints={"epoch":epoch,
                 "model_discriminator_state":model_dicriminator.state_dict(),
                 "model_generator_state":model_generator.state_dict(),
                 "optimizer_generator_state":optimizer_generator.state_dict(),
                 "optimizer_discriminator_state":optimizer_discriminator.state_dict()}
    
    torch.save(obj=Checkpoints,f=checkpoint_path)
    

def load_checkpoints(checkpoint,model_discriminator,model_generator,optimizer_gen,optimizer_disc):
    print("Checkpoints Are Loading...")
    
    starting_epoch=checkpoint["epoch"]
    model_discriminator.load_state_dict(checkpoint["model_discriminator_state"])
    model_generator.load_state_dict(checkpoint["model_generator_state"])
    optimizer_gen.load_state_dict(checkpoint["optimizer_generator_state"])
    optimizer_disc.load_state_dict(checkpoint["optimizer_discriminator_state"])
    return starting_epoch





