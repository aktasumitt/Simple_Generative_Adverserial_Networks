# Generative Adverserial Network (GANs)

## Introduction:
In this project, I aimed to train a (GANs) model using basic Generator-Discriminator architecture with Tensorboard to generate the some handwritten diigit images from random noise.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- I used the Mnist dataset for this project, which consists of 10 labels (handwritten digits) with total 60k images on train and 10k images on test.

## Model:
- A GANs model is a model that aims to generate realistic images similar to the given data by training on random noise. 
- The Generative Adversarial Network essentially consists of two main components: the Generator and the Discriminator. The Generator is a model that generates images from random noise, while the Discriminator is a structure that evaluates how realistic these generated images are. 
- These two components work in opposition to each other. The Discriminator continuously tries to train the Generator by identifying the generated images as fake. Meanwhile, the Generator aims to produce more realistic images using this feedback from the Discriminator's assessments.
- After a certain point, the Discriminator may no longer distinguish the generated images effectively, leading to a slight increase in loss, while the Generator's loss decreases as it produces images closer to reality.

## Train:

- The Discriminator takes both fake and real images separately and passes them through linear structures, ultimately using a sigmoid function to determine their authenticity. The average of these two losses is the discriminator loss.
-  On the other hand, the Generator takes random noise and passes it through linear structures to produce an output image of the same size as the images in the dataset. The structure of this output image compared to reality is the generator loss.
- I chose Adam optimizer with a learning rate of 0.0003 and used BinaryCrossEntropyLoss (BCELoss) for each model. I trained the model for 50 epochs.

## Results:
- After 50 epochs, Generator loss is aproximately 0.87 and discriminator loss is 0.64.There are generated images and graph of values on tensorboard.
<img src="https://drive.google.com/file/d/11VqoUo4y7FUX3PrEMVykpQvpHAh4GYkt/view?usp=drive_link" width="100" height="100">
<img src="https://drive.google.com/file/d/1iuRa7wpeOjmC6-jE0V4v-Lo26d627pmH/view?usp=drive_link" width="100" height="100">
<img src="https://drive.google.com/file/d/12S41N3Qmuvl_zD-yC38bbelzaVl77fz3/view?usp=drive_link" width="100" height="100">


## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Tensorboard files will created into "Tensorboard" folder during training time.
- Then you can generate the images from random noise by setting the "LOAD" and "TEST" values to "True" in the config file.

