# Project 4: Follow me

This project aims at training a model to process the camera image of the gimbal camera of a quadcopter to identify the presence and the locations of background, people and a target person. Once the target person (hero) is detected in the image the drone is supposed to determine its location and follow the hero around.
The task at hand boils down to performing semantic image segmentation for the three desired classes (background, people, hero).
For such a task the neural network architecture of a fully convolutional neural network has shown to produce successfull semantic segmentation models. Therefore such an FCN was designed, trained and tested.

---

## Problem domain and data

Since the problem is considered a supervised learning task the provided simulation engine written in unity was used to acquire labeled (segmented) training and validation data to train the network afterwards.

The following image shows a camera image of the quadcopter in the simulator and the corresponding segmentation map labeling background (red), people (green) and the hero (blue).

![sample data image](./imgs/data_sample.jpeg)

Using the simulator a dataset of 11700 training images and 2614 validation images were acquired for the different scenarios repeatedly in varying locations of the simulation environment:

* patrolling along a predefined path without explicitly seeing the hero
* patrolling over a crowded region that mainly intersects with the heroes path
* directly following the hero
* manually flying the quadcopter to capture the hero in the distance

One such data acquisition run can be seen at https://youtu.be/9hTfsF-sML8.

---

## Network architecture

A fully convolutional network is comprised of encoder and decoder blocks to first extract salient features for the segmentation task and then produce the desired segmented image.

### Encoder
The encoder block is basically a separable 2d convolution layer applying filter kernels creating feature maps followed by batch normalization to facilitate quicker training. Using the stride parameter the dimensionality is usually reduced with each encoder block, reducing also the number of trainable parameters and increasing the receptive field of the following layer. The encoder essentially looks at a patch of the image and extracts features that hold salient information for the segmentation task.

This image shows an encoder block operating on the 160x160x3 input image extracting 32 feature maps (applying 3x3 kernels) using a stride of 2.

<img src="./imgs/encoder.png" alt="encoder" style="max-width: 300px;"/>


### Decoder
The decoder block first applies bilinear upsampling to the input layer to increase the scale up the width and height dimensions fo the input features by factor 2. 

If provided, the upscaled feature maps are then concatenated with skip connections from previous layers.

Using these concatenated layers one or several regular convolutional layers are applied.

The image below depicts the architecture of such a decoder block.
The red feature maps are upsampled to the green feature maps. The blue features are optionally provided skip connection from previous network layers. The yellow block is the convolution result from the preceding concatenated layers and makes up the final ouput of the decoder block.

<img src="./imgs/decoder.png" alt="decoder" style="max-width: 500px;"/>


### 1x1 convolution layer (reasoning layer)

In between the encoder and decoder section of the fully convolutional network sits a convolutional layer with kernel size 1 which acts as a "reasoning" layer just like a fully connected layer.
The main difference to a fully connected layer is that the 1x1 convolution enables the preservation of the location information.

<img src="./imgs/1x1conv.png" alt="1_1conv" style="max-width: 500px;"/>

### Output 
The last decoder block is followed by a convolutional layer with one feature map for each class that should be represented in the segmentation task using a softmax activation function to basically perform pixelwise classification.

### Architecture

The final architecture is then set up in the following way:
* input: 160x160x3 image
* encoder_0: 10 filters, stride 1 
* encoder_1: 32 filters, stride 2 
* encoder_2: 100 filters, stride 2 
* encoder_3: 200 filters, stride 2 
* encoded: 1x1 convolution 512 filters
* decoder_1: 200 filters, skip connections from encoder_2
* decoder_2: 100 filters, skip connections from encoder_1
* decoder_3: 32 filters, skip connections from encoder_0
* output: 1x1 convolution 3 filters, with softmax activation

<img src="./imgs/robo-nd-fcn-model.png" alt="encoder"/>

---


## Training

For the training of the network the following parameters were used:

* epochs: 100 
* batchsize: 64 yields a good gradient estimate for each learning step and resulted in manageable memory allocation on AWS servers
* learning rate: base learning rate = 0.001

    the learning rate was boosted for the first 3 epochs to quickly reach good model states

    beyond epoch #10 the learning rate was exponentially decayed to overcome plateauing and allow for the final improvements of the model

        def lr_schedule(epoch):
            if epoch < 3:
                lr = learning_rate * 10    # boost period 
            else:
                lr = learning_rate * np.power(0.975, epoch-10)
                lr = np.clip(lr, 1e-4, 1e-3)
            print('learning rate:{}'.format(lr))
            return lr

* steps per epoch: 200 such that within each epoch the entire training data set of 11700 samples is used
* validation steps: 50 to ensure that the model is for sure evaluated on all 2614 validation samples

The training on the amazon web service with GPU capabilities took approximately 10 hours and yielded a reasonable model. The validation loss seems to cap out after 30 epochs already but since it didn't show signs of overfitting the training was continued up to a point where it hit a plateau.

![model training](./imgs/model_training.png)


---

## Results

The resulting model is largely able to perform the segmentation task as desired. The model parameters and config can be found in `data/weights/model_weights.h5` and `data/weights/config_model_weights.h5`

The following images show the input image, the correct segmentation and the model estimate for some of the provided validation samples.

![](./imgs/segmentation_follow.png)
![](./imgs/segmentation_patrol_no_target.png)
![](./imgs/segmentation_patrol.png)

### Scores
For the provided sample validation set the model achieves a satisfying final Intersection over Union (IoU) score of 0.5079. However the network seems to still have trouble detecting the hero in the distance. By looking at the missclassified false negative result regarding the detection of the hero it seems to be mainly a problem of low illumination that causes the model to mistake the hero for a regular person.

If the task would change to detecting and following other targets (such as animals or cars) it would be necessary to acquire new training data and retrain the model to perform this segmentation. Possibly one might even have to adapt the network architecture to allow for the detection of smaller targets which appear to be troublesome for the current setup already.

The final model can be seen in action here https://youtu.be/wWINhsZmoBs, used for the segmentation of the input image of the quadcopter that is supposed to search and follow the hero. We can observe that the quadcopter succesfully finds and follows the target.

