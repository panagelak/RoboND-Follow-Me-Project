# Udacity Follow Me Project

The goal of the project is to train a fully convolutional deep neural network to identify and track a target in simulation. The target or "hero" is a specific person which we must distinguish from other people and the environment. The purpose is to segment people from camera images.

## Overview

Given a camera image from the drone, our goal is to assign each pixel a label (**semantic segmentation**) in this case (environment, person, hero), so the drone can follow the hero closely and accurately.  

**Network Architecture**  For this purpose we use a Fully Convolutional Neural Network similar to the image below, because it is important to preserve spatial information too, which traditional convolutional networks can't do
![FCN](http://i66.tinypic.com/331fwx5.png)

**The result of this FCN can be seen below:**

Image on the left is the input image
Image on the middle is the ground truth used for training
Image on the right is the output of our FCN network

![hero](http://i64.tinypic.com/ic2tl4.png)
![persons](http://i67.tinypic.com/20az2ty.png)

### Fully Convolutional Network Parts

![fcn_parts](http://i65.tinypic.com/2e6epar.png)

#### Encoder

The encoder network transforms an image into feature maps.
This part of the network learns and applies feature detectors into our input image.
It squezes the spacial dimensions of the image while increasing the depth (or number of filters) each used to extract usefull features (like edges and shapes)
But in the process we lose most of the spacial information contained in the image making it unsuitable for our purposes.
Such networks are usefull for classification tasks (a.k.a is there a dog in the picture) and unsuitable to answer (where in the image is the dog)

**It is important to note!** that because we downsample (or encode, compress) the image with the use of striding or in other cases max-pooling we lose the finer detail of our images making them blurry when we try to reconstruct them. Also due to the compression it is logical that we lose most of the spacial information, but we also keep only the information or features that distinguish our hero from the environment, this means tha a lot of the features of the environment like trees or roads are irrelevant and prone to disapear. After the compression and feature extraction our goal is to reconstruct the image to it's original size and assign each pixel a label for localization purposes. This is the job of the decoder, but the decoder has information only about what a person is and nothing about the environment, so the reconstructed image will have knoweledge only for the location and shape of the hero, or persons and limited knoweledge for the texture of the environment. Resulting in a reconstructed image of only 3 colors or labels.

**The code for the encoder block**

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

```

```python
def encoder_block(input_layer, filters, strides):
    
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    input_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=1,
                             padding='same', activation='relu')(input_layer)
    output_layer = separable_conv2d_batchnorm(input_layer=input_layer, filters=filters, strides=strides)
    return output_layer
```

#### 1x1 Convolution

After the encoder block we use a 1x1 convolutional layer instead of a fully connected layer (# kernels == # outputs).
A fully connected layer loses spacial information because we must flatten the 4D output shape of a convolutional layer to a 2D tensor (spatial dimensions flattened to a single vector), this result in the loss of spacial information.
Also an added benefit of a 1x1 convolutional layer is that you can feed the network any image size, something that you can't do with a fully connected layer because their output size is fixed.

**the code for the 1x1 convolutional layer is**
```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

#### Decoder

The purpose of the decoder block is to upsample the output from the 1x1 convolution back to the original input size. This can be achieved through the use of transposed convolutions or deconvolutions. A process which is the reverse of a convolution


**the code for the decoder is**
```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    small_ip_layer_upsampled = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([small_ip_layer_upsampled, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(input_layer = output_layer, filters = filters)
    return output_layer
```

#### Extra Techniques

##### Depthwise Seperable Convolutions

They are comprised of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer. The result of this is a significant reduction in the number of paramaters without losing much performance. It improves the runtime performance making them suitable for mobile applications, also the fewer paramaters make our model harder to overfit. An example of how this achieved can be seen below:

input shape : 32x32x3
desired number of output channels : 9
kernel shape: 3x3x3

In regural convolution the 3 input channels get tranversed by the 9 kernels => 9*3*3*3 = 254 paramaters

In Seperable convolutions the 3 input channels get traversed by 1 kernel each => 3x3x3 = 27 paramaters and 3 feature maps
These 3 feature maps get traversed by 9 1x1 convolutions each resulting in 27 + 9*3 = 54 paramaters total!!!

##### Batch Normalization

As we know normalizing our inputs is beneficial to our network, batch normalization extends these idea by normalizing the inputs to each layer of our network during training. For that purpose we use the mean and variance of the values from the current mini-batch.
The benefits of these are:

* Networks train faster : More calculation in the forward pass due to the normalization but we achieve convergence much more quickly resulting in faster training overall

* Allows higher learning rates : Gradient descent usually require low learning rates for the network to converge and furthermore as the network becomes deeper, these gradients become smaller resulting in even more iterations for convergence. Batch normalization allows us to use much higher learning rates which overcomes the above difficulties

* For the above reasons we can more easily build deeper networks, also it provides a bit of reguralization due to the noise it inserts. In some cases like inception modules this can be as powerfull of a technique as dropout.

##### Bilinear Upsampling

Like transposed convolution, Bilinear upsampling is used to upsample layers to higher dimensions or resolutions, the main difference is that while transpose convolutions is a learnable layer, Bilinear upsampling isn't. This can result in the loss of some finer details but helps speed up performance

##### Skip Connections

Skip connections are used as an easy way to retain fine spacial information from previous layers as we upsample the layers to the original size. This can be done by adding the two layers element wise or by concatenate them.
One advantage of concatenate the layers is that **the layers size need not much** unlike the element wise addition, allowing for more flexibillity.
It is important to note that after we concatenate the layers, we should add a regural or seperable convolution after this step for our model to better learn this finer details.

### Appropiate model size

As you can see our network is 7 layers deep, 3 encoder layers 1 1x1 convolution layer and 3 decoder layers. This is the model size i tried first and it worked great, these is indeed a rare occasion.
I also tried going deeper but this resulted in a reduction of the IoU score probably due to overfitting, if i tried to reduce the size further a.k.a 2 encoders and 2 decoders would also result in poor performance, because of underfitting meaning that our model would be incapable of capturing fine details. So these size seemed to be optimal with good results.

![fcnnimg](http://i66.tinypic.com/166hhq9.jpg)

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc_1 = encoder_block(inputs, filters=32, strides=2)
    print("enc_1 : ", enc_1.shape)
    enc_2 = encoder_block(enc_1, filters=64, strides=2)
    print("enc_2 : ", enc_2.shape)
    enc_3 = encoder_block(enc_2, filters=128, strides=2)
    print("enc_3 : ", enc_3.shape)


    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1x1 = conv2d_batchnorm(enc_3, filters=128, kernel_size=1, strides = 1)
    print("conv1x1 : ", conv1x1.shape)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec_1 = decoder_block(conv1x1, enc_2, filters=64)
    print("dec_1 : ", dec_1.shape)
    dec_2 = decoder_block(dec_1, enc_1, filters=32)
    print("dec_2 : ", dec_2.shape)
    dec_3 = decoder_block(dec_2, inputs, filters=num_classes)
    print("dec_3 : ", dec_3.shape)

    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    output_layer = layers.Conv2D(num_classes, kernel_size=1, activation='softmax', padding='same')(dec_3)
    print("Output : ", output_layer.shape)
    
    
    return output_layer
```

### Hyperparamaters

This part is usually the harder part of designing neural networks. My aproach consisted of choosing some "random values", and with some intuition i found some pretty good values relatively fast.

#### Batch size

As we know SGD estimates the derivative of the error function by randomly sampling a subset of the training data. This avoids the prohibitive cost of calculating the error function for the whole dataset. It's like we are taking many **drunk steps** instead of a very slow carefully planned step. In order for our model to train fast i chose a batch size of **16** , although we probably could use a bigger size if the gpu has enough memory this worked fine with a very short calculation time for each epoch.

#### Learning rate

At first i tried a learning rate of 0.1 which perform poorly probably because we overshoot local minimum's, but a learning rate of **a = 0.01** fixed this problem with lower loss and final score so i stick with that.

#### Number of epochs

I used **50 epochs** because these seemed to be the sweet spot of low training and validation loss. If we used fewer epochs we would underfit our model meaning high training and validation loss. On the other hand if we used a large number of epochs like 150 or 200 we would overfit meaning low training loss but high validation loss a clear sighn of overfitting.
We can see the progress of the training below.

**epoch 5**
![ep1](http://i67.tinypic.com/2u89c2w.png)
**epoch 10**
![ep2](http://i63.tinypic.com/5ob0a0.png)
**epoch 15**
![ep3](http://i68.tinypic.com/vsc9ir.png)
**epoch 20**
![ep4](http://i64.tinypic.com/aca078.png)
**epoch 25**
![ep5](http://i68.tinypic.com/2lka9n7.png)
**epoch 30**
![ep6](http://i64.tinypic.com/ei2fir.png)
**epoch 35**
![ep7](http://i63.tinypic.com/65xtux.png)
**epoch 40**
![ep8](http://i63.tinypic.com/292ac1t.png)
**epoch 45**
![ep9](http://i67.tinypic.com/33280tf.png)
**epoch 50**
![ep10](http://i64.tinypic.com/2rrksac.png)

Finally for **steps per epoch** i used 4131 (size of training set) divided by the batch size and for validation steps 200.

**The final IoU score was close to 0.49 which is pretty good**

### Can the model follow other target's?

The answer is unfortunately no because we trained our FCN net only on labeled images of pedestrian and of the hero, it wouldn't know for example what a dog is and would probably classify it as background environment. To achieve these goal we should provide to our model many examples of our desired class under many different conditions (distance, lighting, orientation etc), in order to be able to output labels for these classes too. The architecture of our network though is not needed to change.

### Future enhancements

Choosing hyperparamaters with only intuition or brute force can be proven very time consuming and laborious, thus we could use some sort of automated process to fine tune our paramaters like **Amazon SageMaker Hyperparameter Optimization**. Furthermore dropout might be proven beneficial, although we achieve something similar with batch normalization. At last if we could feed to our neural net voxelgrids or point clouds we should more easily distinguish bewteen the various objects and environment due to the different depth values they would usually have.