# Classification of Covid19 X-rays and Detection of Regions of Interest

## Introduction:

Coronavirus disease (Covid-19) is an irresistible infection caused by a newfound virus. This pandemic has outspread all across the earth. Testing for Coronavirus has been not able to stay aware of the interest as the quantity of cases are expanding day by day. The most generally utilized Coronavirus detection procedure is real-time polymerase chain reaction (RT-PCR). Because of less affectability of RT-PCR, it gives high false negative outcomes. So identifying conceivable Coronavirus diseases by radiological imaging methods, for example, chest X-rays and computed tomography (CT) can give better outcomes in the event of crucial and critical cases and may help isolate high risk patients while test results are anticipated. 

Coronavirus uncovers some radiological marks that can be effortlessly recognized through chest X-rays. For this, radiologists are needed to examine these marks. Although, it is a  time-consuming and error-prone task. Therefore, we need to automate the diagnosis of chest X-rays.

So we are trying to create an automated deep learning model to classify chest X-rays which may quicken the analysis time and would detect covid19 X-rays more accurately. These methodologies can train the weights of networks on large dataset along with  the weights of pre-trained networks on smaller datasets.


## Objective:

The goal of the project is to classify Covid19 X-rays and detection of right regions of interest by providing the feature maps using an automated deep learning model. 

## About the Dataset:

The dataset contains Covid and Non-Covid cases of both X-ray and CT images. But I will be using only X-ray images for evaluation of my deep learning model. With in the dataset there are two separate sub-folders for X-ray images:

5500 Non-COVID images 
4044 COVID images.
Source: data.mendeley.com/datasets/8h65ywd2jr/3#__sid=js0 

## Data Preparation: 

The dataset I am dealing with is an image dataset. So a important part of preprocessing is resizing the images. Essentially all machine learning models train quicker on smaller images. Also most of the deep learning model architectures necessitate that all images to be in similar size. Resizing images to a standard size that is smaller in pixels gives better results and save time while training the model. whereas if we resize all images from smaller to larger pixels, all small image pixels are stretched and this can uncertain our model ability to learn key features like object boundaries, etc. So I have resized all images in the entire dataset to 150*150.  

## Modeling:

Binary Classification:

We know binary classification is used to classify a set of elements into two categories. It is performed based on a classification rule. Mainly binary classification is applied in real time situations. Most commonly and practically applied in medical testing to determine if a patient is suffering from a disease or not. In binary classification rather focusing on accuracy, the relative extent of various kinds of errors is of interest like as we know in medical imaging true positives and false positives are one of the important characteristics and also in medical tests by determining a disease when it is not present(false positive) is different from not determining a disease when it is present(false negative). So we would be checking to see if the false positives prediction is lower as we would not like a lot of false positives in our model.

The correct tool for a image classification position is a convnet. A model that can store a ton of data can possibly be more precise by utilizing more features. For our situation we will utilize a convnet with few layers and few channels for every layer. A basic heap of 3 convolution layers with a ReLU enactment and followed by max-pooling layers. On top of it we stick two fully-connected layers. We end the model with a solitary unit and a sigmoid enactment, which is ideal for a binary classification. To go with it we will likewise utilize the binary cross entropy loss to prepare our model.

In this approach after fitting the model it gives us a accuracy of 0.45-0.91 after 10 epochs, validation accuracy of 95.0 and test accuracy of 80%.


## What is Feature map? 

Feature maps are also called as activation maps as they capture the result of applying the filters to input, such as the input image or another feature map. The idea of visualizing a feature map for a specific input image would be to understand what features of the input are detected or preserved in the feature maps. The expectation would be that the feature maps close to the input detect small detail, whereas feature maps close to the output of the model capture more general features. The feature map is the output of one channel applied to the previous layer. A given channel is drawn across the whole previous layer, moved each pixel in turn. Each position brings about an activation of the neuron and the output is gathered in the feature map. You can see that assuming the receptive field is moved one pixel from one activation to another, the field will cover with the previous activation by (field width - 1) input values. 
So I will be trying to predict the feature map for my model by which we can see the main regions and detect the affected covid areas in the covid chest X-ray images. In the  feature map, significant findings in the Covid-19 X-ray include expanded whiteness of the lungs, corresponding to the seriousness of the infection. As the condition becomes more severe, these markings become invisible, or ‘whited-out’. 

## VGG 19 Model:

It is very deep convolutional network for large-scale image recognition. VGG19 is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer). So in easy terms, VGG is a deep CNN used to classify images. When we give a image as an input to this network, it shapes the matrix to 224,224,3. When we use kernels with 3* 3 size with a stride size of pixel 1, this will help to cover the whole idea of the image. Spatial padding was utilized to safeguard the spatial resolution of the image. Max pooling was performed over a 2 * 2 pixel windows with stride 2. Then this is followed by an activation layer, Rectified linear unit(ReLu). This helps the model classify better as it initiate non-linearity. Also it improves computational time. Then three fully connected layers are implemented in which two are of size 4096 and after that with a layer 1000 channels. And the final layer is a softmax function. 

After training the model,  I evaluated train loss and accuracy as well as test loss and accuracy for VGG19 model: Train accuracy- 84.4% and Test accuracy- 88.4%.
I can say from this VGG19 model worked better than the scratch model. Now we are predicting the feature map for VGG19 model by which we can see the main regions and detect the affected covid areas in the covid chest X-ray images.As We know the result will be a feature map and we can plot all 36 two-dimensional images as an 6x6 square of images. Depending on the image location that is covid-19 X-ray, the same value is given as a parameter to plot the feature map. 

In the above feature map, significant findings in the Covid-19 X-ray include expanded whiteness of the lungs, corresponding to the seriousness of the infection. As the condition becomes more severe, these markings become invisible, or ‘whited-out’.

## Alexnet Model:

AlexNet can likewise be attributed with bringing deep learning to adjoining fields such as medical image analysis and natural language processing. AlexNet is one of the convolutional neural network architecture that does well on image classification. The architecture consists of eight layers: the first five were convolutional layers, some of them followed by max-pooling layers and the last three were fully connected layers. It uses the non-saturating ReLU activation function, which showed improved training performance. But this isn’t what makes AlexNet special, there are some of the features used that are new approaches to convolutional neural networks like ReLU Nonlinearity, Multiple GPUs, Overlapping Pooling. VGG, while compared to AlexNet, has a few differences that separates it from other competing models: instead of utilizing large receptive fields like AlexNet (11x11 with a step of 4), VGG utilizes exceptionally small receptive fields (3x3 with a step of 1).

After training the model,  I evaluated train loss and accuracy as well as test loss and accuracy for Alexnet model: Train accuracy- 96.8% and test accuracy- 89.9%.
I can say from this Alexnet model worked much better than remaining all the models. Now we are predicting the feature map for Alexnet model by which we can see the main regions and detect the affected covid areas in the covid chest X-ray images. As We know the result will be a feature map and we can plot all 36 two-dimensional images as an 6x6 square of images. Depending on the image location that is covid-19 X-ray, the same value is given as a parameter to plot the feature map.

From the Alexnet feature map, significant and clear findings of covid can be found in the Covid-19 X-ray which includes expanded whiteness of the lungs, corresponding to the seriousness of the infection. As the condition becomes more severe, these markings become invisible, or ‘whited-out’. Also Alexnet is better in detecting the virus compared to other model as in the above feature map, we can see it is clearly spotting the white regions on the X-ray by only focusing on the important and required features that it learnt from the model. 




