# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/class_counts.png "Visualization"
[image1-1]: ./writeup_images/random_signs.png "Visualization"
[image2]: ./writeup_images/sign_comparison.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./found_signs/70km.jpg "Traffic Sign 1"
[image5]: ./found_signs/children.jpg "Traffic Sign 2"
[image6]: ./found_signs/dcurve.jpg "Traffic Sign 3"
[image7]: ./found_signs/priority.jpg "Traffic Sign 4"
[image8]: ./found_signs/stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code.](https://github.com/kmlee17/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set (after train-test-split of original training data set into new training and validation data set):

* The size of training set is **27839**
* The size of the validation set is **6960**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.
Here is a snapshot of 20 random signs from the training data set.  This gives a good overview of the type and quality of the images in the set.

![alt text][image1-1]

Here is a histogram of number of images per class of traffic sign.  Notice that some of the classes are underrepresented.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun had more success with using greyscale over color in their paper "Trafic Sign Recognition with Multi-Scale Convolutional Networks".  Greyscale also decreases the array size of the images so this should help with model training time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so that the model can avoid dealing with scale differences which will help the training process with more stable gradients.

I considered generating additional data through data augmentation because it appeared as through there was a class imbalance, but I figured I should try without augmentation first.  The accuracy without augmentation was good enough, though it would be interesting to try out to see if it improves the model.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flattening	      	| outputs 400				|
| Fully connected		| outputs 120      									|
| RELU					|												|
| Dropout		|      									|
| Fully connected		| outputs 84     									|
| RELU					|												|
| Dropout		|      									|
| Fully connected		| outputs 43    									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer as this generally performs better than standard gradient descent.  I used a batch size of 128, 60 epochs, and a learning rate of 0.001.  The accuracy of the model seemed to converge earlier, around epoch 40, but I wanted to let it run longer to see if incremental improvements could be made.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.992**
* validation set accuracy of **0.990**
* test set accuracy of **0.942**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

**I first used the standard LeNet architecture to first validate that I was using Tensorflow correctly, as well as to establish a baseline for accuracy.**

* What were some problems with the initial architecture?

**The initial architecture seemed to be overfitting because of the descrepancy between training and test set accuracy.**

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

**Because the training set was performing quite a bit better than the test set, I adjusted the LeNet architecture by adding in dropout, which is a form of regularization where weights have a probability of being dropped.  I believed this could help with overfitting.  I added the two dropouts after the fully-connected layers.**

* Which parameters were tuned? How were they adjusted and why?

**I didn't adjust the hyperparameters or play with changing the dimensions of the convolutional layers too much.  If I had more time, I would have tried these to bump up the test set accuracy higher, but as is, the model performed fairly well and satisfied the criteria for the project.**

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

**Adding dropout layers to the network architecture seemed to have the biggest impact as this helped greatly with overfitting.**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

All images have good clarity, but it might be difficult to differentiate between the speeds or the curve patterns.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h      		| 70 km/h   									| 
| Children     			| Children 										|
| Double Curve				| Dangerous Right Curve											|
| Priority Road	      		| Priority Road					 				|
| Stop			| Stop    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares less favorably to the accuracy on the test set because of the very small sample size.  It is interesting that the SoftMax predictions were extremely high (sometimes 100%) even for the misclassified sign.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the **28th cell of the jupyter notebook.**

For the first image, the model is sure that this is a speed limit 70 km/h (probability of 1.0), and the image does contain a speed limit 70 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Speed limit (70km/h)   									| 
| .0000     				| Speed limit (20km/h) 										|
| .0000					| Speed limit (30km/h)											|
| .0000	      			| Speed limit (80km/h)					 				|
| .0000				    | Speed limit (120km/h)      							|


For the second image, the model is sure that this is a children crossing (probability of 0.9968), and the image does contain a children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9968         			| Children crossing   									| 
| .0030     				| Dangerous curve to the right 										|
| .0002					| End of no passing											|
| .0000	      			| Road narrows on the right					 				|
| .0000				    | Dangerous curve to the left      							|

For the third image, the model is pretty sure that this is a dangerous curve to the right (probability of 0.8356), and the image does not contain a dangerous curve to the right sign (Double curve). It is worrysome that the actual image class doesn't even show up on the top 5.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.8356         			| Dangerous curve to the right   									| 
| .1644     				| Children crossing 										|
| .0000					| End of no passing											|
| .0000	      			| End of all speed and passing limits					 				|
| .0000				    | Slippery road     							|

For the fourth image, the model is sure that this is a priority road (probability of 1.0), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Priority road   									| 
| .0000     				| No passing 										|
| .0000					| Yield											|
| .0000	      			| No vehicles					 				|
| .0000				    | Roundabout mandatory     							|

For the fifth image, the model is sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Stop  									| 
| .0000     				| Yield										|
| .0000					| Keep right										|
| .0000	      			| Turn left ahead					 				|
| .0000				    | Road work    							|