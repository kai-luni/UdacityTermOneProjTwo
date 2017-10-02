#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./processed/1_normal.png "Original"
[image2]: ./processed/2_gray.png "Grayscaling"
[image3]: ./processed/3_adapt.png "Adaptive Histogram Equalization"
[image4]: ./processed/4_flipped.png "Flipped"
[image5]: ./processed/5_rotated.png "Slightly Rotated"
[image6]: ./processed/6_transformed.png "Slightly Transformed"




###Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


## Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The training set consists of 34799 samples.
* The size of the validation set is ??????????????????
* The test set is consists of 12630 samples.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

## Visualisation of the dataset.

Here is an exploratory visualization of the data set. It shows how the images are distributed among the different classes.

<img src="Pictures/Visualization.png" width="480"/>
<br><br>

## Preprocessing

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I did not see an improvement in accuracy with rgb pictures.

Here is an example of a traffic sign image before and after grayscaling.

<img src="Pictures/Processed/1_normal.png" width="480"/>
<br>
<img src="Pictures/Processed/2_gray.png" width="480"/>
<br><br>


To optimize brightness and contrast I applied Adaptive histogram equalization.

<img src="Pictures/Processed/3_adapt.png" width="480"/>
<br><br>


The more samples you have available for training the higher is the accuracy, the amount of pictures can in some classes be increased by mirroring the picture.

<img src="Pictures/Processed/4_flipped.png" width="480"/>
<br><br>


As you can see in the visualisation, the distibution of the samples in the different classes is very different. To increase the accuracy it seems to me you need to have an equal amount of samples for every class. Even after the mirroring some classes have as little as 200 pictures. If we would now just take 200 samples from every class we would ignore a big part of our valuable samples. The logical consequence is to copy the samples in certain classes to multiply them. To add some variance to every copied picture, every copy is slightly rotated.

<img src="Pictures/Processed/5_rotated.png" width="480"/>
<br><br>


In the program 4000 samples each were used, this means in some classed a sample might have been copied up to 20 times. To further increase the variance I applied some small transformation on a certain percentage.

<img src="Pictures/Processed/6_transformed.png" width="480"/>
<br><br>


As a last step, I normalized the image data in the range from -1 to 1 as this makes sense mathematically for the statistical calculations of the weights.


##2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

## Architecture
My final model consisted of the following layers:

<table>
 <tr>
  <td>Input</td>
  <td>Shape: 32x32x1 Grayscale</td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Convolution 5x5</td>
  <td>Stride:1x1, padding:'VALID', output:28x28x32 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Max pooling</td>
  <td>strides: 2x2, output: 14x14x32</td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:90%</td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Convolution 5x5</td>
  <td>Stride:1x1, padding:'VALID', output:10x10x64 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:80%</td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Convolution 5x5</td>
  <td>Stride:1x1, padding:'VALID', output:6x6x128 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Max pooling</td>
  <td>strides:2x2, output:3x3x128</td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:70%</td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Fully Connected</td>
  <td>output:1152</td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Fully Connected</td>
  <td>output:350</td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td style="font-weight: bold;">Fully Connected</td>
  <td>output:43</td>
 </tr>
</table>


##3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Tensorflow implementation of the Adam Algorithm, which is called Adam Optimizer. This seems to do a good job and I did not touch it. As the count of the used samples is high with 4000 images per class, the learning rate was set to a very small number with 0.00008 and even lower to the second run with 0.00004.

Epochs were set to 100, which took about one hour for each run. Per iteration 192 samples were loaded, a higher number here would have made the whole process faster and increase the needed memory.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 97.7% 
* test set accuracy of 96.0%


>>>>>>>>>>>>>>>
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 >>>>>>>>>>>>>>>>>>>>>>
 
The architecture I used was the LeNet Convolutional Neural Network implementation provided from Udacity. I stayed with this architecture as it had promising results with an accuracy of 89% from the start. My first approach was to add a new Fully connected layer as I learned in class "The deeper the better". The increase in accuracy was small and I added a droput on the second convolutional layer and the accuracy improved more as expected.

Next I went to adding more contrast to the rgb training pictures and with those I was in the range of 92% accuracy and I stayed there for a long time. I started to flip certain images to multiply them, took even amounts of pictures by each class and tried different learning rates and sample sizes. I also started multiplying and slightly changing the training samples by rotation and transformation. It seemed to get over this 92% accuracy was difficult.

After that I changed the pictures to grayscale and applied Adaptive Histogram Equalization. The function from skimage.exposure has the 2 parameters kernel_size nad clip_limit and I kept changing them until especially the brighness of the pictures was satisfying. My final parameters are 575 for kernel_size and 0.009 for clip_limit. With those much better quality samples I was much closes to the accuracy of 93%. 

The breakthrough came when I changed the convolutional layers. I added one more layer which took a bit time to calculate the output sizes of each layer and get a reasonable size for the first fully connected layer. After the new layer no Max Pooling was applied, because it always halfes the picture size and the final output would have been to small. A seemingly good output after the third convolutional layer could be finally archieved by the padding 'VALID'. This reconstruction increased the amount of neurons in the architecture and in contrast to the added fully connected layer in the beginning, this time there was a leap in Accuracy to as high as 95%.

In the last step I finetuned droput and applied it on all 3 convolutional layers, decreased the intensity of the transformation and added some noise to some pictures. Further on I reloaded all the standard pictures, and just applied Adaptive Histogram Equalization without multliplying and so on. In this second round I also halfed the learning rate again. The idea is to finetune the neural network. Just the very lust step of finetuning increased the result another 0.6%.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

The priority road sign hast clearly defines edges and should be easy to be recognized. There is another german traffic sign that is very similar though, even its not included in this project. It call "end of priority road". If this would be included, the two of them might be easily confused from the convolutional neural network.
<img src="Pictures/TrafficSigns/12_priorityRoad.png" width="480"/>
<br>
The stop sign has a relatively complex pattern, which is the text on the sign. Especially on the low resolution this might be hard to grasp for a neural network.
<img src="Pictures/TrafficSigns/14_stop.png" width="480"/>
<br>
The no entry sign is round with good visible edges and its content is very different from any other traffic sign. It should be easily recognized.
<img src="Pictures/TrafficSigns/17_noEntry.png" width="480"/>
<br>
It seems to me that the angle this picture was taken from is not directly from the front and this might be a problem factor.
<img src="Pictures/TrafficSigns/32_endSpeedlimit.png" width="480"/>
<br>
There are again some traffic sign that look very similar to this one.
<img src="Pictures/TrafficSigns/33_turnRight.png" width="480"/>
<br>


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

<table>
 <tr><td>Image</td><td>Prediction</td></tr>
 <tr><td>Priority sign<td/><td>Priority sign<td/></tr>
 <tr><td>Stop sign<td/><td>Stop sign<td/></tr>
 <tr><td>No Entry<td/><td>No Entry<td/></tr>
 <tr><td>End speed limit<td/><td>End speed limit<td/></tr>
 <tr><td>Turn right<td/><td>Turn right<td/></tr>
<table>


The model was able to predict all 5 pictures from the internet correctly, which corresponds with the accuracy on the training data.


##Model Certainty - Softmax Probabilities
<table>
 <tr><td colspan="2">Stop sign</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
 <tr><td colspan="2">Stop sign</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
 <tr><td colspan="2">Stop sign</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
 <tr><td colspan="2">Stop sign</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
 <tr><td colspan="2">Stop sign</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
<table>


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



