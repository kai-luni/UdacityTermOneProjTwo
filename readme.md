
###Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


## Data Set Summary

I used the numpy library to calculate summary statistics of the traffic
signs data set::

* The training set consists of 34799 samples.
* The the validation set consists of 4410 samples.
* The test set is consists of 12630 samples.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

## Exploratory Visualization

Here is an exploratory visualization of the data set. It shows how the images are distributed among the different classes.

<img src="Pictures/Visualization.png" width="480"/>
<br><br>

## Preprocessing

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

## Model Architecture
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


## Model Training

To train the model, I used the Tensorflow implementation of the Adam Algorithm, which is called Adam Optimizer. This seems to do a good job and I did not touch it. As the count of the used samples is high with 4000 images per class, the learning rate was set to a very small number with 0.00008 and even lower to the second run with 0.00002.

Epochs were set to 100, which took about one hour for each run. Per iteration 192 samples were loaded, a higher number here would have made the whole process faster and increase the needed memory.

## Solution Approach

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 97.6% 
* test set accuracy of 96.1%
 
The architecture I used was the LeNet Convolutional Neural Network implementation provided from Udacity. I stayed with this architecture as it had promising results with an accuracy of 89% from the start. My first approach was to add a new Fully connected layer as I learned in class "The deeper the better". The increase in accuracy was small and I added a droput on the second convolutional layer and the accuracy improved more as expected.

Next I went to adding more contrast to the rgb training pictures and with those I was in the range of 92% accuracy and I stayed there for a long time. I started to flip certain images to multiply them, took even amounts of pictures by each class and tried different learning rates and sample sizes. I also started multiplying and slightly changing the training samples by rotation and transformation. It seemed to get over this 92% accuracy was difficult.

After that I changed the pictures to grayscale and applied Adaptive Histogram Equalization. The function from skimage.exposure has the 2 parameters kernel_size and clip_limit and I kept adjusting them until especially the brighness of the pictures was satisfying. My final parameters are 575 for kernel_size and 0.009 for clip_limit. With those much better quality samples I was much closer to the accuracy of 93%. 

The breakthrough came when I changed the convolutional layers. I added one more layer which took a bit time to calculate the output sizes of each layer and get a reasonable size for the first fully connected layer. After the new layer no Max Pooling was applied, because it always halfes the picture size and the final output would have been to small. A seemingly good output after the third convolutional layer could be finally archieved by the padding 'VALID'. This reconstruction increased the amount of neurons in the architecture and in contrast to the added fully connected layer in the beginning, this time there was a leap in Accuracy to as high as 95%. The extra added fully connected layer was removed again at this point.

In the last step I finetuned droput and applied it on all 3 convolutional layers, decreased the intensity of the transformation and added noise to some pictures. Further on I reloaded all the standard pictures, and just applied Adaptive Histogram Equalization without multliplying and so on. In this second round I also quartered the learning rate again. The idea is to finetune the neural network. Just the very lust step of finetuning increased the result another 0.5% to my final accuracy of 96.1%.

## Acquiring New Images
Here are five German traffic signs that I found on the web:

The priority road sign is very similar to another german traffic sign, even its not included in this project. Its called "end of priority road". If this would be included, the two of them might be easily confused from the convolutional neural network.
<br>
<img src="Pictures/TrafficSigns/12_priorityRoad.png" width="480"/>
<br>
<br>
The stop sign has a relatively complex pattern, which is the text on the sign. Especially on the low resolution this might be hard to grasp for a neural network.
<br>
<img src="Pictures/TrafficSigns/14_stop.png" width="480"/>
<br>
<br>
The no entry sign is round with good visible edges and its content is very different from any other traffic sign. It should be easily recognized.
<br>
<img src="Pictures/TrafficSigns/17_noEntry.png" width="480"/>
<br>
<br>
It seems to me that the angle this picture was taken from is not directly from the front and this might be a problem factor.
<br>
<img src="Pictures/TrafficSigns/32_endSpeedlimit.png" width="480"/>
<br>
<br>
There are again some traffic sign that look very similar to this one.
<br>
<img src="Pictures/TrafficSigns/33_turnRight.png" width="480"/>
<br>
<br>


## Performance on New Images

Here are the results of the prediction:

<table>
 <tr><td>Image</td><td>Prediction</td></tr>
 <tr><td>Priority sign</td><td>Priority sign</td></tr>
 <tr><td>Stop sign</td><td>Stop sign</td></tr>
 <tr><td>No Entry</td><td>No Entry</td></tr>
 <tr><td>End speed limit</td><td>End speed limit</td></tr>
 <tr><td>Turn right</td><td>Turn right</td></tr>
</table>


The model was able to predict all 5 pictures from the internet correctly, which corresponds with the accuracy on the test set.


##Model Certainty - Softmax Probabilities

The code to print out the predictions with probabilities can be found in code cell 24.

The prediction of the first picture is with 31% certainty Priority road, which is correct. I did not find that the picture in rank 2 to five have much in common, but their probabilities are also relatively low between 7% and 15%.

<table>
 <tr><td colspan="2" style="font-weight=bold;text-align: center;">Priority road</td></tr>
 <tr style="font-weight=bold;"><td>Probability</td><td>Prediction</td></tr>
 <tr><td>31</td><td>Priority road</td></tr>
 <tr><td>15</td><td>Roundabout mandatory</td></tr>
 <tr><td>13</td><td>Speed Limit 100</td></tr>
 <tr><td>8</td><td>Speed Limit 30</td></tr>
 <tr><td>7</td><td>Speed Limit 50</td></tr>
</table>


The stop sign was also predicted correctly with 31% certainty and the prediction two to five have a higher distance than in the first prediction.

<table>
 <tr><td colspan="2" style="font-weight=bold;text-align: center;">Stop</td></tr>
 <tr style="font-weight=bold;"><td>Probability</td><td>Prediction</td></tr>
 <tr><td>31</td><td>Stop</td></tr>
 <tr><td>8</td><td>Priority road</td></tr>
 <tr><td>7</td><td>Yield</td></tr>
 <tr><td>6</td><td>No vehicles</td></tr>
 <tr><td>5</td><td>Keep right</td></tr>
</table>


In picture number 3 its again 31% for the number one correction. The prediction on the second place is "No passing", which is also round and has some similarities. 

<table>
 <tr><td colspan="2" style="font-weight=bold;text-align: center;">No entry</td></tr>
 <tr style="font-weight=bold;"><td>Probability</td><td>Prediction</td></tr>
 <tr><td>31</td><td>No entry</td></tr>
 <tr><td>12</td><td>No passing</td></tr>
 <tr><td>10</td><td>Stop</td></tr>
 <tr><td>9</td><td>Turn left ahead</td></tr>
 <tr><td>8</td><td>Yield</td></tr>
</table>


Here you can see that the prediction possiblity is with 20% not so high. The picture on the second place does look similar and there are many more signs that have a similarity in their look.

<table>
 <tr><td colspan="2" style="font-weight=bold;text-align: center;">End speedlimit</td></tr>
 <tr style="font-weight=bold;"><td>Probability</td><td>Prediction</td></tr>
 <tr><td>20</td><td>End speedlimit</td></tr>
 <tr><td>13</td><td>End no passing</td></tr>
 <tr><td>6</td><td>End speedlimit 80</td></tr>
 <tr><td>5</td><td>Priority road</td></tr>
 <tr><td>5</td><td>Speed limit 30</td></tr>
</table>


In this picture the convolutional neural network is a bit more sure about the prediction than in the last one. Many of the other predictions are very similar again.

<table>
 <tr><td colspan="2" style="font-weight=bold;text-align: center;">Turn right ahead</td></tr>
 <tr style="font-weight=bold;"><td>Probability</td><td>Prediction</td></tr>
 <tr><td>29</td><td>Turn right ahead</td></tr>
 <tr><td>10</td><td>Ahead only</td></tr>
 <tr><td>10</td><td>Speed limit 30</td></tr>
 <tr><td>5</td><td>Right way next intersection</td></tr>
 <tr><td>3</td><td>Go straight or right</td></tr>
<table>




