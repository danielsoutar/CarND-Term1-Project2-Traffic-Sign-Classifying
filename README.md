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

[image1]: ./class_distribution.png "Class Distribution"
[image2]: ./nonprocessed_data.png "Data prior to pre-processing"
[image3]: ./preprocessed_data.png "Grayscaled and normalised data using global histogram equalisation"
[image4]: ./20_sign.jpeg "Traffic Sign 1"
[image5]: ./80_sign.jpeg "Traffic Sign 2"
[image6]: ./no_entry.jpeg "Traffic Sign 3"
[image7]: ./no_passing_sign.jpeg "Traffic Sign 4"
[image8]: ./priority_road.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/danielsoutar/CarND-Term1-Project2-Traffic-Sign-Classifying/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple numpy to calculate summary statistics of the traffic
signs data set, along with some assert statements to confirm the x and y objects matched up in the number of examples.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the varying classes.
As you can see there is substantial variation in distribution - any model trained on this dataset would need to take care that it was not biased towards the better-represented classes. That being said, it may well be that the actual number of each type of road sign corresponded to our dataset.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because this not only reduced the amount of computation (only convolving one layer rather than three in the first convolution layer), but also because the colours of the signs don't necessarily assist in our classification. Case in point, even if the colours of some signs like the 20 kmph sign were changed completely, we'd still recognise the type of sign. Moreover, signs may be faded, there might be glare, or it may be dark. By converting images to grayscale and normalising them, we simplify and standardise the inputs to our model, which intuitively should improve performance and speed up training.

Here is an example of a traffic sign image before and after grayscaling, along with histogram equalisation.

![alt text][image2]

I normalized the image data because this reduces the amount of learning that the network has to do - by having zero mean the network can train faster. 

Regarding histogram equalisation, it was clear there were images with different levels of contrast. The model shouldn't have to learn separate parameters just to recognise the same sign in different lighting conditions, so I decided to apply global histogram equalisation, where the intensities are brought into line across the entire image. This does increase background noise in some cases, but overall it seems to bring images into similar levels of focus, which should help the network in classifying.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding => 28x28x6 			|
| Max pooling	      	| 2x2 stride => 14x14x6		 					|
| Convolution 5x5     	| 1x1 stride, valid padding => 10x10x64		 	|
| Max pooling	      	| 2x2 stride => 5x5x64		 					|
| Flattened (fc) 		| Flatten the previous layer (5x5x64) => 1600	|
| Fully connected (fc1)	| ReLU(matrix multiply fc & weights w) => 126 	|
| Fully connected (fc2)	| ReLU(matrix multiply fc1 & weights w1) => 84	|
| Logits				| matrix multiply of fc2 & weights w => 43		|
| Softmax				| softmax on logits along with cross-entropy 	|
 

Simpler than I would have liked (the validation accuracy refused to climb any further than 94.6% no matter what I did).

I did play around with dropout, but all it seemed to do in my case was start off at a lower accuracy (as low as 5%!!) and take longer to train. It did get up to roughly 92%, but then it refused to budge. This sounds like underfitting, but even when I added multiple layers and increased the number of parameters it didn't seem to want to go any further. This could potentially be a lack of sufficient augmentation or pre-processing (for example, local histogram equalisation could have improved things still further). Alternatively I may have benefitted from using a lower training rate when the model started making better predictions, since I noticed it jumped around a LOT when it went beyond around 88% on the validation set, and the rate of progress slowed dramatically (and unpredictably).


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimiser which turns out is a faster algorithm to converge than gradient descent (I understand it uses momentum and requires less tuning of the learning rate). The batch size was increased from 128 to 256. The reasoning behind this was to give the model a larger sample of the overall dataset and thus stabilise learning, especially when nearing 93% accuracy.

I have to say David Silver's 10 epochs for 96% accuracy was infuriating - I never got better than 92% for any model with that few epochs, and none bested that score. Gaining those last few percentages was a real struggle - I definitely need a more structured procedure for experimenting. In the end I simply told the model to halt once it achieved the required performance, which was 42 epochs.

The learning rate was left at 0.001, and other than the speed of training I didn't notice it making much of a difference up to that point.

Thankfully I figured how to use AWS, and once a connection was established, it was monstrously quick and learning was dramatically sped up. Should really invest in a GPU or two myself...

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100% (yay!)
* validation set accuracy of 94.6%
* test set accuracy of 92%
* accuracy of 100% on web images

Initially I started off with the LeNet architecture wholesale. It started off well (I read LeCunn's paper, hadn't realised he had used this architecture on traffic signs), and quickly converged to a solid range of 88 to 92% depending on starting values of weights/biases, and in just 10 epochs as well. Clearly I needed to do better to submit.

However, this was where things became nightmarishly difficult! I made a bunch of changes such as adding weights, layers, and implementing dropout. On reflection, I should have metered out these changes one by one. In any case, even when increasing the number of epochs these configurations started off worse and became extremely unstable in their learning beyond around 88%. As I understand it, this is a case of underfitting. Yet I had increased the size of my network, so this was quite bizarre.

I eventually reset to the LeNet architecture and simply increased the number of features in the final convolution, which seemed to do the trick in bumping me past 93%. Perhaps my network was not sufficiently large enough to warrant dropout (which to be fair is a regularisation technique). But I have observed other architectures that aren't drastically larger than mine which DID use dropout with great success. Hence it is a bit mysterious.

In fairness I had not experimented with other activation functions, which potentially could have made a difference. But it seems ReLUs are ideal in that they are non-linear but also computationally inexpensive. 

In my research I discovered a general strategy of navigating multiple convolutional layers into a single fully connected layer such as that found by LeCunn. The intuition seems to be that the lower-level features, which are reliably extracted, serve to cross-reference the higher-level features that we extract in later convolutions. I did try this, but like above I suspect I should have been more progressive in introducing it, for I made no improvement by using it the way I did. 

One thing I do think I was smart to do was create methods for the types of layers. This way I was able to just add another layer or change the sizes of my layers very easily, which did help when messing around with the numbers.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The no-passing image might be difficult to classify because it is fairly blurry, and the no-stop sign's glare in places might throw the network off. The speed signs and the priority road signs do have some strong overall contrasts, which might confuse it if it wasn't focussed on the signs themselves.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20 KMPH 	     		| 20 KMPH	 									| 
| 80 KMPH 	     		| 80 KPMH 	 									| 
| No Entry				| No Entry										|
| No Passing     		| No Passing					 				|
| Priority Road			| Priority Road      							|


The model correctly guessed all of the signs, although it's somewhat confusing since I thought the softmax function guaranteed all numbers would sum to 1. By contrast, my model is actually 100% confident in these predictions, even though it technically makes tiny guesses for one or two other classes as well in some cases:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 20 KMPH 	 									| 
| 1.0     				| 80 KMPH 										|
| 1.0					| No Entry										|
| 1.0	      			| No Passing					 				|
| 1.0				    | Priority Road      							|

I can't possibly blame the tensorflow code, but at the same time the softmax function does accept my arguments and returns incorrect numbers, so I'm a bit lost as to where I could have made a mistake on this!

Overall I feel somewhat mixed about this project - I feel like I've had some good experience finally getting to grips with some of Tensorflow's features and using AWS, but not as well as perhaps I had hoped for personally. I'm definitely keen to revisit this particular project in the future.


