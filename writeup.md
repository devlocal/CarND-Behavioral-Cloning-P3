# Behavioral Cloning Project

## Introduction

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./media/crop.jpg "Crop"
[image2]: ./media/nvidia-cnn-architecture.png "NVidia CNN Architecture"
[image3]: ./media/before.png "Before Balancing"
[image4]: ./media/after.png "After Balancing"
[image5]: ./media/visualization.png "Visualization"
[image6]: ./media/loss.png "Loss"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted

The following files are submitted:

Training script, final model, video files:
1. `model.py` - script to build and train a neural network
1. `data_generator.py` - data generator factory used to train the model
1. `model.h5` - trained model
1. `video.mp4` - video of the car driving one lap on track 1
1. `video-track2.mp4` - video of the car driving one lap on track 2

Helper files and utilities:
1. `basic_logging.py` - helper file to set up logging to `stdout`
1. `monitor.py` - helper utility for organizing captured image and CSV files
1. `video.py` - script to produce a video clip from a series of images

Autonomous driving controller:
1. `drive.py` - a script to drive the car

All source files contain comments describing code functionality.

### Driving the car autonomously 

Using the Udacity provided simulator and `drive.py` file, the car can be driven autonomously around any of track 1 or track 2 by executing 

```sh
python drive.py model.h5
```

and starting simulator in autonomous mode.

The same model file `model.h5` can drive the car on both tracks.

### Training script and model architecture

The file `model.py` contains the code for training a convolution neural network and saving the model. The file shows the pipeline I used for training and validation.

The script builds a network using Keras class `Sequential`. It creates a neural network similar to NVidia network described in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. NVidia team achieved good results with this network architecture and I've decided it is a good starting point for my experiments.

The network starts with a cropping cropping layer (line 19) that removes top 57 and bottom 13 pixels of an image. This allows to isolate the track by removing unrelated background parts of an image and part of the hood. A cropped image example is provided below.

![alt text][image1]

The next step is data normalization (line 21) performed by a lambda layer which computes `x / 127.5 - 1.0` for each data point. This step is followed by a neural network, line 53 invokes `build_network` function which adds neural network layers to the model.

NVidia architecture has been selected as a starting point. It is illustrated on the image below.

![alt text][image2]

The original NVidia model defines 5 convolutional layers with filter sizes increasing from 24 to 64, kernel sizes 5x5 and 3x3, followed by 3 fully connected layers. 'ReLU' is used as activation unit for each convolutional layer.

I found that NVidia network provides good performance to drive the car around track 1, however I was not able to drive the car around track 2 confidently using this architecture.

I have extended the network with an additional convolutional layer (line 42 in `model.py`) with 128 filters, kernel size 3x3 and 'ReLU' activation. This deeper network performed better, it can drive the car on both tracks.

In order to prevent overfitting, two dropout layers are added to the network after the second and the last convolutional layers (line 39 and line 43).

The network is trained with 'Adam' optimizer and MSE accuracy metric.

The script sets up two callbacks for checkpointing (line 61) and early stopping (line 64). This allows to run training until validation starts increasing and then save the best performing model.

The script uses `DataGeneratorFactory` from `data_generator.py` to build training and validation iterators. The data is taken from "train" folder and its subfolders. Images from all three "center", "left" and "right" cameras are used. Data is augmented by flipping each image horizontally and negating steering angle. `DataGeneratorFactory` uses 70% of the data for training and 30% for validation.

Training runs for at most 10 epochs, however can stop early if validation loss goes up.

### Collecting Training Data

Lesson materials explain strategies and tactics of data collection in details. I started by driving on the beginning of the first track and using this data for initial model verification.

After I have confirmed that produced model can be used to drive the car autonomously, I have collected more data from driving on track 1. I used three laps of central driving in forward direction, two laps of central driving in reverse direction and three laps of recovery data mixed forward and reverse. Examples of recovery driving data are [recovery1.mp4](media/recovery1.mp4) and [recovery2.mp4](media/recovery2.mp4).

*Note: all example video clips are composed from collected images with the help of `video.py` script.*

During experiments I have noticed that the car doesn't drive confidently over the bridge. To address this issue I have collected additional data from central driving over the bridge in forward and reverse directions. An example of bridge driving data is [bridge.mp4](media/bridge.mp4)

When I got the model that can drive the car on track 1, I collected training data from driving on track 2. From now on, I used all collected data from track 1 and track 2 for training.

I started with two laps of central driving. With a model trained over the collected data the car was able to drive some distance on track 2 but failed to pass one of the turns. I collected data from driving in turns on track 2. Example of data collected in turns is [turns.mp4](media/turns.mp4).

Similarly to track 1, I have collected recovery data from driving on track 2 from a side of the track to the center, alternating left and right sides. This allowed the network to learn how to return to central driving when the car turns away from the center. An example of track 2 recovery data is [recovery3.mp4](media/recovery3.mp4).

Eventually I was able to get the car to drive full lap on track 2, however in one of the turns the wheels of the car went slightly off-track. I have collected more training data by driving in this part of the track twice to fix this issue. An example of collected data is [turn-fix.mp4](media/turn-fix.mp4).

I found that when collecting turns and recovery data, driving with a slower speed results in collecting more images, as for the same virtual distance on the track the simulator takes more images compared to driving with a higher speed. I used this technique and I think it helped me. An example of slow driving can be seen in the last video clip of driving through turns [turn-fix.mp4](media/turn-fix.mp4)

Totally I have collected 1059 MB of data in 75450 images.

### Exploratory visualization and balancing

I visualized collected data by plotting a histogram showing number of samples stored for different steering angles. The data is distributed in 201 buckets:

![alt text][image5]

As seen, there is a spike for 0 degrees steering which corresponds to the car driving straight. There are small spikes at both ends of the histogram that correspond to the car steering left or right with maximum possible angle.

I use data from all three cameras, and correct steering angle for "left" and "right" cameras by `±0.2`. I also augmented data by flipping images and negating steering angles.

A histogram visualizing augmented data set is shown below. It is fully symmetrical since each data frame is duplicated and the copy is flipped around *steering angle = 0* vertical axis. Additionally, all spikes from the original histogram got tripled - steering angles for left and right cameras were computed by taking steering angle from central image and adjusting it by `±0.2`.

![alt text][image3]

I was concerned with three central peaks. The number of samples in these buckets exceeds the number of samples in other buckets by an order of magnitude. My assumption was that unbalanced steering angle buckets can result in poor model performance. To address this concern I tried to balance the data by randomly removing data points from three buckets with peaks. This brought buckets size to a lower level. Bucket balancing can be enabled by setting parameter `balance=True` when calling `DataIterator.initialize` (see `data_generator.py`, line 169). Data distribution after bucket balancing is shown on the histogram below.

![alt text][image4]

Balancing steering angle buckets did not improve performance. It actually resulted in higher training and validation loss and worse driving behavior. Eventually, I disabled bucket balancing.

### Reducing overfitting in the model

When enough training data has been collected, the model started to overfit after the first epoch. Validation loss after the second epoch was higher than after the first epoch and training stopped by `EarlyStopping` callback.

I have added two dropout layers to combat overfitting - one after the second convolutional layer and one more after the last convolutional layer (see `model.py` lines 39 and 43). I experimented with different values of drop rate `p` (tried `0.5`, `0.25`). I also tried to add a dropout layer after each convolutional layer, drop rate was set to `0.05` in this case. However, adding more dropout layers did not improve accuracy, I guess network layers did not see enough training data.

The best results were produced with two dropout layers and `p=0.25`.

### Training accuracy

The chart below shows training and validation loss reported by `model.py` when producing `model.h5`. For the first 5 epochs training and validation loss were decreasing, after the 6th epoch validation loss went up which means the model has probably started overfitting. To avoid further overfitting `EarlyStopping` terminated training.

![alt text][image6]

### Model parameter tuning

#### Learning rate

The model used an adam optimizer, so the learning rate was not tuned manually (model.py `line 56`).

#### Steering angle adjustment

My initial attempts to train a model were made with data from central camera only. This, however, resulted in poor performance, the car was not able to drive around track.

I added images from left and right cameras to training data. Steering angle associated with the image from the central camera was used as is, steering angles for images from left and right cameras were computed by adding a fixed parameter `SIDE_CAMERA_STEERING_ADJUSTMENT` to central camera steering angle (`data_generator.py`, line 26).

I started with adjustment value `0.2`, then tried `0.1` and `0.3`. I found that the car drives best with the value of the adjustment parameter set to `0.2`, so this is the final value.

#### Training and validation split

My initial set of validation data was images and a CSV file collected from a separate recording of central driving. I wanted the model to perform good at driving and assumed that a separately recorded lap of driving would be a good benchmark.

It turned out that it's not. Model trained with this validation set did not drive the car as good as model trained with a subset of training data used for validation. For subsequent training I used a split of 80/20 for training and validation.

The network did not learn well and often it started overfitting before converging to a low loss value. I experimented with changing different parameters, and the best improvement I got was from changing training/validation split from 80/20 to 70/30.

Final ratio of splitting collected data is 70% for training and 30% for validation.

#### Fighting randomness

I have noticed that training and validation loss may change noticeably if training is restarted with the same parameters. The difference may be as big as 15% of loss value. I had two assumptions of what can be the reason of fluctuations: 1. randomness in parameter initialization; 2. randomness is separating data into training and validation sets.

I guessed that the second reason - randomness in separating data into training and validation sets - is the biggest contributor to loss fluctuations.

In order to verify the assumption I initialized python random number generator with a call to `random.seed` with a fixed parameter `0.7` (see `data_generator.py`, line 182) and ran training multiple times one after another. This trick did not help to get rid of validation and training loss fluctuations. There was still a noticeable difference on two consecutive runs of training with all parameters unchanged.

That means, the reason of fluctuations is likely random parameter initialization. Unfortunately the version of Keras used in the project does not allow to pass seed to initializers (e.g. to `glorot_normal` used by default), and I was not able to verify if fixing seed can stabilize loss. Currently I don't have a solution to this issue.

## Data Monitor Utility

I anticipated that data collection is going to be an important activity on the project, and decided to develop a utility that would help me in collecting and organizing data.

The utility script is contained in `monitor.py`. When launched, it starts monitoring `data` folder where I configure driving simulator `mac_sim.app` to save generated data to.

When a new file is added to the `data` folder, the utility waits at least a second to make sure no more new files are added. When new files stop arriving, it moves all image files together with the CSV log file into a new folder in a different location. This way image files and CSV log from each recording appears in a separate subfolder.

Additionally, the utility uses `video.py` to create a video clip. This helps in reviewing captured data. I leveraged this way of collecting and organizing data to review captured material and if necessary delete and re-create a wrongly produced part without affecting the rest of collected data.

Data iterator is created in such a way that it walks over the entire folder hierarchy looking for CVS files, and is fully compatible with the way I organized the data.