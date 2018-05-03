
# Spring 2018 CS543/ECE549
# Assignment 0: Colorizing Prokudin-Gorskii images of the Russian Empire

![](http://slazebni.cs.illinois.edu/spring18/assignment0/prokudin_gorskii.jpg)

## Background
Sergei Mikhailovich Prokudin-Gorskii (1863-1944) was a photographer who, between the years 1909-1915, traveled the Russian empire and took thousands of photos of everything he saw. He used an early color technology that involved recording three exposures of every scene onto a glass plate using a red, green, and blue filter. Back then, there was no way to print such photos, and they had to be displayed using a special projector. Prokudin-Gorskii left Russia in 1918. His glass plate negatives survived and were purchased by the Library of Congress in 1948. Today, a digitized version of the Prokudin-Gorskii collection is available online.

## Overview
The goal of this assignment is to learn to work with images by taking the digitized Prokudin-Gorskii glass plate images and automatically producing a color image with as few visual artifacts as possible. In order to do this, you will need to extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image. Some starter MATLAB code is available here, though you are not required to use it.

## Data
A zip archive with six input images is available here. Note that the filter order from top to bottom is BGR, not RGB!
Detailed instructions

## Detailed instructions
Your program should divide the image into three equal parts (channels) and align two of the channels to the third (you should try different orders of aligning the channels and figure out which one works the best). For each input image, you will need to include in your report the colorized output and the (x,y) displacement vectors that were used to align the channels.

The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels independently for the x and y axis), score each one using some image matching metric, and take the displacement with the best score. There is a number of possible metrics that one could use to score how well the images match. The most basic one is the L2 norm of the pixel differences of the two channels, also known as the sum of squared differences (SSD), which in MATLAB is simply `sum(sum((image1-image2).^2))`. Note that in our case, the images to be matched do not actually have the same brightness values (they are different color channels), so a cleverer metric might work better. One such possibility is normalized cross-correlation (NCC), which is simply the dot product between the two images normalized to have zero mean and unit norm (see MATLAB function `normxcorr2`).

## For Bonus Points
### Multiscale alignment
The archive contains several high-resolution glass plate scans. For these images, exhaustive search over all possible displacements will become prohibitively expensive. To deal with this case, implement a faster search procedure such as an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale implementation. Alternatively, if you have other ideas for speeding up alignment of high-resolution images, feel free to implement and test those.

### Other improvements
Implement and test any additional ideas you may have for improving the quality of the colorized images. For example, the borders of the photograph will have strange colors since the three channels won't exactly align. See if you can devise an automatic way of cropping the border to get rid of the bad stuff. One possible idea is that the information in the good parts of the image generally agrees across the color channels, whereas at borders it does not.

