# Filtering Image Datasets Using Image-Captioning Neural Networks
### *Nerds of a Feather: Aditya, Gauri, Karishma, Owais, Rubin*


## The Big Picture

The scope of our projects is to develop multiple neural networks that can filter a database of images based on a user-defined phrase. For example, given an unlabelled image dataset and the phrase "Kids playing in grass", our neural network should be able to filter out the images that have kids playing in grass. We intend to test the efficacy of multiple neural networks (with different feature spaces) in filtering images based off user phrases. 

## The How

The neural networks will be trained on the Microsoft COCO dataset (or a subset of it, atleast). These trained models are meant to be able to take in an image and caption it according to vocabulary built up in the network. The next step is to apply these models to a set of images and a user-defined phrase. Our neural network will process every element in the image dataset, assign it a caption and compare the assigned caption to the user phrase (using word similarity measures like cosine similarity, or the like). If the caption and the phrase are sufficiently similar (define a threshold value), the system will recognize that image as being part of the final user-requested dataset. 
   





