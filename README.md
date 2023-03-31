# Filtering Image Datasets Using Image-Captioning Neural Networks
### *Nerds of a Feather: Aditya, Gauri, Karishma, Owais, Rubin*


# The Big Picture

The scope of our projects is to develop multiple neural networks that can filter a database of images based on a user-defined phrase. For example, given an unlabelled image dataset and the phrase "Kids playing in grass", our neural network should be able to filter out the images that have kids playing in grass. We intend to test the efficacy of multiple neural networks (with different feature spaces) in filtering images based off user phrases. 

# The How

The neural networks will be trained on the Microsoft COCO dataset (or a subset of it, atleast). These trained models are meant to be able to take in an image and caption it according to vocabulary built up in the network. The next step is to apply these models to a set of images and a user-defined phrase. Our neural network will process every element in the image dataset, assign it a caption and compare the assigned caption to the user phrase (using word similarity measures like cosine similarity, or the like). If the caption and the phrase are sufficiently similar (define a threshold value), the system will recognize that image as being part of the final user-requested dataset. 
   

# Introduction to Image Captioning:
Image Captioning is a computer vision and natural language processing task that involves generating a textual description of an image. The main goal of Image Captioning is to develop a model that can understand the content of an image and describe it in natural language, similar to how humans describe images. 

This task is challenging because it requires the model to not only understand the objects and their relationships in the image, but also generate a coherent sentence that accurately describes the image.

# Encoder-Decoder Framework:
To perform Image Captioning, an encoder-decoder framework is used. In this framework, an input image is first encoded into a fixed-length vector representation using a convolutional neural network (CNN) for feature extraction. The output of the CNN is a high-dimensional feature vector that captures the salient visual features of the image. 

This feature vector is then passed through a fully connected layer to obtain a lower-dimensional representation, which is used as the image embedding. The image embedding is then decoded into a descriptive text sequence using recurrent neural networks (RNNs) such as long short-term memory (LSTM) networks. The LSTM generates the caption one word at a time, conditioning on the image embedding and previously generated words.

# Dataset Used

## Flickr30k 

The Flickr30k dataset is a large-scale benchmark dataset for image captioning and multimodal research. It consists of 31,783 images collected from Flickr, each of which is paired with five human-generated captions describing the content of the image. The dataset was released in 2014 and has since become one of the most widely used datasets for image captioning research.

The images in the dataset are diverse in terms of content, including scenes, objects, people, and animals, captured under various lighting conditions and camera angles. The captions are relatively descriptive, typically consisting of 10-20 words each, and covering different aspects of the image content.

The dataset is commonly used for training and evaluating image captioning models, as well as for other multimodal research tasks, such as image retrieval, visual question answering, and visual grounding. Due to its large size and diverse content, the Flickr30k dataset has contributed significantly to the development of the field of computer vision and natural language processing.

One of the challenges with the Flickr30k dataset is the presence of errors in some of the captions, which can be caused by mistakes or ambiguities in the original annotation process. This can affect the quality of the training data and potentially lead to suboptimal model performance. Therefore, it is important to carefully preprocess and clean the dataset before using it for model training and evaluation.

# Extracting Tags From Captions: 
Using Natural Language Toolkit (NLTK) library, we extract specific tokens from a text file of captions. First, the script reads the captions and tokenizes each caption into individual words. Then, we tag each word with its corresponding part of speech. 

Next, the script extracts specific tokens such as nouns, verbs, and adjectives based on their part of speech using list comprehension. Then, the script combines these tokens into meaningful phrases such as noun phrases, verb phrases, and adjective phrases.

Finally, the script combines all the extracted phrases into a single list of tokens while also removing duplicates.

# Tags Text Preprocessing Steps:
Before inputting the caption text to the model, several preprocessing steps are performed. The text is first converted to lowercase to reduce the vocabulary size. 

Special characters, numbers, and punctuation marks are removed to simplify the text. Extra spaces and single characters are also removed to ensure the text is clean and consistent. 

Finally, start and end tags are added to the caption text to indicate the beginning and end of a sentence.

# Tokenization and Encoded Representation:
The caption text is tokenized and encoded in a one-hot representation before being passed to the embeddings layer. Tokenization involves splitting the caption text into individual words or subwords, while encoding involves mapping each word or subword to a unique integer ID. 

This process converts the text into a numerical representation that can be used as input to the model. The one-hot encoding is used to represent each word as a vector of zeros and ones, where the index corresponding to the word ID is set to one and all other indices are set to zero. 

The encoded caption text is then passed through the embeddings layer to generate word embeddings, which are dense vector representations that capture the semantic meaning of the words.

# Image Feature Extraction:
To extract features from the images, a pre-trained CNN architecture such as DenseNet 201 is commonly used. The CNN is trained on a large dataset of images and learns to identify and extract the most relevant visual features from an image. 

The output of the CNN is a high-dimensional feature vector that captures the visual content of the image. The size of the image embeddings depends on the type of CNN used and can be adjusted to balance the model's performance and resource requirements.

# Data Generation:
Due to the resource-intensive nature of model training, data generation is performed in batches. The inputs for training are the image embeddings and their corresponding caption text embeddings. 

During inference, the text embeddings are passed word by word to the LSTM network to generate the next word in the caption sequence. The data generation process involves randomly selecting a batch of image-caption pairs from the training dataset and generating the image and caption embeddings for each pair. The embeddings are then fed to the model for training.

# Modelling and Modification:
In the modelling stage, the image embeddings are concatenated with the start sequence tag and passed to the LSTM network, which generates words after each input to form a sentence at the end. To improve the model's performance, a modification is made by adding the image feature embeddings to the output of the LSTMs



