import os
import glob
import streamlit as st
from pathlib import Path
import pandas as pd
import nltk

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from drivers.txtToTags import txtToTags
from drivers.imgToTags import imgToTags
from drivers.findSimilarImgs import findSimilarImgs



UPLOAD_DIR = 'temp'  # Directory where uploaded images will be saved
IMAGE_EXTENSIONS = ['png', 'jpg']  # List of accepted image file extensions
DISPLAY_DIR = 'data/flickr30k-images'  # Directory where displayed images will be saved
DISPLAY_COLUMNS = 5  # Number of columns in which to display images

# Function to display saved images in columns
def display_images(imagesList):
    view_images = [os.path.join(DISPLAY_DIR, img) for img in imagesList]

    groups = [view_images[i:i+DISPLAY_COLUMNS] for i in range(0, len(view_images), DISPLAY_COLUMNS)]
    cols = st.columns(DISPLAY_COLUMNS)
    for group in groups:
        for i, image_file in enumerate(group):
            cols[i].image(image_file)

with st.sidebar:
    st.image("https://cdn.discordapp.com/attachments/1068299204327379098/1085281254196379748/model.png")
    st.title("Image caption generator")
    choice = st.radio("Navigation", ["Text", "Image"])
    st.info("This is an application that can be used to generate images using text or other images as input")




if choice == "Image":
    st.title("Upload your image!")
    file = st.file_uploader("Upload Your Image Here", IMAGE_EXTENSIONS)

    # When Submit button is clicked, save uploaded image to UPLOAD_DIR
    if st.button('Submit'):
        if file is not None:
            try:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                save_path = Path(UPLOAD_DIR, file.name)
                with open(save_path, mode='wb') as w:
                    w.write(file.getvalue())
                if save_path.exists():
                    tags = imgToTags(save_path)
                    os.remove(save_path)
                    st.success(tags)
                    df = findSimilarImgs(tags, threshold = 0.8)
                    st.write(df)

                    display_images(df['Image file name'])

                    
            except Exception as e:
                st.error(f'Error saving file: {e}')

    # When Display button is clicked, save uploaded images to DISPLAY_DIR and display them
    # if st.button('Display'):
    #     try:
    #         print('hello')
    #         display_images()


    #     except Exception as e:
    #         st.error(f'Error displaying images: {e}')




if choice == "Text":
    st.title("Insert Your Text here!")
    text_input = st.text_input("Enter some text 👇")
    
    if st.button('Submit'):
        if text_input:
            tags = txtToTags(text_input)
            st.write(tags)

            df = findSimilarImgs(tags, threshold = 0.8)
            st.write(df)

            display_images(df['Image file name'])



                
                