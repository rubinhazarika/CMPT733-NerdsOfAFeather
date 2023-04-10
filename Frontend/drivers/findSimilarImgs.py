import os, sys, string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from fuzzywuzzy import fuzz


def findSimilarImgs(imgTags, threshold=0.60):
    user_tokens =  ' '.join(imgTags)

    captions = "data/tags30k.txt"
    with open(captions, "r") as f:
        image_data = f.readlines()

    # separate image names and captions
    im_names = [line.split(",")[0].strip() for line in image_data]
    ext_tokens = [line.split(",")[1].strip() for line in image_data]

    sim_DF = pd.DataFrame(list(zip(im_names, ext_tokens)), columns = ["Image file name", "Tokenized caption"])


    fuzzy_ratios = [fuzz.token_sort_ratio(user_tokens, phrase)/100 for phrase in ext_tokens]

    sim_DF['Fuzzy ratio'] = fuzzy_ratios



    fuzzy_threshold = sim_DF[sim_DF['Fuzzy ratio'] >= threshold].sort_values('Fuzzy ratio', ascending=False)

    return (fuzzy_threshold)

