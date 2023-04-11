# Overview 

### Important Notebooks
* __"TagSimilarity.ipynb:"__ Take user phrase, tokenize using the same algorithm as model then utilize a syntactic similarity method to compare sets of tags for similarity, compiling a shortened file with image file names and tokenized captions that meet similarity criteria.

* __"tagPredictor.ipynb:"__ This notebook allowed the reading of a image and extracting tags from it and to functionize it to make a driver python script from it. 

* __"test8kOldModelCustomData.ipynb:"__ This notebook is used to extract tags from Microsoft COCO Val Dataset using older Xception model trained on untokenized Flickr 8k Data.

* __"testDenseNetCustomData.ipynb:"__ This notebook is used to extract tags from Microsoft COCO Val Dataset using final DenseNet model trained on tokenized Flickr 30k Data.

* __"txtToTags.ipynb:"__ This notebook is used to functionize the process of tags extraction from text. This could then be easily converted into a python script.
