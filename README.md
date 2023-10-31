# Sentiment Analysis using BERT
[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) is a state-of-the-art ML model for Natural Language Processing. 
BERT is used for NLP  tasks such as, Text classification, Sentiment Analysis, Question Answering and etc.

Here, this project's goal is to classify the sentiment of a review text either "positive" or "negative"

## Dataset 
The model is trained on IMDB review text. The data can be found [here.](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Model
The ```bert-base-uncased``` version of BERT model is loaded from HuggingFace transformers library. And, the sentiment classifier is built on it. 

Further, model is finetuned with [AdamW](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#adamw) optimizer provided by HuggingFace with a learning rate of ```2e-5``` and over ```epochs = 8``` number of Epochs. 

## Results
The model trained and validated achieved accuracy of **95%** and **90%** on respective datasets. 
It also attained an accuracy of **90%** on test data. 

                    precision    recall  f1-score   support

    negative           0.90      0.89      0.90      1232
    positive           0.90      0.91      0.90      1268

    accuracy                               0.90      2500
    macro avg          0.90      0.90      0.90      2500
    weighted avg       0.90      0.90      0.90      2500
    
The model can be downloaded from drive link [here.](https://drive.google.com/drive/folders/1X2k8HcyIr_-d1thesmdy17AGE2nOssag?usp=sharing)

