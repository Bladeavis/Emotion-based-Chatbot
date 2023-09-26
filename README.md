# Sentiment Analysis Comparison und Emotion-based-Chatbot

## Project Contributors
- Mia Theresa Nick (BERT)
- Nurhayat Altunok (Naive Bayes)

## Context
This project was implemented to develop a chatbot that utilizes sentiment analysis models to understand and respond to user input effectively. The project aims to compare different sentiment analysis models to select the most suitable one for the chatbot's implementation.

## Research Questions
Main Research Questions:
1. Which Sentiment Analysis Model is best suited for our application?
2. Which Sentiment Analysis Model exhibits the highest accuracy?
3. What triggers the results of sentiment analysis? (POS-Tagging)

In this research endeavor, we conducted a comparative study of sentiment analysis models to identify the one that aligns best with our chatbot application's requirements. Our primary focus was on model accuracy and reliability.

### Key Findings
- BERT exhibited the highest accuracy, achieving an 83% accuracy rate. It is best suited for our application.
- Categorization of words alone is insufficient for sentiment analysis, as words in a sentence affect each other.
- Complex and lengthy sentences tend to lower the model's accuracy.
- Out of 41 sentences analyzed, 15 were interpreted differently by Naive Bayes and BERT, with BERT producing more accurate results in most cases.

## Data Sources
### NLTK Movie Reviews Dataset
- The primary data source for our sentiment analysis study is the NLTK Movie Reviews dataset.
- This dataset contains movie reviews collected from various internet sources and is categorized into positive and negative sentiments.
- Format: The dataset is available in NLTK's built-in corpus format, organized as text files, with each file representing a single movie review.
- Accessibility: The NLTK Movie Reviews dataset is publicly accessible and can be obtained directly through the NLTK library in Python. To access it, follow these steps:
  1. Install NLTK: `pip install nltk`
  2. Import NLTK and download the dataset (Python):
     ```python
     import nltk 
     nltk.download('movie_reviews')
     ```

## Running the Code
### Python Version
Python version used: 3.11.4 Core: Anaconda

### Naive Bayes
You can open this code from GoogleColab for fast results and operation: https://colab.research.google.com/drive/1VN_lPzProdsQ1FZau1Q9a6MzDzD616KB?usp=sharingd


To run the Naive Bayes model, follow these steps:
1. Install the required libraries: `pip install nltk`
2. Download NLTK data:
   ```python
   import re
   import nltk
   
   nltk.download('punkt')
   nltk.download('words')
   nltk.download('movie_reviews')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

   from nltk.tokenize import word_tokenize
   from nltk.corpus import words
   from nltk.stem import WordNetLemmatizer
   from nltk.corpus import movie_reviews
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.metrics import accuracy_score, classification_report

   
4. Run the Naive Bayes script.

### BERT
## Important Note:
This model takes a long time to load and Data Split (>45 Minutes). For this reason, the BERT file results faster when run through GoogleColab (with T4 GPU - 2 Minutes). Since GoogleColab runs this code using GPU, the Train-Test split and model loading takes 2 minutes, but when you run this code on your own computer on a platform like VS Code, this part of the code takes a very long time (>45 Minutes), so you can see how the code works by clicking on the GoogleColab link below: https://colab.research.google.com/drive/1yQqHO6CPsZcRDi5zXi0xH3TxAa2NnwwQ?usp=sharing 

To run the BERT model, follow these steps:
1. Install the required libraries: `pip install nltk` and `pip install transformers` and `pip install torch` 

2. Download NLTK data:
   ```python
   import re
   import nltk
   import numpy as np
   import torch
   
   nltk.download('punkt')
   nltk.download('words')
   nltk.download('movie_reviews')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

   from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, classification_report
   from nltk.tokenize import word_tokenize
   from nltk.corpus import words
   from nltk.stem import WordNetLemmatizer
   from nltk.corpus import movie_reviews
   from collections import defaultdict

3. Run the BERT script.


### Prototype Chatbot with Sentiments - BERT

Here you can access the combination of the Chatbot we downloaded from Huggingface and the BERT Model that analyzes Sentiment: https://colab.research.google.com/drive/1UVocScWBv3zDgruO6nvds5m0RObqJNDd?usp=sharing

Chatbot Model = "facebook/blenderbot_small-90M"
Sentiment Model = "distilbert-base-uncased"

1. Install the required libraries: `pip install transformers` and `pip install torch` 

   ```python
   import torch
   import numpy as np
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer

2. Run the Prototype Chatbot with Sentiments -BERT script.

