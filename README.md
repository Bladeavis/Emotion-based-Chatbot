# Sentiment Analysis Comparison und Emotion-based-Chatbot

## Project Contributors
- Mia Theresa Nick
- Nurhayat Altunok

## Context
This project was implemented to develop a chatbot that utilizes sentiment analysis models to understand and respond to user input effectively. The project aims to compare different sentiment analysis models to select the most suitable one for the chatbot's implementation.

## Research Questions
Main Research Questions:
1. Which Sentiment Analysis Model is best suited for our application?
2. Which Sentiment Analysis Model exhibits the highest accuracy?
3. What triggers the results of sentiment analysis? (POS-Tagging)

In this research endeavor, we conducted a comparative study of sentiment analysis models to identify the one that aligns best with our chatbot application's requirements. Our primary focus was on model accuracy and reliability.

### Key Findings
- BERT exhibited the highest accuracy, achieving an 83% accuracy rate.
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
Python version used: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]

### Naive Bayes
To run the Naive Bayes model, follow these steps:
1. Install the required libraries: `pip install nltk`
2. Download NLTK data:
   ```python
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
To run the BERT model, follow these steps:
1. Install the required libraries: `pip install nltk` and `pip install transformers`
2. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('words')
   nltk.download('movie_reviews')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   import torch
   from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, classification_report
   import numpy as np
   from nltk.tokenize import word_tokenize
   from nltk.corpus import words
   from nltk.stem import WordNetLemmatizer
   from nltk.corpus import movie_reviews
   from collections import defaultdict
3. Run the BERT script.

