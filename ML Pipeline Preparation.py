#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import numpy as np
import pandas as pd
import pickle
#import os
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.preprocessing import FunctionTransformer

nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])


# In[3]:


# load data from database

engine = create_engine('sqlite:///DisasterResponse.db')

df = pd.read_sql_table('df',engine)
X = df['message']
Y = df.iloc[:,4:]


# In[4]:


df.head()


# In[ ]:





# In[6]:


Y.head()


# In[7]:


Y.isnull().sum()


# In[8]:


X.isnull().sum()


# In[9]:


Y.fillna(0,inplace=True)


# In[10]:


Y.isnull().sum()


# ### 2. Write a tokenization function to process your text data

# In[11]:


def tokenize(text):
    # tokens the text
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizers = WordNetLemmatizer()
    
    # clean tokens
    
    process_tokens = []
    
    for token in tokens:
        token = lemmatizers.lemmatize(token).lower().strip('!"#$%\'()*+,-./:;<=>?@[\\]^_,{|}~')
        token = re.sub(r'\[[^.,;:]]*\]','',token)
        # add token to compiled list
        
        if token !='':
            process_tokens.append(token)
            
    return process_tokens
    
    


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[12]:


pipeline = Pipeline ([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[ ]:





# In[14]:


pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[15]:


y_pred = pipeline.predict(X_test)


# In[16]:


print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=Y.columns))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[17]:


parameters = {'vect__ngram_range': ((1,2), (2,2)),
              #'tfidf__use_idf': (True, False)}
              'clf__estimator__n_estimators': [1, 10]}

cv = GridSearchCV(pipeline, parameters)


# In[18]:


cv.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[19]:


y_pred = cv.predict(X_test)


# In[20]:


print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=Y.columns))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# In[21]:


def compute_text_lenght(data):
    return np.array([len(text) for text in data]).reshape(-1,1)


# In[22]:


pipeline = Pipeline([
    ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                 ('tfidf', TfidfTransformer()),
                                                 ])),
                                                 
                              ('lenght', Pipeline([('count', FunctionTransformer(compute_text_lenght,
                                                                                 validate = False))]))])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])


# In[23]:


pipeline.fit(X_train, y_train)


# In[24]:


y_pred = pipeline.predict(X_test)


# In[25]:


print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=Y.columns))


# In[26]:


parameters = {'features__text__vect__ngram_range': ((1,2), (2,2)),
              #'tfidf__use_idf': (True, False)}
              'clf__estimator__n_estimators': [1, 10]}
                 

cv = GridSearchCV(pipeline, parameters)


# In[27]:


cv.fit(X_train, y_train)


# In[28]:


y_pred = cv.predict(X_test)


# In[29]:


print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=Y.columns))


# In[30]:


print(accuracy_score(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred])))


# ### 9. Export your model as a pickle file

# In[31]:


pickle.dump(cv, open('model.p', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




