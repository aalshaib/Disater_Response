import sys
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





def load_data(database_filepath):
    #DisasterResponse.db
    engine = create_engine('sqlite:///'+database_filepath)

    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    Y.fillna(0, inplace=True)
    
    
    return X, Y, category_names


    
    


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


def build_model():
    
    #def compute_text_lenght(data):
       # return np.array([len(text) for text in data]).reshape(-1,1)
    
    model = Pipeline ([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]),                   target_names=category_names))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
       
        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()