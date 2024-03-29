import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    return categories, messages

def clean_data(categories,messages):
    df = messages.merge(categories, on = 'id')
    
    print("check df ....")
    print(df)

    categories = categories.categories.str.split(';', expand=True)
    
    print("check categories ....")
    print(categories)
    
    row = categories.loc[0,:].values
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    print("check df ....")
    print(df)
    
    return df


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///'+database_filename)
    
    df.to_sql('df', engine, index=False)
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        categories, messages = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(categories, messages)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()