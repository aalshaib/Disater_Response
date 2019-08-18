import json
import plotly
import pandas as pd
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
def compute_text_lenght(data):
    return np.array([len(text) for text in data]).reshape(-1,1)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)


# load model
model = joblib.load("../models/classifier.pkl")

#length of text
df['length_of_text'] = compute_text_lenght(df['message'])

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #extract genre 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #extract categories 
    categories_name = list(df.iloc[:,4:].columns)
    categories_corr = df.iloc[:,4:].corr().values
    #extract length of text in each type
    
    direct = df.loc[df.genre=='direct','length_of_text']
    news = df.loc[df.genre=='news','length_of_text']
    social = df.loc[df.genre=='social','length_of_text']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {'data':[
            Heatmap(
                z=categories_corr,
                x=categories_name,
                y=categories_name[::-1]
            )
        ],
         'layout':{
             'title': 'categories correlation',
             'height': 800
         }
        },
        
        {
            'data':[
                Histogram(
                    y=direct,
                    name='direct',
                    opacity=0.5
                ),
                Histogram(
                    y=news,
                    name='news',
                    opacity=0.5
                ),
                Histogram(
                    y=social,
                    name='social',
                    opacity=0.5
                ),
            ],
            'layout':{
                'title':'Length of Text',
                'yaxis':{
                    'title':'Count'
                },
                'xaxis':{
                    'title':'Text Length'
                }
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()