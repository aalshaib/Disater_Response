# Disaster Response Pipeline Project

### overview :
There are a lot of different problems in life. Many organizations react to different types of disasters and  take care of them by  observing messages to understand the needs of the situations. Organizations normaly have the least capacity to understand messages during a big disaster, so predictive modeling can help classify different messages more efficiently.

In this project:
The ETL pipeline that cleaned messages by  using regex and NLTK. 

The ML pipeline was for building a model to train, predict messages.


### Files:
- process_data.py:  script that clean data and splitting up categories and making new columns for each as target variables.

- train_classifier.py: Script to tokenize messages from cleaned data and create new columns through feature engineering. The data features are trained with a ML pipeline. 

- run.py: to run Flask app that classifies messages based on the model and shows data visualizations in heat map and histogram .

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
