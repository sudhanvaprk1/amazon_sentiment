"""
This file contains utility functions which are to be used to preprocess raw data 
"""
from properties import get_config
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data():
    """
    Loads the train and resources data from the configured data folder path
    """
    try:
        config = get_config()
        train_data = pd.read_csv(f"{config['data_path']}/train_data.csv")
        resources_data = pd.read_csv(f"{config['data_path']}/resources.csv")
        return train_data, resources_data
    except Exception as e:
        print(e)
        print("Loading of data failed")


def decontracted(phrase):
    """
    Cleans the input phrase
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def get_stopwords():
    """
    Returns configured stopwords
    """
    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
    return stopwords

def preprocess_text(text_data, stopwords):
    """
    Preprocesses text by removing punctuations, expanding short forms to long forms and special characters
    """
    preprocessed_text = []
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text


def preprocess_cat_features(train_data):
    """
    Preprocesses categorical features
    """
    # preprocessing feature project_grade_category
    train_data['project_grade_category'] = train_data['project_grade_category'].str.replace(' ','_')
    train_data['project_grade_category'] = train_data['project_grade_category'].str.replace('-','_')
    train_data['project_grade_category'] = train_data['project_grade_category'].str.lower()

    # preprocessing project_subject_categories
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(' The ','')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(' ','')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace('&','_')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(',','_')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.lower()

    # preprocessing teacher prefix
    train_data['teacher_prefix']=train_data['teacher_prefix'].fillna('Mrs.')
    train_data['teacher_prefix'] = train_data['teacher_prefix'].str.replace('.','')
    train_data['teacher_prefix'] = train_data['teacher_prefix'].str.lower()
    
    # preprocessing project subject categories
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(' The ','')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(' ','')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace('&','_')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(',','_')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.lower()
    train_data['project_subject_subcategories'].value_counts()

    # preprocessing school state
    train_data['school_state'] = train_data['school_state'].str.lower()

    # preprocessing project title
    stopwords = get_stopwords()
    preprocessed_titles = preprocess_text(train_data['project_title'].values, stopwords)
    train_data['project_title'] = preprocessed_titles

    # preprocessing essay
    # merge two column text dataframe: 
    train_data["essay"] = train_data["project_essay_1"].map(str) +\
                            train_data["project_essay_2"].map(str) + \
                            train_data["project_essay_3"].map(str) + \
                            train_data["project_essay_4"].map(str)

    preprocessed_essays = preprocess_text(train_data['essay'].values, stopwords)
    train_data['preprocessed_essays'] = preprocessed_essays

    return train_data


def preprocess_num_features(train_data_cat_prep, resources_data):
    """
    Preprocesses numeric feature price by applying scaling techniques
    """
    price_data = resources_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
    project_data = pd.merge(train_data_cat_prep, price_data, on='id', how='left')

    # applying standard scaling
    scaler = StandardScaler()
    scaler.fit(project_data['price'].values.reshape(-1, 1))
    project_data['std_price']=scaler.transform(project_data['price'].values.reshape(-1, 1))

    # applying min max scaling
    scaler = MinMaxScaler()
    scaler.fit(project_data['price'].values.reshape(-1, 1))
    project_data['nrm_price']=scaler.transform(project_data['price'].values.reshape(-1, 1))

    return project_data



def preprocess_features():
    """
    Preprocesses numerical and categorical features present in the donors choose data
    """
    # loading data
    train_data, resources_data = load_data()

    # preprocessing categorical features
    train_data_cat_prep = preprocess_cat_features(train_data)

    # preprocessing numerical features
    preprocessed_data_all = preprocess_num_features(train_data_cat_prep, resources_data)

    # writing preprocessed data at the data path
    preprocessed_data_all.to_csv("data/man_preprocessed_data.csv")    


if __name__ == "__main__":
    preprocess_features()




    
