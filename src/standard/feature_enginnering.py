from properties import get_config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
import numpy as np
from scipy.sparse import hstack, coo_matrix, save_npz
from sklearn.preprocessing import Normalizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import Parallel, delayed
from xgboost import XGBClassifier
import pickle
from tqdm import tqdm

def get_preprocessed_data():

    config = get_config()
    data_path = config['data_path']
    preprocessed_data = pd.read_csv(f"{data_path}/preprocessed_data.csv")

    return preprocessed_data

def get_glove_vectors():
    # Loading glove vectors for calculating TFIDF W2V
    with open('artifacts/glove_vectors', 'rb') as f:
        model = pickle.load(f)
        glove_words = set(model.keys())
        return model, glove_words

def stratified_train_test_split(data):

    # Seperating majority and minority class
    data_majority = data[data['project_is_approved'] == 1]
    data_minority = data[data['project_is_approved'] == 0]

    # We will use sampling with replacement strategy
    data_upsampled = resample(data_minority,
                            replace = True,
                            n_samples = len(data_majority),
                            random_state = 42)

    print("Upsampled minority data now has shape: " + str(data_upsampled.shape))

    # Joining the data back
    data = pd.concat([data_majority, data_upsampled])

    # train test split with stratification on target column
    X = data.drop('project_is_approved', axis=1)
    y = data['project_is_approved'].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
    return x_train, x_test, y_train, y_test

def sentiment_collector(text):
    """
    Adds sentiment in x_train and y_train
    """
    results = SentimentIntensityAnalyzer().polarity_scores(text)
    return results


def append_sentiment(x_train, x_test):
    # adding sentiment to x_train and x_test data
    all_sentiments_train = Parallel(n_jobs=-1)(delayed(sentiment_collector)(text) for text in list(x_train['essay'].values))
    all_sentiments_test = Parallel(n_jobs=-1)(delayed(sentiment_collector)(text) for text in list(x_test['essay'].values))

    # appending four scores that vader returns as features in the data
    pd.options.mode.chained_assignment = None
    negative_scores_train = []
    neutral_scores_train = []
    positive_scores_train = []
    compound_scores_train = []

    negative_scores_test = []
    neutral_scores_test = []
    positive_scores_test = []
    compound_scores_test = []

    for entry in all_sentiments_train:
        negative_scores_train.append(entry['neg'])
        neutral_scores_train.append(entry['neu'])
        positive_scores_train.append(entry['pos'])
        compound_scores_train.append(entry['compound'])
        
    for entry in all_sentiments_test:
        negative_scores_test.append(entry['neg'])
        neutral_scores_test.append(entry['neu'])
        positive_scores_test.append(entry['pos'])
        compound_scores_test.append(entry['compound'])
        
    x_train['negative_score'] = negative_scores_train
    x_train['neutral_score'] = neutral_scores_train
    x_train['positive_score'] = positive_scores_train
    x_train['compound_score'] = compound_scores_train
        
    x_test['negative_score'] = negative_scores_test
    x_test['neutral_score'] = neutral_scores_test
    x_test['positive_score'] = positive_scores_test
    x_test['compound_score'] = compound_scores_test
    return x_train, x_test

def feature_vectorization(x_train, x_test, y_train, y_test, df_column, 
                         representation, glove_words, model):
    
    """
    This function will perform TFIDF and TFIDF W2V featurizations on the data
    """
    
    # Initialising vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,4), min_df=10, max_features=10000)

    # Fitting on train data
    tfidf_vectorizer.fit(x_train[df_column].values)
    
    if representation == 'TFIDF':
        
        # Transforming column into vectors
        x_train_column_tfidf = tfidf_vectorizer.transform(x_train[df_column].values)
        x_test_column_tfidf = tfidf_vectorizer.transform(x_test[df_column].values)
        
        # Printing basic information about the op
        print("After Vectorisation: " + str(df_column))
        print("X Train Essay TFIDF: " + str(x_train_column_tfidf.shape))
        print("X Test Essay TFIDF: " + str(x_test_column_tfidf.shape))
        
        # Returning the vectors
        return x_train_column_tfidf, x_test_column_tfidf
    
    else:
        
        # Making a dictionary with key as word and its corresponding IDF as value
        dictionary = dict(zip(tfidf_vectorizer.get_feature_names_out(), list(tfidf_vectorizer.idf_)))
        tfidf_words = set(tfidf_vectorizer.get_feature_names_out())
        
        # Initialising lists to gather tfidf_w2v values for train and test data
        x_train_column_tfidf_w2v = [] 
        x_test_column_tfidf_w2v = []
        
        # Processing train data
        for text in tqdm(x_train[df_column]): 
            vector = np.zeros(300) 
            tf_idf_weight =0; 
            for word in text.split(): 
                if (word in glove_words) and (word in tfidf_words):
                    vec = model[word] 
                    tf_idf = dictionary[word]*(text.count(word)/len(text.split())) 
                    vector += (vec * tf_idf) 
                    tf_idf_weight += tf_idf
            if tf_idf_weight != 0:
                vector /= tf_idf_weight
            x_train_column_tfidf_w2v.append(vector)
        
        # Processing test data
        for text in tqdm(x_test[df_column]): 
            vector = np.zeros(300) 
            tf_idf_weight =0; 
            for word in text.split(): 
                if (word in glove_words) and (word in tfidf_words):
                    # Vector for word
                    vec = model[word] 
                    # Tf Idf score calculation
                    tf_idf = dictionary[word]*(text.count(word)/len(text.split()))
                    # Tfidf weighted W2v calculation
                    vector += (vec * tf_idf) 
                    tf_idf_weight += tf_idf
            if tf_idf_weight != 0:
                vector /= tf_idf_weight
            x_test_column_tfidf_w2v.append(vector)
        
        # Printing basic info about the op
        print("Total number of TFIDF-W2V vectors for train data on feature " + str(df_column) + ": " + str(len(x_train_column_tfidf_w2v)))
        print("Length of a Single vector train data: " + str(len(x_train_column_tfidf_w2v[0])))
        print("Total number of TFIDF-W2V vectors for test data on feature " + str(df_column) + ": " + str(len(x_test_column_tfidf_w2v)))
        print("Length of a Single vector test data: " + str(len(x_test_column_tfidf_w2v[0])))
        
        # Returning train data and test data vectors
        return x_train_column_tfidf_w2v, x_test_column_tfidf_w2v

def norm_numeric_feat(x_train, y_train, x_test, y_test, df_column):
    
    """
    This function will normalise the numerical columns
    """
    
    # Initialising Normalizer
    normalizer = Normalizer()
    
    # Fitting and transforming train and test data
    train_column_norm = normalizer.fit_transform(x_train[df_column].values.reshape(1,-1)).reshape(-1,1)
    test_column_norm = normalizer.fit_transform(x_test[df_column].values.reshape(1,-1)).reshape(-1,1)
    
    # Printing necessary info about the op
    print("After normalization of: " + str(df_column))
    print(train_column_norm.shape, y_train.shape)
    print(test_column_norm.shape, y_test.shape)
    print("="*100)
    
    # Returning train and test columns
    return train_column_norm, test_column_norm

def calculate_unique_val_frequency(data, cat_column, target):
    
    # Getting all the unique categories in a columnn
    unique_cats = data[cat_column].unique()
    
    # calculating positive and negative class frequencies for all of the unique categories
    positive_frequency = [len(data.loc[(data[cat_column] == i) & (data[target] == 1)]) for i in unique_cats]
    negative_frequency = [len(data.loc[(data[cat_column] == i) & (data[target] == 0)]) for i in unique_cats]
    
    # Encoding the positive and negatives frequencies
    encoded_positive_frequency = [positive_frequency[i]/(positive_frequency[i] + negative_frequency[i]) for i in range(len(unique_cats))]
    encoded_negative_frequency = []
    encoded_negative_frequency[:] = [1 - x for x in encoded_positive_frequency]
    
    # Making two dicts containing positive and negative frequencies with respect to each category
    encoded_positive_values = dict(zip(unique_cats, encoded_positive_frequency))
    encoded_negative_values = dict(zip(unique_cats, encoded_negative_frequency))
    
    return encoded_positive_values, encoded_negative_values

def do_response_coding(x_val, y_val, cat_column, target):

    # Preprocessing the x and y dfs/numpy arrays
    x_val = x_val.reset_index()
    y_val = pd.DataFrame(y_val).reset_index()
    concat_data = pd.concat([x_val, y_val], axis=1)
    concat_data[target] = concat_data[0]
    
    # getting the positive and negative frequencies for the cat_columns
    positive_freq_catcolumn, negative_freq_catcolumn = calculate_unique_val_frequency(concat_data, cat_column, target)
    
    # Mapping the positive and negative frequencies and returning the dataframe
    df = pd.DataFrame()
    df[str(cat_column) + '_pos'] = x_val[cat_column].map(positive_freq_catcolumn)
    df[str(cat_column) + '_neg'] = x_val[cat_column].map(negative_freq_catcolumn)
    
    return df

def gather_response_coded_responses(x_val, y_val, cat_columns, target):
    
    response_coded_dfs = Parallel(n_jobs=-1)(delayed(do_response_coding)(x_val, y_val, cat_col, target) for cat_col in cat_columns)
    concatenated_feature_df = pd.concat(response_coded_dfs, axis=1)
    print('Converted the categorical columns using response coding')
    return concatenated_feature_df

def engineer_features():

    # Loading preprocessed data
    data = get_preprocessed_data()

    # train test split
    x_train, x_test, y_train, y_test = stratified_train_test_split(data)

    # adding sentiment features on essay column
    x_train, x_test = append_sentiment(x_train, x_test)

    # segregating sentiment features
    sentiment_data_train = x_train[['negative_score', 'neutral_score', 'positive_score']].values
    sentiment_data_test = x_test[['negative_score', 'neutral_score', 'positive_score']].values
    print("Generated sentiment features")

    # fetching glove model and glove words
    glove_model, glove_words = get_glove_vectors()
    print("Fetched glove vectors..")

    # TFIDF representation of essay feature
    x_train_essay_tfidf, x_test_essay_tfidf = feature_vectorization(x_train, x_test, y_train, y_test, 'essay', 'TFIDF', glove_words, glove_model)
    print("Generated TFIDF vectors for 'essay' feature")

    # # TFIDF W2V representation of essay feature
    # x_train_essay_tfidf_w2v, x_test_essay_tfidf_w2v = feature_vectorization(x_train, x_test, y_train, y_test, 'essay', 'TFIDF-W2V')

    # Normalising numerical columns: price and teacher_number_of_previously_posted_projects
    x_train_price_norm, x_test_price_norm = norm_numeric_feat(x_train, y_train, x_test, y_test, 'price')
    x_train_tnoppp_norm, x_test_tnoppp_norm = norm_numeric_feat(x_train, y_train, x_test, y_test, 'teacher_number_of_previously_posted_projects')    
    print("Normalised price and teacher_number_of_previously_posted_projects")

    # Response coding of categorical columns
    categorical_columns = ['teacher_prefix', 'school_state', 'project_grade_category',
                           'clean_categories','clean_subcategories']

    print("Performing response coding on project_is_approved")
    categorical_columns_train_df = gather_response_coded_responses(x_train, y_train, categorical_columns, 'project_is_approved')
    categorical_columns_test_df = gather_response_coded_responses(x_test, y_test, categorical_columns, 'project_is_approved')
    categorical_data_train = categorical_columns_train_df.values
    categorical_data_test = categorical_columns_test_df.values

    # stacking all engineered data

    # Set 1 - TFIDF 
    final_x_train_tfidf = hstack((x_train_essay_tfidf, categorical_data_train, x_train_price_norm,
                                x_train_tnoppp_norm, sentiment_data_train)).tocsr()

    final_x_test_tfidf = hstack((x_test_essay_tfidf, categorical_data_test, x_test_price_norm,
                                x_test_tnoppp_norm, sentiment_data_test)).tocsr()
    
    print("Stacked all features. Saving sparse feature matrices.")
    save_npz("data/train_tfidf.npz", final_x_train_tfidf)
    save_npz("data/test_tfidf.npz", final_x_test_tfidf)

    with open("data/y_train.npy", "wb") as file:
        np.save(file, y_train)

    with open("data/y_test.npy", "wb") as file:
        np.save(file, y_test)


    # # Set 2 - TFIDF W2V
    # final_x_train_tfidf_w2v = hstack((coo_matrix(x_train_essay_tfidf_w2v), categorical_data_train, x_train_price_norm,
    #                             x_train_tnoppp_norm, sentiment_data_train)).tocsr()

    # final_x_test_tfidf_w2v = hstack((coo_matrix(x_test_essay_tfidf_w2v), categorical_data_test, x_test_price_norm,
    #                             x_test_tnoppp_norm, sentiment_data_test)).tocsr()

    return final_x_train_tfidf, final_x_test_tfidf, y_train, y_test


if __name__ == "__main__":
    lis = engineer_features()
    print(lis)

    