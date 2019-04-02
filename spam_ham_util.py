from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def df_to_numeric(train_df, test_df):
    # This will ingore any words that appear in more than 85% of documents
    # or less than %5 of documents
    count_vec = CountVectorizer(max_df=0.85,min_df=0.05)
    
    # Here we train the vectorizer on the training set, her it learn the vocabulary
    X_train = count_vec.fit_transform(train_df['text']).toarray()
    X_train = np.hstack((X_train, np.ones((X_train.shape[0],1))))
    Y_train = (train_df['class'] == 'spam').values.astype(int)
    
    # Here we transform the test set using the vocabulary from the training set
    X_test = count_vec.transform(test_df['text']).toarray()
    X_test = np.hstack((X_test, np.ones((X_test.shape[0],1))))
    Y_test = (test_df['class'] == 'spam').values.astype(int)
    
    return X_train, X_test, Y_train, Y_test