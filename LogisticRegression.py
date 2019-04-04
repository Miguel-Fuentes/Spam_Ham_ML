# These libraries were not written by us
import pandas as pd
from collections import namedtuple

# This library was written by us
import spam_ham_util

DataSet = namedtuple('DataSet',['X_train', 'X_test', 'Y_train', 'Y_test'])

train1 = pd.read_pickle('data/dataset 1/train.pkl')
test1 = pd.read_pickle('data/dataset 1/test.pkl')

train2 = pd.read_pickle('data/dataset 2/train.pkl')
test2 = pd.read_pickle('data/dataset 2/test.pkl')

train3 = pd.read_pickle('data/dataset 3/train.pkl')
test3 = pd.read_pickle('data/dataset 3/test.pkl')

dataset1 = DataSet(*spam_ham_util.df_to_numeric(train1, test1))
dataset2 = DataSet(*spam_ham_util.df_to_numeric(train2, test2))
dataset3 = DataSet(*spam_ham_util.df_to_numeric(train3, test3))

# This accuracy calculator was not written by us
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# This library was written by us
import log_regression

ITERS, LEARNING_RATE = 100000, 0.01

for l2_reg in [0, 0.15, 0.3, 0.45, 0.6, 0.75]:
    accs = []
    for dataset in [dataset1, dataset2, dataset3]:
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X_train, dataset.Y_train, test_size=0.30, random_state=17)
        
        model = log_regression.log_regression(ITERS, l2_reg, LEARNING_RATE)
        model.train(X_train, Y_train)
        
        accs.append(accuracy_score(model.predict(X_test), Y_test))
    print(f'Average accuracy with l2_reg = {l2_reg} is: {sum(accs)/len(accs)}')

best_reg_val = 0.45

for index, dataset in enumerate([dataset1, dataset2, dataset3]):
    model = log_regression.log_regression(ITERS, best_reg_val, LEARNING_RATE)
    model.train(dataset.X_train, dataset.Y_train)
    
    print(f'Accuracy of dataset{index + 1} is : {accuracy_score(model.predict(dataset.X_test), dataset.Y_test)}')
    