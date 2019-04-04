# These libraries were not written by us
import pandas as pd
import naive_bayes

train1 = pd.read_pickle('data\\dataset 1\\train.pkl')
test1 = pd.read_pickle('data\\dataset 1\\test.pkl')

train2 = pd.read_pickle('data\\dataset 2\\train.pkl')
test2 = pd.read_pickle('data\\dataset 2\\test.pkl')

train3 = pd.read_pickle('data\\dataset 3\\train.pkl')
test3 = pd.read_pickle('data\\dataset 3\\test.pkl')

model1 = naive_bayes.NaiveBayes()
model1.train(train1, ['spam', 'ham'])
print(f'Accuracy on data set 1: {naive_bayes.accuracy(model1, test1)}')

model2 = naive_bayes.NaiveBayes()
model2.train(train2, ['spam', 'ham'])
print(f'Accuracy on data set 2: {naive_bayes.accuracy(model2, test2)}')

model3 = naive_bayes.NaiveBayes()
model3.train(train1, ['spam', 'ham'])
print(f'Accuracy on data set 3: {naive_bayes.accuracy(model3, test3)}')