import pandas as pd
from collections import defaultdict, Counter, namedtuple
from math import log

def accuracy(model, data):
    correct = (data['class'] == model.predict(data['text'])).sum()
    return correct / len(data)

class NaiveBayes:
    def __init__(self):
        self.count_total_tuple = namedtuple('count_total_tuple',['counts','total'])
        self.counts = {}    
    
    def train(self, data, classes):
        for _class in classes:
            all_words = ''.join(list(data[data['class'] == _class]['text'])).split()
            word_counts = dict(Counter(all_words))
            self.counts[_class] = self.count_total_tuple(defaultdict(int,word_counts),len(all_words))
            
    def classify(self, doc):
        max_log_prod = float('-inf')
        classification = None
        for _class, count_tuple in self.counts.items():
            words = doc.split()
            word_counts = [count_tuple.counts[word] + 1 for word in words]
            oov_words = word_counts.count(1)
            log_freqs = [log(count/(len(count_tuple.counts) + oov_words + count_tuple.total)) for count in word_counts]
            total = sum(log_freqs)
            if total > max_log_prod:
                max_log_prod = total
                classification = _class
        return classification
    
    def predict(self, doc_series):
        return doc_series.apply(lambda x: self.classify(x))