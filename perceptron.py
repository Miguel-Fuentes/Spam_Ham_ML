import os
import collections
import re

class Perceptron:
    
    def __init__(self):
        self.weights = {'w0': 0.1}
        self.bias = 0.1
        self.trainingSet = {}
        self.trainingSetVoocab = []
        self.testSet = {}
        self.classes = ["ham", "spam"]
        self.epochs = 100
        self.lr = 0.01
        
    def bagOfWords(self, text):
        return dict(collections.Counter(re.findall(r'\w+', text)))
    
    def getData(self, data, directory, trueClass):
        for dir_entry in os.listdir(directory):
            dir_entry_path = os.path.join(directory, dir_entry)
            if os.path.isfile(dir_entry_path):
                with open(dir_entry_path,encoding='utf8',errors='ignore') as f:
                    text = f.read()
                    data.update({dir_entry_path: {'text': text, 'freqWords': bagOfWords(text), 'trueClass': trueClass}})
    
        
    def vocabSet(self, dataSet):
        vocab = []
        for i in dataSet:
            for word in dataSet[i]['freqWords']:
                if word not in vocab:
                    vocab.append(word)
        return vocab

    def computeWeights(self, dataSet, weights, lr, epochs):
        for i in range(epochs):
            for data in dataSet:
                sumWeights = weights['w0']
                for j in dataSet[data]['freqWords']:
                    if j not in weights:
                        weights[j] = 0
                    sumWeights += weights[j] * dataSet[data]['freqWords'][j]
                perceptronOutput = 0
                if sumWeights > 0:
                    perceptronOutput = 1
                targetVal = 0
                if dataSet[data]['trueClass'] == 'spam':
                    targetVal = 1
                for k in dataSet[data]['freqWords']:
                    weights[k] += float(lr) * float(targetVal - perceptronOutput) * \
                    float(dataSet[data]['freqWords'][k])
    
    def classify(self, data, weights):
        sumWeights = weights['w0']
        for i in data['freqWords']:
            if i not in weights:
                weights[i] = 0
            sumWeights += weights[i] * data['freqWords'][i]
        if sumWeights > 0:
            return 1 # spam
        else:
            return 0 # ham
        
    def train(self, trainDir):
        self.getData(self.trainingSet, trainDir + "/spam", "spam")
        self.getData(self.trainingSet, trainDir + "/ham", "ham")
        
        self.trainingSetVocab = self.vocabSet(self.trainingSet)
        
        for i in self.trainingSetVocab:
            self.weights[i] = 0.0
        
        self.computeWeights(self.trainingSet, self.weights, self.lr, self.epochs)
        
    def test(self, testDir):
        self.getData(self.testSet, testDir + "/spam", "spam")
        self.getData(self.testSet, testDir + "/ham", "ham")
        correctGuesses = 0
        for i in self.testSet:
            guess = self.classify(self.testSet[i], self.weights)
            if guess == 1:
                self.testSet[i]['learnedClass'] = 'spam'
                if self.testSet[i]['trueClass'] == self.testSet[i]['learnedClass']:
                    correctGuesses += 1
            if guess == 0:
                self.testSet[i]['learnedClass'] = 'ham'
                if self.testSet[i]['trueClass'] == self.testSet[i]['learnedClass']:
                    correctGuesses += 1
                    
        print ("Learning constant: %.4f" % float(self.lr))
        print ("Number of iterations: %d" % int(self.epochs))
        print ("Emails classified correctly: %d/%d" % (correctGuesses, len(self.testSet)))
        print ("Accuracy: %.4f%%" % (float(correctGuesses) / float(len(self.testSet)) * 100.0))

        
def main(trainDir, testDir):
    perceptron = Perceptron()
    
    perceptron.train(trainDir)
    
    perceptron.test(testDir)
        
        
if __name__ == '__main__':
    main('dataset 3/train', 'dataset 3/test')