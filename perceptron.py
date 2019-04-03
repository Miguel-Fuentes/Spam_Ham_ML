import os
import collections
import re

class Perceptron:
    
    def __init__(self):
        self.weights = {'w0': 0.5}
        self.bias = 0.1
        self.trainingSet = {}
        self.trainSet70 = {}
        self.trainSet30 = {}
        self.trainingSetVocab = []
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
    
    def preTrain(self):
                
        lenTrain70= round(len(self.trainingSet.keys()) * 0.7) 
        lenTrain30 = len(self.trainingSet.keys()) - lenTrain70
        
        trainingSetKeys = list(self.trainingSet.keys())
        train70Keys = trainingSetKeys[-lenTrain70:]
        train30Keys = trainingSetKeys[:lenTrain30]

        for i in train70Keys:
            self.trainSet70[i] = self.trainingSet[i] 
            
        for j in train30Keys:
            self.trainSet30[j] = self.trainingSet[j]
        
        
    def train(self, trainSet, lr, epochs):
        self.lr = lr
        self.epochs = epochs
          
        #self.trainingSetVocab = self.vocabSet(self.trainingSet)
        self.trainingSetVocab = self.vocabSet(trainSet)
        
        for i in self.trainingSetVocab:
            self.weights[i] = 0.0
        
        self.computeWeights(self.trainingSet, self.weights, self.lr, self.epochs)
        
    def test(self, testSet):
        #self.getData(self.testSet, testDir + "/spam", "spam")
        #self.getData(self.testSet, testDir + "/ham", "ham")
        correctGuesses = 0
        for i in testSet:
            guess = self.classify(testSet[i], self.weights)
            if guess == 1:
                testSet[i]['learnedClass'] = 'spam'
                if testSet[i]['trueClass'] == testSet[i]['learnedClass']:
                    correctGuesses += 1
            if guess == 0:
                testSet[i]['learnedClass'] = 'ham'
                if testSet[i]['trueClass'] == testSet[i]['learnedClass']:
                    correctGuesses += 1
        '''          
        print ("Learning constant: %.4f" % float(self.lr))
        print ("Number of iterations: %d" % int(self.epochs))
        print ("Emails classified correctly: %d/%d" % (correctGuesses, len(testSet)))
        print ("Accuracy: %.4f%%" % (float(correctGuesses) / float(len(testSet)) * 100.0))
        '''
        #print ("Emails classified correctly: %d/%d" % (correctGuesses, len(testSet)))
        return (float(correctGuesses) / float(len(testSet)) * 100.0)

        
def main(trainDir, testDir):
    lrs = [0.01, 0.03, 0.05, 0.1, 0.15]
    epochs = [5, 10, 20, 50, 100]
    maxAccuracy = 0
    bestEpoch = 0
    bestLr = 0
    
    perceptron = Perceptron()
    perceptron.getData(perceptron.trainingSet, trainDir + "/spam", "spam")
    perceptron.getData(perceptron.trainingSet, trainDir + "/ham", "ham")
    perceptron.getData(perceptron.testSet, testDir + "/spam", "spam")
    perceptron.getData(perceptron.testSet, testDir + "/ham", "ham")
    
    print("Data initialized:")
    print("Started training...")
    
    perceptron.preTrain()
    for i in range(len(epochs)):
        for j in range(len(lrs)):
            perceptron.train(perceptron.trainSet70, lrs[i], epochs[j])
            acc = perceptron.test(perceptron.trainSet30)
            if acc > maxAccuracy:
                maxAccuracy = acc
                bestEpoch = epochs[j]
                bestLr = lrs[i]
                
    print('Best Acc: ', maxAccuracy)
    print('best Epoch: ', bestEpoch)
    print('Best lr: ', bestLr)
    
    perceptron.train(perceptron.trainingSet, bestLr, bestEpoch*5)
    acc = perceptron.test(perceptron.testSet)
    print(acc)
    
        
if __name__ == '__main__':
    main('data/dataset 1/train', 'data/dataset 1/test')