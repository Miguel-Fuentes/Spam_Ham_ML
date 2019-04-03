import os #for getting data from OS
import collections #imported for Counter
import re #regex for bag of words

class Perceptron:
    
    def __init__(self):
        self.weights = {'w0': 0.5}
        self.bias = 0.1
        self.trainingSet = {}
        self.trainSet70 = {}
        self.trainSet30 = {}
        self.testSet = {}
        self.classes = ["ham", "spam"]
        self.epochs = 100
        self.lr = 0.01
        
    def bagOfWords(self, text):
        '''
        Returns a dictionary where key is word and value is the 
        count of that word in the text passed in
        '''
        return dict(collections.Counter(re.findall(r'\w+', text)))
    
    def getData(self, data, directory, trueClass):
        '''
        Takes a dictionary, path to data, and class of data at that path.
        NOTE: THIS WILL NOT WORK FOR ANY OTHER DATA BECAUSE OF DIRECTORY STRUCTURE
        Places the text, a dictionary of bagOfWords and the true class of the data
        in the dictioary passed into this function
        '''
        for dir_entry in os.listdir(directory):
            dir_entry_path = os.path.join(directory, dir_entry)
            if os.path.isfile(dir_entry_path):
                with open(dir_entry_path,encoding='utf8',errors='ignore') as f:
                    text = f.read()
                    data.update({dir_entry_path: {'text': text, 'freqWords': bagOfWords(text), 'trueClass': trueClass}})
    
        
    def vocabSet(self, dataSet):
        '''
        Called inside train method, takes the training data
        and returns a set of all words in the data. The set
        is used to initialize the weights of each word before
        training begins when weights are computed. These words
        become the keys for weights dictionary
        '''
        vocab = []
        for i in dataSet:
            for word in dataSet[i]['freqWords']:
                if word not in vocab:
                    vocab.append(word)
        return vocab

    def computeWeights(self, dataSet, weights, lr, epochs):
        '''
        Takes the dataset, weights, learning rate, and iterations
        as input. This is the core of the perceptron algorithm.
        This is done during training and is called from train()
        Iterates through the dataset and its word frequency
        dictionary and takes the sum of weights of the words times
        the frequency of the words in email. If the word in the email
        is not initialized then it is added and initialized to 0.
        For each email if the sum of weights is greater than 0 then
        the perceptron output is 1. Then update the weights of words
        by multiplying the learning rate times difference between 
        prediction and truth and also the frequency of the word.
        '''
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
        '''
        Takes an instance of the data and the dictionary
        of weights. Takes the sum of products of weight
        of word and frequency of word in dataset. If the
        word is new initialize weight of the word to zero.
        If the sum is greater than 0 classifies as spam
        and as ham if sum is less than 0.
        Called in test method
        '''
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
        '''
        After getData() is called and self.trainingSet has the data,
        this functiona takes that training data and splits it into 
        70% training and 30% testing for hyperparameter training
        The data is store in self.trainSet70 and self.trainSet30
        '''
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
        '''
        Takes training data set, learning rate, and epochs
        as input. Gets a vocab set and initialzies each
        words weight as zero. Then calls compute weights
        to update the weights
        '''
        self.lr = lr
        self.epochs = epochs
          
        trainingSetVocab = self.vocabSet(trainSet)
        
        for i in trainingSetVocab:
            self.weights[i] = 0.0
        
        self.computeWeights(self.trainingSet, self.weights, self.lr, self.epochs)
        
    def test(self, testSet):
        '''
        Takes a test data set as input and classifies each
        instance of the data if the guess matches the true
        class of the instance then increments correctGuesses
        Returns correctGuess and percentage of accuracy
        '''
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
        return correctGuesses, (float(correctGuesses) / float(len(testSet)) * 100.0)

        
def main(trainDir, testDir):
    lrs = [0.01, 0.03, 0.05, 0.1, 0.15]
    epochs = [5, 75, 100]
    maxAccuracy = 0
    bestEpoch = 0
    bestLr = 0
    
    perceptron = Perceptron()
    perceptron.getData(perceptron.trainingSet, trainDir + "/spam", "spam")
    perceptron.getData(perceptron.trainingSet, trainDir + "/ham", "ham")
    perceptron.getData(perceptron.testSet, testDir + "/spam", "spam")
    perceptron.getData(perceptron.testSet, testDir + "/ham", "ham")
    
    print("\nData initialized:")
    print("Started training...")
    
    '''
    Hyper parameter tuning. Selects the best 
    epochs and learning rate based on accuracy
    and those are used to train the full data set
    '''
    perceptron.preTrain()
    for i in range(len(epochs)):
        for j in range(len(lrs)):
            perceptron.train(perceptron.trainSet70, lrs[j], epochs[i])
            guess, acc = perceptron.test(perceptron.trainSet30)
            if acc > maxAccuracy:
                maxAccuracy = acc
                bestEpoch = epochs[i]
                bestLr = lrs[j]
    
    print("\nBest results from hyperparameter tuning:")            
    print('Best Acc: ', maxAccuracy)
    print('best Epoch: ', bestEpoch)
    print('Best lr: ', bestLr)
    
    perceptron.train(perceptron.trainingSet, bestLr, bestEpoch)
    guess, acc = perceptron.test(perceptron.testSet)
    print("\nRESULTS FROM ENTIRE DATASET:")
    print ("\nLearning rate: %.4f" % float(perceptron.lr))
    print ("Number of epochs: %d" % int(perceptron.epochs))
    print ("Emails classified correctly: %d/%d" % (guess, len(perceptron.testSet)))
    print ("Accuracy: %.4f%%" % (float(guess) / float(len(perceptron.testSet)) * 100.0))

        
if __name__ == '__main__':
    print("DATASET 1---------------------------------")
    main('data/dataset 1/train', 'data/dataset 1/test')
    print("\nDATASET 2---------------------------------")
    main('data/dataset 2/train', 'data/dataset 2/test')
    print("\nDATASET 3---------------------------------")
    main('data/dataset 3/train', 'data/dataset 3/test')
