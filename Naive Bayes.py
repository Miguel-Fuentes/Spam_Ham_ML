import collections
import math

# Print current working directory
print("Current working dir : %s" % os.getcwd())

#set of the classes to classify data
Classes = {"c", "j"}

#set of training data
trainingData = {("Chinese Beijing Chinese", "c"),
                ("Chinese Chinese Shanghai", "c"),
                ("Chinese Macao", "c"),
                ("Tokyo Japan Chinese", "j")}
#set of test data
testSet = ("Chinese Chinese Chinese Tokyo Japan")

#Takes in training data and produces a set of all the words from the training data
def ExtractVocabulary(trainSet):
    vocabSet= set()
    for x in trainSet:
        for word in x[0].split():
            vocabSet.add(word)
            #print(word)
    return vocabSet

#Takes in training data and produces the number of documents in the training data
def CountDocs(trainSet):
    numDocs = 0
    for x in trainSet:
        numDocs += 1
    return numDocs

#Takes in training data and a class(label) and produces the number of documents with that class label
def CountDocsInClass(trainSet, aClass):
    numC = 0
    for x in trainSet:
        if x[1]== aClass:
            numC += 1
    return numC

#Takes in training data and a class(label) and produces a string with all words from documents with that class
def ConcatenateTextOfAllDocsInClass(trainSet, aClass):
    allWords = " "
    allWordsList = []
    for x in trainSet:
        if x[1] == aClass:
            for word in x[0].split():
                allWordsList.append(word)
    allWords = allWords.join(allWordsList)
    return allWords

#Takes in a string and a term and produces the number of times term was found in the string
def CountTokensofTerm(atext, term):
    numTerms = 0
    for word in atext.split():
        if word == term:
            numTerms += 1
    return numTerms

def TrainMultinomialNB(setClass, trainsetD):
    V = ExtractVocabulary(trainsetD)
    N = CountDocs(trainsetD)
    prior ={}
    condprob = collections.defaultdict(dict)
    for c in setClass:
        Nc = CountDocsInClass(trainsetD, c)
        prior[c] = Nc/N
        textC = ConcatenateTextOfAllDocsInClass(trainsetD, c)
        for t in V:
            Tct = CountTokensofTerm(textC, t)
            TctT = 0
            for tp in V:
                TctT = TctT + CountTokensofTerm(textC, tp) + 1
            condprob[t][c] = (Tct + 1)/TctT
    return V, prior, condprob

def ExtractTokensFromDoc(Vocab, testData):
    dataTokens = []
    for x in testData.split():
        for word in Vocab:
            if x == word:
                dataTokens.append(word)
    return dataTokens


def ApplyMultinomialNB(setClass, Vocab, PriorKnow, Condprobs, testData):
    W = ExtractTokensFromDoc(Vocab, testData)
    score = {}
    for c in setClass:
        score[c] = math.log(PriorKnow[c])
        for t in W:
            score[c] += math.log(Condprobs[t][c])
    return max(score, key=score.get)

'''
vocab = TrainMultinomialNB(Classes, trainingData)[0]
print(vocab)
prior = TrainMultinomialNB(Classes,trainingData)[1]
print(prior)
condprob = TrainMultinomialNB(Classes, trainingData)[2]
print(condprob)
classification = ApplyMultinomialNB(Classes, vocab, prior, condprob, testSet)
print(classification)
'''
