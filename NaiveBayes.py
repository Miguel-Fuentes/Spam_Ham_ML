import collections
import math
import os


#Function that returns a set contain all the words from the training data
def ExtractVocabulary(trainingSet):
    vocabSet= set()
    for x in trainingSet:
        for word in x[0].split():
            vocabSet.add(word)
    return vocabSet

#Function that returns the number of documents from the training data
def CountDocs(trainingSet):
    numDocs = 0
    for x in trainingSet:
        numDocs += 1
    return numDocs

#Function that returns the number of documents of a certain class from the training data
def CountDocsInClass(trainingSet, aClass):
    numC = 0
    for x in trainingSet:
        if x[1]== aClass:
            numC += 1
    return numC

#Function that returns a string cocntaining all the words from documents of specified class
def ConcatenateTextOfAllDocsInClass(trainingSet, aClass):
    allWords = " "
    allWordsList = []
    for x in trainingSet:
        if x[1] == aClass:
            for word in x[0].split():
                allWordsList.append(word)
    allWords = allWords.join(allWordsList)
    return allWords

#Function that returns the number of times a term was found in a string
def CountTokensofTerm(atext, term):
    numTerms = 0
    for word in atext.split():
        if word == term:
            numTerms += 1
    return numTerms

#Function that returns the vocab, prior and conditional probabilities found from training data
def TrainMultinomialNB(setClasses, trainingSet):
    print("Data initialized:")
    print("Stared training...")
    V = ExtractVocabulary(trainingSet)
    N = CountDocs(trainingSet)
    prior ={}
    condprob = collections.defaultdict(dict)
    TctT = 0
    for c in setClasses:
        Nc = CountDocsInClass(trainingSet, c)
        prior[c] = Nc/N
        textC = ConcatenateTextOfAllDocsInClass(trainingSet, c)
        for t in V:
            Tct = CountTokensofTerm(textC, t)
            if TctT == 0:
                for tp in V:
                    TctT = TctT + CountTokensofTerm(textC, tp) + 1
            condprob[t][c] = (Tct + 1)/TctT
    return V, prior, condprob

#Function that returns a list of the words from a test data instance thats part of the vocab
def ExtractTokensFromDoc(Vocab, testData):
    dataTokens = []
    for x in testData[0].split():
        for word in Vocab:
            if x == word:
                dataTokens.append(word)
    return dataTokens

#Function that returns the class with the highest probability of occuring from the testdata used
def ApplyMultinomialNB(setClass, Vocab, PriorKnow, Condprobs, testData):
    W = ExtractTokensFromDoc(Vocab, testData[0])
    score = {}
    for c in setClass:
        score[c] = math.log(PriorKnow[c])
        for t in W:
            score[c] += math.log(Condprobs[t][c])
    return max(score, key=score.get)

'''
Function that uses the training set to find the prior/conditional probabilities and vocab to calculate which class
an instance of the test set is. Then compares it to the original class of that instance and returns the accuracy or
how many instances had the same classification after using the algorithm 
'''
def Accuracy(setClasses, trainingSet, testSet):
    #print("Initializing trainNB")
    trainNB = TrainMultinomialNB(setClasses, trainingSet)
    correctClass = 0
    for instance in testSet:
        #print("Finding classification of testSet")
        if instance[1] == ApplyMultinomialNB(setClasses, trainNB[0], trainNB[1], trainNB[2], instance):
            correctClass += 1
    #print("Found accuracy")
    acc = correctClass/CountDocs(testSet)
    return acc

#Function that goes through each dataset, puts the training/test data into lists and reports the accuracy
def NaiveBayes():
    trainList = []
    testList = []
    classes = {"spam", "ham"}
    os.chdir("data")
    for dataset in ["dataset 1", "dataset 2", "dataset 3"]:
        os.chdir(dataset)
        for folder in ["train", "test"]:
            os.chdir(folder)
            for classfication in classes:
                os.chdir(classfication)
                for file in os.listdir():
                    filename = os.fsdecode(file)
                    if filename.endswith(".txt"):
                        with open(filename, encoding='utf8', errors='ignore') as f:
                            if folder == "train":
                                trainList.append((f.read(), classfication))
                            elif folder == "test":
                                testList.append((f.read(), classfication))
                os.chdir('..')
            os.chdir('..')
        print(dataset.capitalize() + "----------------------")
        print("The accuracy for " + dataset + " is " + str(Accuracy(classes, trainList, testList)))
        trainList.clear()
        testList.clear()
        os.chdir('..')
    os.chdir('..')


NaiveBayes()

