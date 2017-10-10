# encoding: utf-8

from numpy import *

#将一段文本处理成单词列表
def textParse(textString):
    import re
    listOfTokens = re.split('\W+',textString)
    returnList = []
    for token in listOfTokens:
        if len(token)>2:
            returnList.append(token.lower())
    return returnList

#将docList处理成词汇表
def createVocabList(docList):
    vocabList = set([])
    for doc in docList:
        vocabList = vocabList | set(doc)
    return list(vocabList)

def wordList2Vec(vocabList,wordList):
    returnVec = [0]*len(vocabList)
    for word in wordList:
        if word in vocabList:  #此地方的判断就此文件的实现来说有点多余,word一定会出现在vocabList中,换个场景不一定
            returnVec[vocabList.index(word)] = 1  #词集模型
            #returnVec[vocabList.index(word)] += 1 #词袋模型
    return returnVec

def trainNB(trainMat,trainClasses):
    numTrainDoc = len(trainMat)
    numWords = len(trainMat[0])
    p0Denom = 2.0; p1Denom = 2.0
    p0Vec = ones(numWords); p1Vec = ones(numWords)
    pSpam = sum(trainClasses)/float(len(trainClasses))
    for i in range(numTrainDoc):
        if trainClasses[i]==0:
            p0Vec = p0Vec+trainMat[i]
            p0Denom = p0Denom+sum(trainMat[i])
        else:
            p1Vec = p1Vec+trainMat[i]
            p1Denom = p1Denom+sum(trainMat[i])
    p0Vec = log(p0Vec/float(p0Denom))
    p1Vec = log(p1Vec/float(p1Denom))
    return p0Vec,p1Vec,pSpam

def classifyNB(Vec,p0V,p1V,pSpam):
    p0 = sum(Vec*p0V) + log(1.0-pSpam)
    p1 = sum(Vec*p1V) + log(pSpam)
    if p0>p1:
        return 0
    else:
        return 1

#将多个函数封装在一起,该函数测试朴素贝叶斯算法的正确率
def spamTest():
    docList = []; classList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    print(vocabList)
    trainingSet = list(range(50)); testSet = []
    #随机找10个做测试数据(下标)
    for i in range(10):
        randomIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    #算法训练
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(wordList2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = wordList2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1
    print("errorCount = %d" %errorCount)

spamTest()






