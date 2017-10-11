# encoding: utf-8

from numpy import *

def signomid(inX):
    return longfloat(1.0/(1.0+exp(-inX)))

def classifyVector(inV,weights):
    ans = signomid(sum(array(inV)*array(weights)))
    if ans>0.5:
        return 1
    else:
        return 0

def stocGradAscent(dataMat,labelMat,numIter=150):
    dataMatrix = array(dataMat)
    labelMatrix = array(labelMat)
    m,n = shape(dataMatrix)
    weights = ones(n)
    weights = array(weights)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4.0/(1.0+i+j) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            hypothesis = signomid(sum(dataMatrix[randIndex]*weights))
            error = labelMatrix[randIndex] - hypothesis
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#数据共22列,最后一列是label
def colicTest():
    dataMat = []; labelMat = []
    fTest = open('horseColicTest.txt')
    fTrain = open('horseColicTraining.txt')
    for line in fTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[21]))
    weights = stocGradAscent(dataMat,labelMat,500)
    errorCount = 0
    numTestVec = 0
    for line in fTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr,weights))!=int(currLine[21]):
            errorCount += 1
    errorRate = errorCount/float(numTestVec)
    print('the error rate of this is: %f' %errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0
    for i in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests,errorSum/float(numTests)))


multiTest()
