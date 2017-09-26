#_*_ coding: utf-8_*_

import numpy
from os import listdir
import operator


#数字图像格式化为1*1024的矩阵
def img2vector(filename):
    returnVector = numpy.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,i*32+j] = int(lineStr[j])
    return returnVector


#训练数据的读入
def getTrainingDataMat():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    numberOfFile = len(trainingFileList)
    trainingMat = numpy.zeros((numberOfFile,1024))
    for i in range(numberOfFile):
        fileNameStr = trainingFileList[i]
        nameStr = fileNameStr.split('.')[0]
        classNumber = int(nameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
    return trainingMat,hwLabels

def classify0(inX,trainMat,labels,k):
    dataSize = numpy.shape(trainMat)[0]
    diffMat = trainMat - numpy.tile(inX,(dataSize,1))
    sqDiffMat = diffMat**2
    sqDistMat = numpy.sum(sqDiffMat,axis = 1)   #axis=1,按行求和
    distMat = sqDistMat**0.5
    sortedDistMatIndex = numpy.argsort(distMat)
    classCount = {}
    index = 0
    for i in range(k):
        voteIlabel = labels[sortedDistMatIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#测试,看该算法的正确率
def trainTest():
    trainMat,hwLabels = getTrainingDataMat()
    trainingFileList = listdir('testDigits')
    numberOfFile = len(trainingFileList)
    errorCount = 0.0
    for i in range(numberOfFile):
        fileNameStr = trainingFileList[i]
        nameStr = fileNameStr.split('.')[0]
        classNumber = int(nameStr.split('_')[0])
        inX = img2vector('testDigits/%s' %(fileNameStr))
        classifierNumber = classify0(inX,trainMat,hwLabels,3)
        print("the cassifier come back with: %d, the real answer is: %d" %(classifierNumber,classNumber))
        if( classifierNumber!=classNumber ):
            errorCount += 1.0
    print("the total number of errors is: %d" %errorCount)
    print("the total error rate is: %f" %(errorCount/numberOfFile))

trainTest()
