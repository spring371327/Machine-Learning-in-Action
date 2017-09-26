#_*_coding: utf-8_*_

import numpy
import matplotlib
import operator
import matplotlib.pyplot as plt

#1--实现数据从文件的读入
def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    datingDataMat = numpy.zeros((numberOfLines,3))
    datingLabels = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        datingDataMat[index,:] = listFromLine[0:3]
        datingLabels.append(int(listFromLine[-1]))
        index += 1
    return datingDataMat,datingLabels

#2--实现数据的规格化
def autoNorm(dataSet):
    minVals = numpy.min(dataSet,0)
    maxVals = numpy.max(dataSet,0)
    ranges = maxVals - minVals
    normDataSet = numpy.zeros(numpy.shape(dataSet))
    scale = numpy.shape(dataSet)
    normDataSet = dataSet - numpy.tile(minVals,(scale[0],1))
    normDataSet = normDataSet/numpy.tile(maxVals,(scale[0],1))
    return normDataSet,ranges,minVals

#3--算法的是实现过程
def classify0(inX,datingDataSet,datingLabels,k):
    scale = numpy.shape(datingDataSet)
    diffMat = numpy.tile(inX,(scale[0],1)) - datingDataSet
    sqDiffMat = diffMat**2
    sqDistMat = numpy.sum(sqDiffMat,axis=1)   #axis是按行求和
    distancesMat = sqDistMat**0.5
    sortedDistIndex = numpy.argsort(distancesMat)
    classCount = {}
    for i in range(k):
        voteIlabel = datingLabels[sortedDistIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#4--kNN（k近临)算法测试,用到前10%的数据样本做测试集,并统计算法的正确率
def datingClassTest():
    testScale = 0.10  #10%的测试数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')  #数据从文件读入
    normMat,ranges,minVals = autoNorm(datingDataMat)   #数据规格化
    m = numpy.shape(datingDataMat)  #获取数据集的行数，每行为一个样本
    m = m[0]
    testNum = int(m*testScale)
    errorCount = 0.0
    for i in range(testNum):
        classifierResult = classify0(normMat[i,:],normMat[testNum:m,:],datingLabels[testNum:m],8)  #程序判断此人属于哪种类别
        print("%d ------ %d"%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):     #统计错误测试的个数
            errorCount += 1.0
    print( "the total error rate is: %f" %(errorCount/float(testNum)) )


#5--kNN（k近临)算法的实用
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    #数据的输入
    percentTats = float(input("percentage of time spent playing viedo games?"))
    ffMiles = float(input("frequent flier miles earned per years?"))
    iceCream = float(input("liters of ice cream consumed per years?"))
    #训练数据的读取
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #数据规格化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #将输入的数据规格化成数组
    inArr = numpy.array([ffMiles,percentTats,iceCream])
    #当前输入数据的归一化
    inArr = (inArr - minVals) / ranges
    #判断此人的类别
    classifyResult = classify0(inArr,normMat,datingLabels,8)
    #结果的输出
    print("The result ids:",resultList[classifyResult-1])  #逗号的作用相当于加空格


datingClassTest()
#classifyPerson()