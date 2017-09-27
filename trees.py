#_*_ coding: utf-8_*_

import numpy
import matplotlib
import operator
from math import log

#给定一个集合,计算该集合的信息熵(信息的杂乱程度)
def calShannonEntropy(dataSet):
    numberOfEntry = len(dataSet)
    labelCount = {}
    for item in dataSet:
        currentLabel = item[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEntropy = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numberOfEntry
        shannonEntropy -= prob * log(prob,2)
    return shannonEntropy

#划分数据集合,将同一个类别的划分到returnDataSet集合中并返回
def spliteDataSet(dataSet,axis,value):
    returnDataSet = []
    for item in dataSet:
        if item[axis]==value:
            reduceItem = item[:axis]
            reduceItem.extend(item[axis+1:])
            returnDataSet.append(reduceItem)
    return returnDataSet

#选择最好的分类属性进行分类,找到用哪个属性分类能得到的信息增益最大值
def chooseBestFeatureToSplit(dataSet):
    numberOfFeature = len(dataSet[0]) - 1
    baseEntropy = calShannonEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numberOfFeature):
        labelList = []
        for j in range(len(dataSet)):
            labelList.append(dataSet[j][i])
        uniqueVals = set(labelList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = spliteDataSet(dataSet,i,value)
            prob = float(len(subDataSet))/len(dataSet)
            newEntropy += prob*calShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if( infoGain>bestInfoGain ):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#找一个出现次数最多的类型
def majorityClass(classList):
    classCount = {}
    for item in classList:
        classCount[item] = classCount.get(item,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#构建决策树
def createTree(dataSet,labels):
    classList = []
    for item in dataSet:
        classList.append(item[-1])
    #递归退出条件
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityClass(classList)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeature]
    myTree = {labels[bestFeature]:{}}
    del(labels[bestFeature])
    featureValue = []
    for item in dataSet:
        featureValue.append(item[bestFeature])
    uniqueVals = set(featureValue)
    for value in uniqueVals:
        subLabels = labels[:] #每次都用一个新的subLabels,消除python中的列表引用传递的影响
        myTree[bestLabel][value] = createTree(spliteDataSet(dataSet,bestFeature,value),subLabels)
    return myTree

#构造测试数据用的函数
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

dataSet,labels = createDataSet()
myTree = createTree(dataSet,labels)
print(myTree)




















