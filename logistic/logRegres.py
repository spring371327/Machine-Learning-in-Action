# encoding: utf-8

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))

def gradAscent(dataMat,labelMat):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()
    m,n = shape(dataMatrix)
    weights = ones((n,1))
    alpha = 0.001
    maxCycles = 500
    for i in range(maxCycles):
        hypothesis = sigmoid(dataMatrix*weights)
        error = labelMatrix - hypothesis
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    #weights = weights.getA()
    import matplotlib
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if labelMat[i]==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = array(arange(-3.0,3.0,0.1))
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('logistic回归算法')
    #绘图中显示汉字
    matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.show()

def stochasticGradAscent(dataMat,labelMat,numIter = 150):
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4.0/(1.0+i+j) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            hypothesis = sigmoid(sum(dataMat[randIndex]*weights))
            error = labelMat[randIndex] - hypothesis
            weights = weights + alpha*error*array(dataMat[randIndex])
            del(dataIndex[randIndex])
    return weights


dataMat,labelMat = loadDataSet()
#weights = gradAscent(dataMat,labelMat)
#print(weights)
#plotBestFit(weights)
#随机化梯度上升回归
weights = stochasticGradAscent(dataMat,labelMat)
plotBestFit(weights)

















