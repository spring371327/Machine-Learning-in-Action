#encoding: utf-8

from numpy import *

def stumpClassify(dataMatrix,dim,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dim]<=threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dim]>threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMatrix = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEstimate = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        Min = dataMatrix[:,i].min()
        Max = dataMatrix[:,i].max()
        stepSize = (Max-Min)/numSteps
        for j in range(-1,int(numSteps)+1):
            threshVal = Min + float(j)*stepSize
            for inequal in ['lt','gt']:
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errorArr = mat(ones((m,1)))
                #print(predictVals)
                #print(labelMatrix)
                errorArr[predictVals==labelMatrix] = 0
                weightError = D.T * errorArr
                if weightError<minError:
                    minError = weightError
                    bestClassEstimate = predictVals.copy()#值传递
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEstimate

def adaBoostTrain(dataArr,classLabels,numIter=40):
    record = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIter):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print('D:',D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        record.append(bestStump)
        #print('classEst:',classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        #print("aggClassEst:",aggClassEst.T)
        aggError = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggError.sum()/m
        print("errorRate:",errorRate)
        if errorRate==0.0:
            break
    return record

def adaClassify(dataToClass,record):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(record)):
        classEst = stumpClassify(dataMatrix,record[i]['dim'],record[i]['thresh'],record[i]['ineq'])
        aggClassEst += record[i]['alpha']*classEst
        #print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeatures = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeatures-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        if int(float(curLine[-1]))==1:
            labelMat.append(1)
        else:
            labelMat.append(-1)
    return dataMat,labelMat

dataMat,labelMat = loadDataSet('horseColicTraining.txt')
record = adaBoostTrain(dataMat,labelMat,50)
testDataMat,testLabelMat = loadDataSet('horseColicTest.txt')
ans = adaClassify(testDataMat,record)
m = shape(testDataMat)[0]
errArr = mat(ones((m,1)))
print(errArr[sign(ans)!=mat(testLabelMat).T].sum())
print("testErrorRate = %.3f:" %(errArr[sign(ans)!=mat(testLabelMat).T].sum()/float(m)))

















































