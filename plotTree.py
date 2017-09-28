#_*_ coding: utf-8_*_

import matplotlib
import numpy
import matplotlib.pyplot as plt
import trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xytext=centerPt,bbox=nodeType,arrowprops=arrow_args,va="center",ha="center")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if( type(secondDict[key]) == dict ):
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(centerPt,parentPt,textString):
    midX = (parentPt[0]-centerPt[0])/2.0 + centerPt[0]
    midY = (parentPt[1]-centerPt[1])/2.0 + centerPt[1]
    createPlot.ax1.text(midX,midY,textString)


def plotTree(myTree,parentPt,nodeText):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntPt = (plotTree.xOff + float(numLeafs)/plotTree.totalW/2.0 + 0.5/plotTree.totalW,plotTree.yOff)
    plotMidText(cntPt,parentPt,nodeText)
    plotNode(firstStr,cntPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key],cntPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(facecolor='white')
    fig.clf()
    createPlot.ax1 = fig.add_subplot(111)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    #plotNode(u'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    #plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.show()

def classify(inTree,labels,testVec):
    firstStr = list(inTree.keys())[0]
    secondDict = inTree[firstStr]
    index = labels.index(firstStr)
    for key in secondDict.keys():
        if key==testVec[index]:
            if type(secondDict[key])==dict:
                classLabel = classify(secondDict[key],labels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inTree,fw)
    fw.close()
def reloadTree(filename):
    import pickle
    fr = open(filename,'rb')
    intree = pickle.load(fr)
    fr.close()
    return intree


dataSet = []
labels = ['age','prescript','astigmatic','tearRate']
fr = open('lenses.txt')
for line in fr.readlines():
   dataSet.append(line.strip().split('\t'))
#intree = trees.createTree(dataSet,labels)
intree = reloadTree('intree.txt')
createPlot(intree)
storeTree(intree,'intree.txt')





#myTree = {'no surfacing':{0:'no',1:{'flipper':{0:'no',1:'yes'}},3:'maybe'}}





















