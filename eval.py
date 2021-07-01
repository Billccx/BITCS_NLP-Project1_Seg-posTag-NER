import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def getset(s,st):
    '''
    NER模块中的一个辅助函数，从正确答案中抽取出某个类型的实体
    @param s: 答案行字符串
    @param st: 某类型的实体集合
    @return:
    '''
    if s != '!!!':
        s = s.split()
        for item in s:
            item=item[1:-1].split(',')
            st.add((int(item[0]),int(item[1])))


def getEntity(state,line):
    '''
    获取字符串中的所有命名实体，NER模块中调用
    @param state: 需要获取的实体类型(nt/ns/nr)
    @param line: 句子字符串
    @return: 该类型的实体列表(list)
    '''
    #line=line.strip()
    #line=line.split()
    entity=[]
    l = len(line)
    i = 0
    while (i < l):
        if line[i] == state:
            start = i
            while (i < l and line[i] == state):
                i += 1
            end = i
            entity.append((start,end))
        else:
            i += 1
    return entity


def getOutDict(Dictionary,OutDict,goldtxt,Gold):
    OutDictNum=0 #测试集中未登录词的个数，含有重复
    InDictNum=0
    for line in goldtxt:
        line=line.strip()
        line=line.replace('   ','  ')
        words=line.split('  ')
        seg=[]
        start=0
        for word in words:
            end=start+len(word)
            seg.append((start,end))
            start=end
            if word not in Dictionary:
                OutDict.add(word)
                OutDictNum+=1
            else: InDictNum+=1
        Gold.append(seg)
    return OutDictNum,InDictNum


def getOutDict2(Dictionary,OutDict,goldtxt,Gold):
    OutDictNum=0 #测试集中未登录词的个数，含有重复
    InDictNum=0
    for line in goldtxt:
        line=line.strip()
        line=line.replace('   ',' ')
        words=line.split(' ')
        seg=[]
        start=0
        for word in words:
            end=start+len(word)
            seg.append((start,end))
            start=end
            if word not in Dictionary:
                OutDict.add(word)
                OutDictNum+=1
            else: InDictNum+=1
        Gold.append(seg)
    return OutDictNum,InDictNum


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    '''
    绘制混淆矩阵
    @param cm:
    @param classes:
    @param title:
    @param cmap:
    @return:
    '''
    plt.rc('font', family='Times New Roman', size='8')  #设置字体样式、大小
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # str_cm = cm.astype(np.str).tolist()
    # for row in str_cm:
    #     print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) #侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('ConfusionMatrix.jpg', dpi=900)
    plt.show()


def SegmentEvaluate(OutDict, Predict, Gold, testpath, OutDictNum, InDictNum):
    '''
    对分词结果进行评估
    @param OutDict: 未登录词(set)
    @param Predict: 预测结果
    @param Gold: 正确结果
    @param testpath: 原始文本文件路径
    @param OutDictNum: 未登录词数量
    @param InDictNum:  登录词数量
    @return: 准确率，召回率，F1，未登录词召回率，登录词召回率
    '''
    A=0 #predict 所有词语个数，含重复
    B=0 #gold 所有词语个数，含重复
    correct=0
    OOV=0
    IV=0
    testset = open(testpath, encoding='utf-8')
    for line,segA,segB in zip(testset,Predict,Gold):
        A+=len(segA)
        B+=len(segB)
        setA=set(segA)
        setB=set(segB)
        AandB=setA&setB
        correct+=len(AandB)
        for item in AandB:
            word=line[item[0]:item[1]]
            if word in OutDict:
                OOV+=1
            else:
                IV+=1
    Precision=correct/A
    Recall=correct/B
    F1=2*Precision*Recall/(Precision+Recall)
    OOVRecall=OOV/OutDictNum
    IVRecall=IV/InDictNum
    return Precision,Recall,F1,OOVRecall,IVRecall


def PosTagEvaluate(Tags,pred,testtag):
    '''
    对词性标注结果进行评估
    @param Tags: 词性标签列表
    @param pred: 预测结果 list(list)
    @param testtag: list(string)
    @return: precision 全局准确率，TagPrecision各词性准确率,TagRecall各词性召回率
    '''
    testset=open(testtag, encoding='utf-8')
    cnt=0
    correct=0
    predtagDict={}
    goldtagDict={}
    InterCorrect={}
    y_true=[]
    y_pred=[]
    for Tag in Tags:
        InterCorrect[Tag]=0
        predtagDict[Tag]=0
        goldtagDict[Tag]=0
    for predtag,goldtag in zip(pred,testset):
        goldtag=goldtag.strip()
        goldtag=goldtag.split()
        y_true.extend(goldtag)
        y_pred.extend(predtag)
        if(len(predtag)!=len(goldtag)):
            print(predtag,goldtag)
            continue
        l=len(predtag)
        for i in range(l):
            cnt+=1
            predtagDict[predtag[i]]+=1
            goldtagDict[goldtag[i]]+=1
            if(predtag[i]==goldtag[i]):
                InterCorrect[predtag[i]]+=1
                correct+=1
    precision=correct/cnt
    TagPrecision={}
    TagRecall={}
    for Tag in Tags:
        if(predtagDict[Tag]):
            TagPrecision[Tag]=InterCorrect[Tag]/predtagDict[Tag]
        else:
            TagPrecision[Tag]=0
        if(goldtagDict[Tag]):
            TagRecall[Tag]=InterCorrect[Tag]/goldtagDict[Tag]
        else:
            TagRecall[Tag]=0
    print('The whole precision is {}.'.format(precision*100))
    for Tag in Tags:
        print('Tag {}:'.format(Tag))
        print('Precision:{}\tRecall:{}'.format(TagPrecision[Tag],TagRecall[Tag]))
    confmatrx=confusion_matrix(y_true,y_pred,Tags)
    #print(confmatrx)
    plot_Matrix(np.array(confmatrx,dtype=float),Tags)
    return precision,TagPrecision,TagRecall


def CRFPosTagEvaluate(filepath, Tags):
    '''
    对CRF++的词性标注结果进行评估
    @param filepath: 预测结果路径
    @param Tags: 词性标签列表
    @return: ans(list)正确答案，predict(list)预测结果
    '''
    file=open(filepath,'r',encoding='utf-8')
    ans=[]
    predict=[]

    cnt = 0
    correct = 0
    predtagDict = {}
    goldtagDict = {}
    InterCorrect = {}

    for Tag in Tags:
        InterCorrect[Tag] = 0
        predtagDict[Tag] = 0
        goldtagDict[Tag] = 0

    for line in file:
        if line=='\n': continue
        cnt+=1
        line=line.strip()
        line=line.split()
        pred=line[2] #预测结果
        tag=line[1] #真实标签
        ans.append(tag)
        predict.append(pred)

        predtagDict[pred] += 1
        goldtagDict[tag] += 1
        if (pred == tag):
            InterCorrect[pred] += 1
            correct += 1

    precision = correct / cnt
    TagPrecision = {}
    TagRecall = {}
    for Tag in Tags:
        if (predtagDict[Tag]):
            TagPrecision[Tag] = InterCorrect[Tag] / predtagDict[Tag]
        else:
            TagPrecision[Tag] = 0
        if (goldtagDict[Tag]):
            TagRecall[Tag] = InterCorrect[Tag] / goldtagDict[Tag]
        else:
            TagRecall[Tag] = 0
    print('The whole precision is {}.'.format(precision * 100))

    for Tag in Tags:
        print('Tag {}:'.format(Tag))
        print('Precision:{}\tRecall:{}'.format(TagPrecision[Tag], TagRecall[Tag]))

    confmatrx = confusion_matrix(ans, predict, Tags)
    plot_Matrix(np.array(confmatrx, dtype=float), Tags)
    return ans,predict


def NEREvaluate(gtpath,ntpath,nrpath,nspath,pred):
    ntfile = open(ntpath, 'r', encoding='utf-8')
    nsfile = open(nspath, 'r', encoding='utf-8')
    nrfile = open(nrpath, 'r', encoding='utf-8')
    #predfile = open(pred, 'r', encoding='utf-8')

    ntA=ntB=nsA=nsB=nrA=nrB=0
    ntcorrect=nrcorrect=nscorrect=0
    for nt,nr,ns,p in zip(ntfile,nrfile,nsfile,pred):
        nt=nt.strip()
        ns=ns.strip()
        nr=nr.strip()

        ntans=set()
        nrans=set()
        nsans=set()

        getset(nt,ntans)
        getset(nr,nrans)
        getset(ns,nsans)

        ntA += len(ntans)
        nrA += len(nrans)
        nsA += len(nsans)

        ntpred=set(getEntity('nt',p))
        nspred=set(getEntity('ns',p))
        nrpred=set(getEntity('nr',p))

        ntB += len(ntpred)
        nrB += len(nrpred)
        nsB += len(nspred)

        ntcorrect += len(ntans & ntpred)
        nrcorrect += len(nrans & nrpred)
        nscorrect += len(nsans & nspred)

    ntprecision=ntcorrect/ntB
    ntrecall=ntcorrect/ntA
    ntF1=2*ntprecision*ntrecall/(ntprecision+ntrecall)

    nrprecision = nrcorrect / nrB
    nrrecall = nrcorrect / nrA
    nrF1 = 2 * nrprecision * nrrecall / (nrprecision + nrrecall)

    nsprecision = nscorrect / nsB
    nsrecall = nscorrect / nsA
    nsF1 = 2 * nsprecision * nsrecall / (nsprecision + nsrecall)

    return ntprecision,ntrecall,ntF1,nrprecision,nrrecall,nrF1,nsprecision,nsrecall,nsF1



