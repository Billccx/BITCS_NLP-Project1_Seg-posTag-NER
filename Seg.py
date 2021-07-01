import numpy as np
import time
from Viterbi import *
from utils import *
from eval import *
import random
import json

def Init(Tags,Trans,Emission,Begin,TagNum):
    for Tagi in Tags:
        temp={}
        for Tagj in Tags:
            temp[Tagj]=0.
        Trans[Tagi]=temp
        Emission[Tagi]={}
        Begin[Tagi]=0
        TagNum[Tagi]=0


def getTag(word):
    l=len(word)
    Tag=[]
    if(l==1):
        Tag.append('S')
    elif(l==2):
        Tag.extend(['B','E'])
    elif(l>2):
        Tag.extend(['B']+['M']*(l-2)+['E'])
    else:
        print('Error')
    return Tag


def train(trainset,Tags,Trans,Emission,Begin,TagNum,Dictionary):
    cnt=0
    for line in trainset:
        cnt+=1
        sTag=[]
        wordlist=[]
        line=line.strip()
        if(line==''): continue
        sentence = line.replace(' ', '')
        line=line.replace('   ','  ')
        line=line.split('  ')
        for word in line:
            wordlist.append(word)
            sTag.extend(getTag(word))
        if(len(sentence)!=len(sTag)):
            print('Error')
            print(line)

        Begin[sTag[0]]+=1.
        for i in range(len(sTag)-1):
            Trans[sTag[i]][sTag[i+1]]+=1.

        for index,cha in enumerate(sentence):
            TagNum[sTag[index]] += 1.
            for Tag in Tags:
                if(cha not in Emission[Tag]):
                    Emission[Tag][cha]=0.
                if(sTag[index]==Tag):
                    Emission[Tag][cha]+=1.
        Dictionary.update(wordlist)
    return cnt


def normalize(Tags,Trans,Emission,Begin,TagNum,SentenceNum):
    #初始状态向量
    for Tag in Tags:
        if(Begin[Tag]):
            Begin[Tag]=np.log(Begin[Tag]/SentenceNum)
        else:
            Begin[Tag]=-float('inf')
    #状态转移矩阵
    for i in Tags:
        linesum=0.
        for j in Tags:
            linesum+=Trans[i][j]
        for j in Tags:
            if(Trans[i][j]):
                Trans[i][j]=np.log(Trans[i][j]/linesum)
            else:
                Trans[i][j]=-float('inf')
    #发射矩阵
    for i in Tags:
        for cha in Emission[i]:
            if(Emission[i][cha]):
                Emission[i][cha]=np.log(Emission[i][cha]/TagNum[i])
            else:
                #Emission[i][cha]=-float('inf')
                Emission[i][cha] = np.log(1 / TagNum[i])


def tst(path,Tags,Trans,Emission,Begin,Predict,isSave=False,savepath=None):
    testset=open(path,'r',encoding='utf-8')
    if isSave:
        if savepath: o=open(savepath, 'w', encoding='utf-8')
        else: o = open('./corpus/segment/HMMsegment.txt', 'w', encoding='utf-8')
    for line in testset:
        if(line=='\n' or len(line)==0): continue
        if(line[-1]=='\n'):
            line=line[:-1]
        state=Viterbi2(line,Tags,Trans,Emission,Begin)
        s=''
        for i in range(len(state)):
            if(state[i]=='S' or state[i]=='E'):
                s+=line[i]+' '
            else: s+=line[i]
        #print(s)
        seg=s.split()
        pos = []
        start=0
        for word in seg:
            end=start+len(word)
            pos.append((start,end))
            start=end
        Predict.append(pos)

        if isSave:
            s2 = ''
            for item in pos:
                s2+=line[item[0]:item[1]]+' '
            o.write(s2+'\n')
    o.close()



class WordDict:
    def __init__(self,dicpath=None,dic=None,shuffle=None):
        self.dic={}
        self.getDict(dicpath,dic)
        self.isShuffle=shuffle

    def getDict(self,dicpath,dic):
        if dicpath:
            f=open(dicpath,encoding='utf-8')
            for line in f:
                line=line.strip()
                word=line.split()[0]
                if(word in self.dic):
                    self.dic[word]+=1
                else:
                    self.dic[word]=1
        if dic:
            for item in dic:
                self.dic[item]=1


class RMM(WordDict):
    def ReverseMaximumMatching(self,sentence):
        l=len(sentence)
        seg=[]
        end=l
        while end>0:
            temp=end-1
            for start in range(0,end):
                if sentence[start:end] in self.dic:
                    temp=start
                    break
            seg.append((temp,end))
            end=temp
        seg.reverse()
        return seg

    def tst(self,testset,isSave=False,savepath=None):
        f = open(testset, encoding='utf-8')
        if isSave:
            if savepath: o = open(savepath, 'w', encoding='utf-8')
            else: o=open('./corpus/segment/RMMsegment.txt', 'w', encoding='utf-8')
        predict = []
        for sentence in f:
            sentence = sentence.strip()
            seg = self.ReverseMaximumMatching(sentence)
            predict.append(seg)
            if isSave:
                s = ''
                for item in seg:
                    s+=sentence[item[0]:item[1]]+' '
                o.write(s+'\n')
        o.close()
        return predict


class DAG(WordDict):
    def BuildDAG(self,sentence):
        self.n = len(sentence)
        self.DAG={}
        self.dis=[10000 for _ in range(self.n+1)]
        self.vis=[0 for _ in range(self.n+1)]
        self.path=[-1 for _ in range(self.n+1)]

        for i in range(self.n):
            self.DAG[i]=[i+1]
        self.DAG[self.n]=[]

        for start in range(self.n+1):
            for end in range(start+1,self.n+1):
                if sentence[start:end] in self.dic and end not in self.DAG[start]:
                    self.DAG[start].append(end)


    def dijkstra(self):
        self.dis[0]=0
        self.vis[0]=1
        for node in self.DAG[0]:
            self.dis[node]=1
            self.path[node]=0

        for i in range(0,self.n):
            mindis=1000000
            next=-1
            for node in range(0,self.n+1):
                if self.vis[node]==0 and self.dis[node]<mindis:
                    mindis=self.dis[node]
                    next=node

            self.vis[next]=1
            for end in self.DAG[next]:
                if self.dis[next]+1<self.dis[end]:
                    self.dis[end]=self.dis[next]+1
                    self.path[end]=next

        p=[]
        pre=self.n
        while(pre!=-1):
            p.append((self.path[pre],pre))
            pre=self.path[pre]
        p.reverse()
        return p[1:]

    def DAGsegment(self,sentence):
        self.BuildDAG(sentence)
        seg=self.dijkstra()
        return seg

    def tst(self,testset,isSave=False,savepath=None):
        f = open(testset, encoding='utf-8')
        if isSave:
            if savepath: o=open(savepath, 'w', encoding='utf-8')
            else: o=open('./corpus/segment/DAGsegment.txt', 'w', encoding='utf-8')
        predict = []
        for sentence in f:
            sentence = sentence.strip()
            seg = self.DAGsegment(sentence)
            predict.append(seg)
            if isSave:
                s = ''
                for item in seg:
                    s+=sentence[item[0]:item[1]]+' '
                o.write(s+'\n')
        o.close()
        return predict


class DAG2(WordDict):
    def BuildDAG(self,sentence):
        self.n = len(sentence)
        self.DAG={}
        self.dis=[10000 for _ in range(self.n+1)]
        self.vis=[0 for _ in range(self.n+1)]
        self.path=[[] for _ in range(self.n+1)]
        for i in range(self.n):
            self.DAG[i]=[i+1]
        self.DAG[self.n]=[]

        for start in range(self.n+1):
            for end in range(start+1,self.n+1):
                if sentence[start:end] in self.dic and end not in self.DAG[start]:
                    self.DAG[start].append(end)

    def dijkstra(self):
        self.dis[0]=0
        self.vis[0]=1
        self.path[0].append(-1)
        for node in self.DAG[0]:
            self.dis[node]=1
            self.path[node].append(0)

        for i in range(0,self.n):
            mindis=1000000
            next=-1
            for node in range(0,self.n+1):
                if self.vis[node]==0 and self.dis[node]<mindis:
                    mindis=self.dis[node]
                    next=node

            self.vis[next]=1
            for end in self.DAG[next]:
                if self.dis[next]+1<self.dis[end]:
                    self.dis[end]=self.dis[next]+1
                    self.path[end].clear()
                    self.path[end].append(next)
                elif self.dis[next]+1==self.dis[end]:
                    self.path[end].append(next)
        p=[]
        pre=self.n
        while(pre!=-1):
            if self.path[pre]:
                x=random.choice(self.path[pre])
            p.append((x,pre))
            pre=x
        p.reverse()
        return p[1:]

    def DAGsegment(self,sentence):
        self.BuildDAG(sentence)
        seg=self.dijkstra()
        return seg

    def tst(self,testset,isSave=False,savepath=None):
        f = open(testset, encoding='utf-8')
        if isSave:
            if savepath: o = open(savepath, 'w', encoding='utf-8')
            else: o=open('./corpus/segment/DAG2segment.txt', 'w', encoding='utf-8')
        predict = []
        for sentence in f:
            sentence=sentence.strip()
            seg = self.DAGsegment(sentence)
            predict.append(seg)
            if isSave:
                s = ''
                for item in seg:
                    s+=sentence[item[0]:item[1]]+' '
                o.write(s+'\n')
        o.close()
        return predict


def DictSegment(s,mode):
    '''
    基于词典的分词算法
    :param s: 输入字符串
    :param mode: 模式参数，可选RMM(逆向最大匹配)，DAG(N-最短路)，RDAG(随机化N-最短路)
    :return: 分词结果(string)
    '''
    if mode=='RMM':
        tool = RMM(dicpath='./Dictionary/CoreNatureDictionary.txt')
        s = s.strip()
        pred = tool.ReverseMaximumMatching(s)
        ans = ''
        for item in pred:
            ans += s[item[0]:item[1]] + ' '
        return ans
    elif mode=='DAG':
        tool = DAG(dicpath='./Dictionary/CoreNatureDictionary.txt')
        s = s.strip()
        pred = tool.DAGsegment(s)
        ans = ''
        for item in pred:
            ans += s[item[0]:item[1]] + ' '
        return ans
    elif mode=='RDAG':
        tool=DAG2(dicpath='./Dictionary/CoreNatureDictionary.txt')
        s=s.strip()
        pred=tool.DAGsegment(s)
        ans = ''
        for item in pred:
            ans += s[item[0]:item[1]] + ' '
        return ans
    elif mode=='HMM':
        Tags, Trans, Emission, Begin=loadpara(mode='Seg')
        s = s.strip()
        state = Viterbi2(s, Tags, Trans, Emission, Begin)
        ans = ''
        for i in range(len(state)):
            if (state[i] == 'S' or state[i] == 'E'):
                ans += s[i] + ' '
            else:
                ans += s[i]
        return ans


def StatisticSegment(s):
    '''
    基于统计的分词算法
    :param s: 输入字符串
    :return: 分词结果
    '''
    Tags, Trans, Emission, Begin = loadpara(mode='Seg')
    s = s.strip()
    state = Viterbi2(s, Tags, Trans, Emission, Begin)
    ans = ''
    for i in range(len(state)):
        if (state[i] == 'S' or state[i] == 'E'):
            ans += s[i] + ' '
        else:
            ans += s[i]
    return ans


def BatchSeg(mode,path,outpath=None):
    '''
    工具包外部调用接口
    对整个文件进行分词
    @param path: 原始文本路径
    @param outpath: 分词结果保存位置
    @param mode: 模式参数 可选HMM,RMM,DAG,SDAG
    @return:
    '''
    # path = './corpus/segment/test5_SEG+POSTAG.txt'
    Tags, Trans, Emission, Begin = loadpara(mode='Seg')
    Predict = []
    Innerdicpath = './Dictionary/CoreNatureDictionary.txt'

    if mode=='HMM':
        since = time.time()
        tst(path, Tags, Trans, Emission, Begin, Predict, isSave=True,savepath=outpath)
        time_elapsed = time.time() - since
        print('Using HMM')
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
        print('segment speed is {} kb/s'.format(519 / time_elapsed))

    elif mode=='DAG':
        dag = DAG(dicpath=Innerdicpath)
        since = time.time()
        pred = dag.tst(path, isSave=True,savepath=outpath)
        time_elapsed = time.time() - since
        print('Using DAG')
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
        print('segment speed is {} kb/s'.format(519 / time_elapsed))

    elif mode=='RDAG':
        dag = DAG2(dicpath=Innerdicpath)
        # rmm=RMM(dic=Dictionary)
        since = time.time()
        pred = dag.tst(path, isSave=True,savepath=outpath)
        time_elapsed = time.time() - since
        print('Using SDAG')
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
        print('segment speed is {} kb/s'.format(519 / time_elapsed))

    elif mode=='RMM':
        print('Using RMM')
        rmm = RMM(dicpath=Innerdicpath)
        since = time.time()
        pred = rmm.tst(path, isSave=True,savepath=outpath)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
        print('segment speed is {} kb/s'.format(519 / time_elapsed))


def DemoSeg():
    '''
    分词算法性能测评
    @return:
    '''
    Tags = ['B', 'M', 'E', 'S']
    Trans = {}
    Emission = {}
    Begin = {}
    TagNum = {}
    Dictionary=set()
    OutDict=set()
    Predict=[]
    Gold=[]

    trainset = open('./corpus/segment/train2.txt', encoding='utf-8')

    #原先是test4
    testsetpath='./corpus/segment/test5_SEG+POSTAG.txt'


    #goldtxt = open('./corpus/gold2.txt', encoding='utf-8')
    goldtxt = open('./corpus/pos/testtxt.txt', encoding='utf-8')
    dicpath = './Dictionary/CoreNatureDictionary.txt'
    Init(Tags,Trans,Emission,Begin,TagNum)
    SentenceNum=train(trainset,Tags,Trans,Emission,Begin,TagNum,Dictionary)
    normalize(Tags, Trans, Emission, Begin, TagNum, SentenceNum)

    savelist = [Tags, Trans, Emission, Begin]
    savename = ['Tags', 'Trans', 'Emission', 'Begin']
    for item,name in zip(savelist,savename):
        with open('./modelparameters/Seg/{}.json'.format(name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False))

    #HMM
    since = time.time()
    tst(testsetpath,Tags,Trans,Emission,Begin,Predict,isSave=True)
    time_elapsed = time.time() - since
    print('Using HMM')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('segment speed is {} kb/s'.format(519/time_elapsed))
    #OutDictNum,InDictNum=getOutDict(Dictionary, OutDict, goldtxt, Gold)
    #此处按texttxt进行了修改
    OutDictNum, InDictNum = getOutDict2(Dictionary, OutDict, goldtxt, Gold)
    Precision,Recall,F1,OOVRecall,IVRecall=SegmentEvaluate(OutDict, Predict, Gold, testsetpath, OutDictNum, InDictNum)
    print('Precision={},Recall={},F1={},OOVRecall={},IVRecall={}'.format(Precision*100,Recall*100,F1*100,OOVRecall*100,IVRecall*100))
    goldtxt.close()


    #DAG
    dag = DAG(dic=Dictionary)
    since = time.time()
    pred = dag.tst(testsetpath,isSave=True)
    time_elapsed = time.time() - since
    print('\nUsing DAG')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('segment speed is {} kb/s'.format(519 / time_elapsed))
    Precision, Recall, F1, OOVRecall, IVRecall = SegmentEvaluate(OutDict, pred, Gold, './corpus/segment/test4.txt',
                                                                 OutDictNum, InDictNum)
    print('Precision={},Recall={},F1={},OOVRecall={},IVRecall={}'.format(Precision * 100, Recall * 100, F1 * 100,OOVRecall * 100, IVRecall * 100))
    goldtxt.close()


    #SDAG
    print('\nUsing RandomDAG')
    dag = DAG2(dic=Dictionary)
    # rmm=RMM(dic=Dictionary)
    since = time.time()
    pred = dag.tst(testsetpath,isSave=True)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('segment speed is {} kb/s'.format(519 / time_elapsed))
    Precision, Recall, F1, OOVRecall, IVRecall = SegmentEvaluate(OutDict, pred, Gold, './corpus/segment/test4.txt',
                                                                 OutDictNum, InDictNum)
    print('Precision={},Recall={},F1={},OOVRecall={},IVRecall={}'.format(Precision * 100, Recall * 100, F1 * 100,OOVRecall * 100, IVRecall * 100))
    goldtxt.close()


    #RMM
    print('\nUsing RMM')
    #rmm=RMM(dicpath=dicpath)
    rmm=RMM(dic=Dictionary)
    since = time.time()
    pred=rmm.tst(testsetpath,isSave=True)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('segment speed is {} kb/s'.format(519/time_elapsed))
    Precision, Recall, F1, OOVRecall, IVRecall = SegmentEvaluate(OutDict, pred, Gold, './corpus/segment/test4.txt',
                                                                 OutDictNum, InDictNum)
    print('Precision={},Recall={},F1={},OOVRecall={},IVRecall={}'.format(Precision * 100, Recall * 100, F1 * 100,OOVRecall * 100, IVRecall * 100))
    goldtxt.close()


if __name__ == '__main__':
    # inpath = './corpus/segment/test5_SEG+POSTAG.txt'
    # outpath = './result/Result.txt'
    # s='美国的华莱士，比你们高到不知道哪里去了，我和他谈笑风生！'

    DemoSeg()
    # BatchSeg(mode='RDAG',path=inpath,outpath=outpath)
    # print(DictSegment(s,mode='DAG'))
    # print(DictSegment(s, mode='RDAG'))
    # print(DictSegment(s, mode='RMM'))
    # print(StatisticSegment(s))

    # 原始文本文件所在路径
    # inpath = './corpus/segment/test5_SEG+POSTAG.txt'
    # outpath = './result/Seg/HMMSegResult.txt'
    # BatchSeg(mode='HMM', path=inpath, outpath=outpath)
    #
    # outpath = './result/Seg/DAGSegResult.txt'
    # BatchSeg(mode='DAG', path=inpath, outpath=outpath)
    #
    # outpath = './result/Seg/RDAGSegResult.txt'
    # BatchSeg(mode='RDAG', path=inpath, outpath=outpath)
    #
    # outpath = './result/Seg/RMMSegResult.txt'
    # BatchSeg(mode='RMM', path=inpath, outpath=outpath)


