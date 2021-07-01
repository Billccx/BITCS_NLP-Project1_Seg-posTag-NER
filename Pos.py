import numpy as np
import json
import time
from Viterbi import *
from utils import *
from eval import *
import CRFPP


def Init(Tags,Trans,Emission,Begin,TagNum):
    """
    初始化HMM的各参数矩阵
    @param Tags:(list) 列举所有的词性标签
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始状态向量
    @param TagNum: 各词性标签的出现次数
    """
    for Tagi in Tags:
        temp={}
        for Tagj in Tags:
            temp[Tagj]=0.
        Trans[Tagi]=temp
        Emission[Tagi]={}
        Begin[Tagi]=0
        TagNum[Tagi]=0


def train(traintxt,traintag,Tags,Trans,Emission,Begin,TagNum):
    """
    训练HMM
    @param traintxt: 训练集文本
    @param traintag:  训练集词性标签
    @param Tags:词性标签列表
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始状态向量
    @param TagNum: 各词性标签数量
    @return: 训练集句子数量
    """
    cnt=0
    for words,tags in zip(traintxt,traintag):
        cnt+=1
        tags=tags.strip()
        sTag=tags.split()
        words=words.strip()
        words=words.split()
        Begin[sTag[0]]+=1.
        for i in range(len(sTag)-1):
            Trans[sTag[i]][sTag[i+1]]+=1.

        for index,word in enumerate(words):
            TagNum[sTag[index]] += 1.
            for Tag in Tags:
                if(word not in Emission[Tag]):
                    Emission[Tag][word]=0.
                if(sTag[index]==Tag):
                    Emission[Tag][word]+=1.
    return cnt


def normalize(Tags,Trans,Emission,Begin,TagNum,SentenceNum):
    """
    对HMM各参数矩阵进行正则化
    @param Tags: 词性标签列表
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始状态向量
    @param TagNum: 各标签数量
    @param SentenceNum: 训练集句子数量
    """
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
        for word in Emission[i]:
            if(Emission[i][word]):
                Emission[i][word]=np.log(Emission[i][word]/TagNum[i])
            else:
                Emission[i][word]=-float('inf')


def tst(testtxt,Tags,Trans,Emission,Begin):
    '''
    返回按词标注的pred,无法进行评估，但速度更快，用于计算分词效率
    @param testtxt: 测试集文本路径
    @param Tags: 标签列表
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始向量矩阵
    @return: 预测结果 list(list)
    '''
    pred=[]
    f=open('./corpus/pos/POSresult.txt','w',encoding='utf-8')
    for line in testtxt:
        line=line.strip()
        line=line.split()
        state=Viterbi2(line,Tags,Trans,Emission,Begin)
        pred.append(state)
        s=''
        for i in range(len(state)):
            s+=str(line[i])+'/'+str(state[i])+' '
        f.write(s+'\n')
    f.close()
    return pred


def tst2(testtxt,Tags,Trans,Emission,Begin):
    """
    返回按字标注的pred,用于对自己的分词结果准确性进行评估
    @param testtxt: 测试集文本路径
    @param Tags: 标签列表
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始向量矩阵
    @return: 预测结果 list(list)
    """
    pred2=[]
    f=open('./corpus/pos/POSresult.txt','w',encoding='utf-8')
    for line in testtxt:
        line=line.strip()
        line=line.split()
        state=Viterbi2(line,Tags,Trans,Emission,Begin)
        state2=[]
        for index,item in enumerate(state):
            state2.extend([item]*len(line[index]))
        pred2.append(state2)
        s=''
        for i in range(len(state)):
            s+=str(line[i])+'/'+str(state[i])+' '
        f.write(s+'\n')
    f.close()
    return pred2


def Tagging(s,mode):
    """
    外部工具包调用接口
    @param s: 需要进行词性标注的预分词字符串
    @param mode: 模式参数 可选HMM,CRF
    """
    if mode=='HMM':
        Tags, Trans, Emission, Begin=loadpara(mode='Tag')
        s = s.strip()
        s = s.split()
        state = Viterbi2(s, Tags, Trans, Emission, Begin)
        res=''
        for i in range(len(state)):
            res += str(s[i]) + '/' + str(state[i]) + ' '
        return res

    elif mode=='CRF':
        crf_model = './modelparameters/CRF/model'
        tagger = CRFPP.Tagger("-m " + crf_model)
        s = s.strip()
        s = s.split()
        for word in s:
            tagger.add(word)
        tagger.parse()
        res=''
        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                word = tagger.x(i, j)
                tag = tagger.y2(i)
                res+=word+'/'+tag+' '
        return res


def PosTagDemo():
    '''
    词性标注算法性能分析
    @return:
    '''
    Tags=['f', 'z', 'n', 'o', 'q', 'c', 'ad', 'nx', 'vd', 'u', 'l', 'i', 'p', 'nz', 'ns', 'b', 'w', 'nr', 'r', 'y', 'h', 'an', 'd', 's', 't', 'v', 'k', 'j', 'm', 'e', 'nt', 'a', 'vn']
    Trans = {}
    Emission = {}
    Begin = {}
    TagNum = {}

    #训练集路径
    traintxt=open('./corpus/pos/traintxt.txt', encoding='utf-8')
    traintag=open('./corpus/pos/traintag.txt', encoding='utf-8')

    #正确的分词结果
    #testtxt=open('./corpus/testtxt.txt', encoding='utf-8')
    #自己的分词结果
    testtxt = open('./corpus/segment/RMMsegment.txt', encoding='utf-8')

    #测试集标签
    testtag='./corpus/pos/testtag.txt'
    testtagbycha='./corpus/posbycha/testtagbycha.txt'


    #CRF词性标注
    crf_model = './modelparameters/CRF/model'
    input_file = './corpus/pos/testtxt.txt'
    output_file = './corpus/CRFIO/new.txt'
    testtagbycha = './corpus/posbycha/testtagbycha.txt'
    tagger = CRFPP.Tagger("-m " + crf_model)

    since = time.time()
    pred = crf_segmenter(input_file, output_file, tagger)
    time_elapsed = time.time() - since
    print('CRFPosTag complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('CRFPosTag speed is {} kb/s'.format(626 / time_elapsed))

    PosTagEvaluate(Tags, pred, testtagbycha)


    #HMM词性标注
    #训练HMM
    Init(Tags, Trans, Emission, Begin, TagNum)
    SentenceNum=train(traintxt, traintag, Tags, Trans, Emission, Begin, TagNum)
    normalize(Tags, Trans, Emission, Begin, TagNum, SentenceNum)

    #保存HMM参数
    savelist=[Tags, Trans, Emission, Begin]
    savename=['Tags', 'Trans', 'Emission', 'Begin']
    for item,name in zip(savelist,savename):
        with open('./modelparameters/Tag/{}.json'.format(name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False))

    since = time.time()
    #pred=tst(testtxt, Tags, Trans, Emission, Begin)
    pred=tst2(testtxt, Tags, Trans, Emission, Begin)
    time_elapsed = time.time() - since
    print('HMMPosTag complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('HMMPosTag speed is {} kb/s'.format(626 / time_elapsed))

    PosTagEvaluate(Tags,pred,testtagbycha)


def DemoPosTag():
    Tags=['f', 'z', 'n', 'o', 'q', 'c', 'ad', 'nx', 'vd', 'u', 'l', 'i', 'p', 'nz', 'ns', 'b', 'w', 'nr', 'r', 'y', 'h', 'an', 'd', 's', 't', 'v', 'k', 'j', 'm', 'e', 'nt', 'a', 'vn']
    Trans = {}
    Emission = {}
    Begin = {}
    TagNum = {}

    #训练集文本路径
    traintxt=open('./corpus/pos/traintxt.txt', encoding='utf-8')

    #训练集标签路径
    traintag=open('./corpus/pos/traintag.txt', encoding='utf-8')

    #使用正确的预分词结果
    #testtxt=open('./corpus/testtxt.txt', encoding='utf-8')

    #使用自己的预分词结果
    testtxt = open('./corpus/segment/RMMsegment.txt', encoding='utf-8')

    #按词标注的词性ground truth
    #testtag='./corpus/pos/testtag.txt'

    #按字标注的词性ground truth
    testtagbycha='./corpus/posbycha/testtagbycha.txt'

    Init(Tags, Trans, Emission, Begin, TagNum)
    SentenceNum=train(traintxt, traintag, Tags, Trans, Emission, Begin, TagNum)
    normalize(Tags, Trans, Emission, Begin, TagNum, SentenceNum)

    #存储训练完成的模型参数
    savelist=[Tags, Trans, Emission, Begin]
    savename=['Tags', 'Trans', 'Emission', 'Begin']
    for item,name in zip(savelist,savename):
        with open('./modelparameters/Tag/{}.json'.format(name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False))


    since = time.time()
    #pred=tst(testtxt, Tags, Trans, Emission, Begin)
    pred=tst2(testtxt, Tags, Trans, Emission, Begin)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('PosTag speed is {} kb/s'.format(436.95 / time_elapsed))
    PosTagEvaluate(Tags,pred,testtagbycha)


def main():
    Tags=['f', 'z', 'n', 'o', 'q', 'c', 'ad', 'nx', 'vd', 'u', 'l', 'i', 'p', 'nz', 'ns', 'b', 'w', 'nr', 'r', 'y', 'h', 'an', 'd', 's', 't', 'v', 'k', 'j', 'm', 'e', 'nt', 'a', 'vn']
    Trans = {}
    Emission = {}
    Begin = {}
    TagNum = {}

    #训练集文本路径
    traintxt=open('./corpus/pos/traintxt.txt', encoding='utf-8')

    #训练集标签路径
    traintag=open('./corpus/pos/traintag.txt', encoding='utf-8')

    #使用正确的预分词结果
    #testtxt=open('./corpus/testtxt.txt', encoding='utf-8')

    #使用自己的预分词结果
    testtxt = open('./corpus/segment/RMMsegment.txt', encoding='utf-8')

    #按词标注的词性ground truth
    #testtag='./corpus/pos/testtag.txt'

    #按字标注的词性ground truth
    testtagbycha='./corpus/posbycha/testtagbycha.txt'

    Init(Tags, Trans, Emission, Begin, TagNum)
    SentenceNum=train(traintxt, traintag, Tags, Trans, Emission, Begin, TagNum)
    normalize(Tags, Trans, Emission, Begin, TagNum, SentenceNum)

    #存储训练完成的模型参数
    savelist=[Tags, Trans, Emission, Begin]
    savename=['Tags', 'Trans', 'Emission', 'Begin']
    for item,name in zip(savelist,savename):
        with open('./modelparameters/Tag/{}.json'.format(name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False))


    since = time.time()
    #pred=tst(testtxt, Tags, Trans, Emission, Begin)
    pred=tst2(testtxt, Tags, Trans, Emission, Begin)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    print('PosTag speed is {} kb/s'.format(436.95 / time_elapsed))
    PosTagEvaluate(Tags,pred,testtagbycha)


def BatchTagging(mode,path,outpath=None):
    '''
    工具包外部调用接口
    对整个预分词文件进行标注
    @param mode: 模式参数 可选HMM,CRF
    @param path: 预分词文件路径
    @param outpath: 标注结果保存位置
    @return:
    '''
    if mode=='HMM':
        Tags, Trans, Emission, Begin = loadpara(mode='Tag')
        txt=open(path,'r',encoding='utf-8')
        outfile=open(outpath,'w',encoding='utf-8')
        for line in txt:
            line = line.strip()
            line = line.split()
            state = Viterbi2(line, Tags, Trans, Emission, Begin)
            s = ''
            for i in range(len(state)):
                s += str(line[i]) + '/' + str(state[i]) + ' '
            outfile.write(s + '\n')
        outfile.close()

    elif mode=='CRF':
        txt = open(path, 'r', encoding='utf-8')
        outfile=open(outpath, 'w', encoding='utf-8')
        crf_model = './modelparameters/CRF/model'
        tagger = CRFPP.Tagger("-m " + crf_model)
        for line in txt:
            tagger.clear()
            line = line.strip()
            line = line.split()

            for word in line:
                tagger.add(word)
            tagger.parse()

            s=''
            size = tagger.size()
            xsize = tagger.xsize()
            for i in range(0, size):
                for j in range(0, xsize):
                    word = tagger.x(i, j)
                    tag = tagger.y2(i)
                    s+=word+'/'+tag+' '
            outfile.write(s+'\n')
        txt.close()
        outfile.close()


def crf_segmenter(input_file, output_file, tagger):
    input_data = open(input_file, 'r', encoding='utf-8')
    output_data = open(output_file, 'w', encoding='utf-8')
    pred=[]
    for line in input_data.readlines():
        ans=[]
        tagger.clear()
        line=line.strip()
        line=line.split()

        for word in line:
            tagger.add(word)
        tagger.parse()

        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                char = tagger.x(i, j)
                tag = tagger.y2(i)
                #output_data.write(char+'/'+tag+' ')
                output_data.write(tag + ' ')
                ans+=[tag for _ in range(len(line[i]))]
        output_data.write('\n')
        pred.append(ans)
    input_data.close()
    output_data.close()
    return pred


if __name__ == '__main__':
    PosTagDemo()
    # s = '美国 的 华莱士 ， 比 你们 高 到 不 知道 哪里 去 了 ， 我 和 他 谈笑风生 ！'
    # print(Tagging(s,mode='HMM'))
    # inpath = './corpus/segment/RMMSegment.txt'
    # outpath = './result/RMMResult.txt'
    # BatchTagging(mode='CRF',path=inpath,outpath=outpath)