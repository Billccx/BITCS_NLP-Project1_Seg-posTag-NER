import numpy as np
import json
from Viterbi import *
from utils import *
from eval import *
import CRFPP

def HMM_NER(path, Tags, Trans, Emission, Begin):
    '''
    用于对整个文件进行NER
    @param path: 预分词的文件路径
    @param Tags: 词性标签列表
    @param Trans: 转移矩阵
    @param Emission: 发射矩阵
    @param Begin: 初始状态向量
    @return: (词性)标注结果 list(list)
    '''
    testtxt = open(path,'r', encoding='utf-8')
    pred2 = []
    for line in testtxt:
        line = line.strip()
        line = line.split()
        state = Viterbi2(line, Tags, Trans, Emission, Begin)
        state2 = []
        for index, item in enumerate(state):
            state2.extend([item] * len(line[index]))
        pred2.append(state2)
    return pred2


def DemoHMMNER():
    '''
    HMM NER性能测评
    :return:
    '''
    #加载模型参数
    Tags, Trans, Emission, Begin=loadpara(mode='Tag')
    #testtagbycha = './corpus/posbycha/testtagbycha.txt'

    #预分词的文本文件路径
    testtxtpath = './corpus/segment/HMMsegment.txt'

    # NER正确答案文件路径
    gtpath = './corpus/NERdata/gold/NERgold.txt'
    gtntpath = './corpus/NERdata/gold/NERgoldnt.txt'
    gtnrpath = './corpus/NERdata/gold/NERgoldnr.txt'
    gtnspath = './corpus/NERdata/gold/NERgoldns.txt'

    #NER
    pred = HMM_NER(testtxtpath, Tags, Trans, Emission, Begin)

    ntprecision,ntrecall,ntF1,nrprecision,nrrecall,nrF1,nsprecision,nsrecall,nsF1=NEREvaluate(gtpath,gtntpath,gtnrpath,gtnspath,pred)

    print('对于人名实体的识别准确率为:{}% ,召回率为{}% ,F1-measure={}% '.format(nrprecision*100,nrrecall*100,nrF1*100))
    print('对于地名实体的识别准确率为:{} ,召回率为{} ,F1-measure={}% '.format(nsprecision*100,nsrecall*100,nsF1*100))
    print('对于机构名实体的识别准确率为:{} ,召回率为{} ,F1-measure={}% '.format(ntprecision*100,ntrecall*100,ntF1*100))


def NER(s,mode):
    '''
    工具包的NER单句调用接口
    @param s: 预分词字符串
    @return:
    '''
    if mode=='HMM':
        line=s.replace(' ','')
        Tags, Trans, Emission, Begin = loadpara(mode='Tag')
        s=s.strip()
        s=s.split()
        state = Viterbi2(s, Tags, Trans, Emission, Begin)
        state2 = []
        for index, item in enumerate(state):
            state2.extend([item] * len(s[index]))

    elif mode=='CRF':
        line = s.replace(' ', '')
        crf_model = './modelparameters/CRF/NERmodel3'
        tagger = CRFPP.Tagger("-m " + crf_model)
        state2 = []
        tagger.clear()
        s = s.strip()
        s = s.split()

        for word in s:
            tagger.add(word)
        tagger.parse()

        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                word = tagger.x(i, j)
                tag = tagger.y2(i)
                state2 += [tag] * len(word)

        for i in range(len(state2)):
            if state2[i] == 'nt-B' or state2[i] == 'nt-I':
                state2[i] = 'nt'
            elif state2[i] == 'ns-B' or state2[i] == 'ns-I':
                state2[i] = 'ns'
            elif state2[i] == 'nr-B' or state2[i] == 'nr-I':
                state2[i] = 'nr'

    ntpred = set(getEntity('nt', state2))
    nspred = set(getEntity('ns', state2))
    nrpred = set(getEntity('nr', state2))
    print('\n人名: ',end='')
    for item in nrpred:
        print(line[item[0]:item[1]],end='  ')
    print('\n地名: ',end='')
    for item in nspred:
        print(line[item[0]:item[1]],end='  ')
    print('\n机构名: ',end='')
    for item in ntpred:
        print(line[item[0]:item[1]],end='  ')


def CRFNER(path,outpath=None):
    txt = open(path, 'r', encoding='utf-8')
    crf_model = './modelparameters/CRF/NERmodel3'
    tagger = CRFPP.Tagger("-m " + crf_model)
    pred=[]
    for line in txt:
        ans=[]
        tagger.clear()
        line = line.strip()
        line = line.split()

        for word in line:
            tagger.add(word)
        tagger.parse()


        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                word = tagger.x(i, j)
                tag = tagger.y2(i)
                ans+=[tag]*len(word)

        for i in range(len(ans)):
            if ans[i]=='nt-B' or ans[i]=='nt-I':
                ans[i]='nt'
            elif ans[i]=='ns-B' or ans[i]=='ns-I':
                ans[i]='ns'
            elif ans[i]=='nr-B' or ans[i]=='nr-I':
                ans[i]='nr'
        pred.append(ans)
    txt.close()
    return pred


def DemoCRFNER():
    '''
    CRF NER性能测评
    :return:
    '''
    # 预分词的文本文件路径
    path = './corpus/segment/HMMsegment.txt'

    # NER正确答案文件路径
    gtpath = './corpus/NERdata/gold/NERgold.txt'
    gtntpath = './corpus/NERdata/gold/NERgoldnt.txt'
    gtnrpath = './corpus/NERdata/gold/NERgoldnr.txt'
    gtnspath = './corpus/NERdata/gold/NERgoldns.txt'

    pred = CRFNER(path=path)

    ntprecision, ntrecall, ntF1, nrprecision, nrrecall, nrF1, nsprecision, nsrecall, nsF1 = NEREvaluate(gtpath,
                                                                                                        gtntpath,
                                                                                                        gtnrpath,
                                                                                                        gtnspath, pred)

    print('对于人名实体的识别准确率为:{}% ,召回率为{}% ,F1-measure={}% '.format(nrprecision * 100, nrrecall * 100, nrF1 * 100))
    print('对于地名实体的识别准确率为:{} ,召回率为{} ,F1-measure={}% '.format(nsprecision * 100, nsrecall * 100, nsF1 * 100))
    print('对于机构名实体的识别准确率为:{} ,召回率为{} ,F1-measure={}% '.format(ntprecision * 100, ntrecall * 100, ntF1 * 100))


def BatchNER(mode,path,outpath=None):
    '''
    对整个文件进行命名实体识别
    :param mode: 模式参数 可选CRF,HMM
    :param path: 输入的预分词文件所在路径
    :param outpath: 输出的NER结果保存路径
    :return:
    '''
    txt=open(path,'r',encoding='utf-8')
    outfile=open(outpath,'w',encoding='utf-8')
    if mode=='HMM':
        cnt=0
        Tags, Trans, Emission, Begin = loadpara(mode='Tag')
        for line in txt:
            #print(cnt)
            line = line.strip()
            s = line
            s=s.replace(' ','')
            line = line.split()
            state = Viterbi2(line, Tags, Trans, Emission, Begin)
            state2 = []
            res=''
            for index, item in enumerate(state):
                state2.extend([item] * len(line[index]))
            l=len(state2)
            i=0
            while(i<l):
                if state2[i]=='nt' or state2[i]=='nr' or state2[i]=='ns':
                    res+='[ '+s[i]
                    j=i+1
                    while j<l and state2[j]==state2[i]:
                        res += s[j]
                        j+=1
                    res+=' ]'+state2[i]+' '
                    i = j
                else:
                    res+=s[i]
                    i+=1
            res+='\n'
            outfile.write(res)
            cnt+=1

    elif mode=='CRF':
        crf_model = './modelparameters/CRF/NERmodel3'
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
            words=[]
            tags=[]
            for i in range(0, size):
                for j in range(0, xsize):
                    word = tagger.x(i, j)
                    tag = tagger.y2(i)
                    words.append(word)
                    tags.append(tag)

            l=len(words)
            for i in range(l):
                if tags[i] == 'nt-B' or tags[i] == 'nr-B' or tags[i] == 'ns-B':
                    s += '[ ' + words[i] + ' '
                    j = i + 1
                    while j < l and tags[j] == (tags[i][0:2]+'-I'):
                        s += words[j]+' '
                        j += 1
                    s += ']' + tags[i][0:2] + ' '
                    i = j
                else:
                    s += words[i]+' '
                    i += 1
            outfile.write(s+'\n')
        txt.close()
        outfile.close()

if __name__ == '__main__':
    # path = './corpus/segment/RMMsegment.txt'
    #
    # out = './result/crfNERout.txt'
    # BatchNER(mode='CRF', path=path, outpath=out)
    #
    # out = './result/crfHMMout.txt'
    # BatchNER(mode='HMM', path=path, outpath=out)
    s = '联合国 总部 在 美国 纽约 ， 在 瑞士 日内瓦 、 奥地利 维也纳 、 肯尼亚 内罗毕 、 泰国 曼谷 、 埃塞俄比亚 亚的斯亚贝巴 、 黎巴嫩 贝鲁特 、 智利 圣地亚哥 设 有 办事处 ， 首席 行政 长官 是 联合国 秘书长 ， 现 由 安东尼 古特雷斯 担任 。'
    NER(s, mode='CRF')
    NER(s,mode='HMM')

    # print('Using CRF')
    # DemoCRFNER()
    #
    # print('Using HMM')
    # DemoHMMNER()

    # s = '美国 的 华莱士 ， 联合国 你们 高 到 不 知道 哪里 去 了 ， 我 和 他 谈笑风生 ！'
    # NER(s,mode='HMM')

