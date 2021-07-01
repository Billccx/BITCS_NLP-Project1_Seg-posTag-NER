import numpy as np
import re

def ChangeTag(T,ChangeRules):
    if T in ChangeRules:
        return ChangeRules[T]
    else:
        return T

def solve(filepath,ChangeRules=None):
    file=open(filepath,'r', encoding='utf-8')
    pattern1 = re.compile(r'\[[^\]]+\]nt')
    pattern2 = re.compile(r'\[[^\]]+\]nz')
    pattern3 = re.compile(r'\[[^\]]+\]ns')
    pattern4 = re.compile(r'\[[^\]]+\]i')
    pattern5 = re.compile(r'\[[^\]]+\]l')
    patterns = [pattern1, pattern2, pattern3, pattern4,pattern5]

    TagSet=[]
    SentenceSet=[]

    for line in file:
        SenTag = [] #该行的所有词语标签
        Sentence=[]
        cmplx = []  #该行中的所有复合词
        line = line.strip()
        if(len(line)==0): continue
        for pattern in patterns:
            cmplx.extend(pattern.findall(line))
        for item in cmplx:
            line = line.replace(item, '#..#')

        line=line.split()
        if line[0][0:3]=='199':
            line=line[1:]
        cnt=0

        for item in line:
            if item=='#..#':
                start = cmplx[cnt].find('[') + 1
                end = cmplx[cnt].find(']')
                subwords = cmplx[cnt][start:end].split()
                for word in subwords:
                    Sentence.append(word.split('/')[0])
                    SenTag.append(ChangeTag(word.split('/')[1],ChangeRules))
                cnt+=1
            else:
                Sentence.append(item.split('/')[0])
                SenTag.append(ChangeTag(item.split('/')[1],ChangeRules))
            if(len(SenTag)!=len(Sentence)):
                print(Sentence)
        TagSet.append(SenTag)
        SentenceSet.append(Sentence)

    Tags = set()
    for item in TagSet:
        Tags.update(item)
    return TagSet,Tags,SentenceSet

def changeFormat(filepath,newfilepath):
    file = open(filepath, 'r', encoding='utf-8')
    nfile = open(newfilepath, 'w', encoding='utf-8')
    for line in file:
        line=line.strip()
        s=line.replace('。/w','。/w\n')
        if(len(s)>0 and s[-1]!='\n'): s+='\n'
        nfile.write(s)
    file.close()
    nfile.close()



def main():
    filepath='./corpus/POS2.txt'
    filepath2='./corpus/POS2new.txt'
    traintxt=open('./corpus/traintxt.txt','w', encoding='utf-8')
    traintag = open('./corpus/traintag.txt', 'w', encoding='utf-8')

    testtxt = open('./corpus/testtxt.txt', 'w', encoding='utf-8')
    testtag = open('./corpus/testtag.txt', 'w', encoding='utf-8')
    crftrain=open('./corpus/CRFIO/crftrain.txt', 'w', encoding='utf-8')
    crftest=open('./corpus/CRFIO/crftest.txt', 'w', encoding='utf-8')

    ChangeRules={'Bg':'b','Mg':'m','Ag':'a','Dg':'d','Ng':'n','Tg':'t','Rg':'r','Vg':'v','Yg':'y'}
    changeFormat(filepath,filepath2)
    TagSet,Tags,SentenceSet=solve(filepath2,ChangeRules)
    for s,t in zip(SentenceSet[:-3000],TagSet[:-3000]):
        for word,tag in zip(s,t):
            traintxt.write(str(word)+' ')
            traintag.write(str(tag) +' ')
            crftrain.write(str(word)+' '+str(tag)+'\n')
        traintag.write('\n')
        traintxt.write('\n')
        crftrain.write('\n')
    traintag.close()
    traintxt.close()
    crftrain.close()

    for s,t in zip(SentenceSet[-3000:],TagSet[-3000:]):
        for word,tag in zip(s,t):
            testtxt.write(str(word)+' ')
            testtag.write(str(tag) +' ')
            crftest.write(str(word) + ' ' + str(tag) + '\n')
        testtag.write('\n')
        testtxt.write('\n')
        crftest.write('\n')
    testtag.close()
    testtxt.close()
    crftest.close()

    print(Tags)

if __name__ == '__main__':
    main()




