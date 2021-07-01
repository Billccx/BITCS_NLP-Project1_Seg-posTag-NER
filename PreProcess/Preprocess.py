# trainset2 = open('./corpus/train2.txt','w', encoding='utf-8')
# with open('./corpus/train.txt','r',encoding='gbk') as f:
#     for line in f:
#         s = line.replace('。', '。\n')
#         trainset2.write(s)


# if __name__ == '__main__':
#     trainset = open('./corpus/train.txt', encoding='utf-8')
#     testset = open('./corpus/test2.txt', encoding='utf-8')


def getGoldpre(path):
    testcorpus=open(path, encoding='utf-8')
    gold = open('./corpus/gold.txt','w', encoding='utf-8')
    for line in testcorpus:
        s=line.replace('。', '。\n')
        # if(s[-1]==s[-2] and s[-1]=='\n'):
        #     s=s[:-1]
        gold.write(s)

# getGold('./corpus/test.txt')


def cleanspaceline(path,targetpath):
    file=open(path, encoding='utf-8')
    target=open(targetpath,'w', encoding='utf-8')
    for line in file:
        line=line.strip(' ')
        if(line=='\n'):
            continue
        else:
            target.write(line)

cleanspaceline('./corpus/gold.txt','./corpus/gold2.txt')
cleanspaceline('./corpus/test2.txt','./corpus/test4.txt')