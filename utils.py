import json

def loadpara(mode):
    '''
    加载HMM模型参数
    @param mode: 任务类型 分词请选择Seg,词性标注和NER请选择Tag
    @return: Tags,Trans,Emission,Begin
    '''
    if mode=='Tag':
        f = open('modelparameters/Tag/Tags.json', 'r', encoding='utf-8')
        Tags = json.load(f)
        f.close()

        f = open('modelparameters/Tag/Trans.json', 'r', encoding='utf-8')
        Trans = json.load(f)
        f.close()

        f = open('modelparameters/Tag/Emission.json', 'r', encoding='utf-8')
        Emission = json.load(f)
        f.close()

        f = open('modelparameters/Tag/Begin.json', 'r', encoding='utf-8')
        Begin = json.load(f)
        f.close()

    elif mode=='Seg':
        f = open('./modelparameters/Seg/Tags.json', 'r', encoding='utf-8')
        Tags = json.load(f)
        f.close()

        f = open('./modelparameters/Seg/Trans.json', 'r', encoding='utf-8')
        Trans = json.load(f)
        f.close()

        f = open('./modelparameters/Seg/Emission.json', 'r', encoding='utf-8')
        Emission = json.load(f)
        f.close()

        f = open('./modelparameters/Seg/Begin.json', 'r', encoding='utf-8')
        Begin = json.load(f)
        f.close()

    return Tags,Trans,Emission,Begin


