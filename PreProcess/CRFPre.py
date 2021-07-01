def PreProcess(path,outpath):
    file=open(path,'r',encoding='utf-8')
    out = open(outpath, 'w', encoding='utf-8')
    pre=''
    for line in file:
        if line=='\n':
            pre=''
            out.write('\n')
            continue
        else:
            s=''
            line=line.strip()
            line=line.split()
            s+=line[0]+' '
            if line[1]=='nt' or line[1]=='nr' or line[1]=='ns':
                if line[1]==pre:
                    s+=line[1]+'-I'
                else:
                    s+=line[1]+'-B'
            else:
                s+='O'
            pre=line[1]
            out.write(s+'\n')

if __name__ == '__main__':
    path='../corpus/CRFIO/crftrain.txt'
    out='../corpus/CRFIO/CRFNER/crfNERtrain.txt'
    PreProcess(path,out)
