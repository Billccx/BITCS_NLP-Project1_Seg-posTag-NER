def solve(anspath,txtpath,gtpath,gtntpath,gtnrpath,gtnspath):
    ans=open(anspath,'r',encoding='utf-8')
    txt=open(txtpath,'r',encoding='utf-8')
    gold=open(gtpath,'w',encoding='utf-8')
    goldnt = open(gtntpath, 'w', encoding='utf-8')
    goldnr = open(gtnrpath, 'w', encoding='utf-8')
    goldns = open(gtnspath, 'w', encoding='utf-8')
    for line1,line2 in zip(ans,txt):
        flag=0
        fnt=0
        fns=0
        fnr=0
        state=''
        s=''
        snt=''
        snr=''
        sns=''
        line1=line1.strip()
        line1=line1.split()
        #nt
        state='nt'
        l=len(line1)
        i=0
        while(i<l):
            if line1[i]==state:
                flag=1
                fnt=1
                start=i
                while(i<l and line1[i]==state):
                    i+=1
                end=i
                # s+='({},{})'.format(start,end)+'#'+line2[start:end]+'#'+state+' '
                # snt+= '({},{})'.format(start, end) + '#' + line2[start:end] + '#' + state + ' '

                s += '({},{})'.format(start, end) +' '
                snt += '({},{})'.format(start, end)+' '

            else: i+=1

        state = 'nr'
        l = len(line1)
        i = 0
        while (i < l):
            if line1[i] == state:
                flag = 1
                fnr=1
                start = i
                while (i<l and line1[i] == state):
                    i += 1
                end = i
                # s += '({},{})'.format(start, end) + '#' + line2[start:end] + '#' + state + ' '
                # snr += '({},{})'.format(start, end) + '#' + line2[start:end] + '#' + state + ' '
                s += '({},{})'.format(start, end) + ' '
                snr += '({},{})'.format(start, end) + ' '
            else:
                i += 1

        state = 'ns'
        l = len(line1)
        i = 0
        while (i < l):
            if line1[i] == state:
                flag = 1
                fns=1
                start = i
                while (i<l and line1[i] == state):
                    i += 1
                end = i
                # s += '({},{})'.format(start, end) + '#' + line2[start:end] + '#' + state + ' '
                # sns+= '({},{})'.format(start, end) + '#' + line2[start:end] + '#' + state + ' '
                s += '({},{})'.format(start, end) + ' '
                sns += '({},{})'.format(start, end) + ' '
            else:
                i += 1

        if flag==0:
            s+='!!!'
        if fnt==0:
            snt += '!!!'
        if fnr==0:
            snr += '!!!'
        if fns==0:
            sns += '!!!'
        s+='\n'
        snt+='\n'
        snr += '\n'
        sns += '\n'
        gold.write(s)
        goldnt.write(snt)
        goldnr.write(snr)
        goldns.write(sns)