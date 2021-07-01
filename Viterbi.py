def search(p,path,state,t):
    if p=='':
        return
    else:
        search(path[p][t-1],path,state,t-1)
        state.append(p)

'''
def Viterbi(observation,Tags,Trans,Emission,Begin):
    l=len(observation)
    dp=[{} for i in range(2)]
    path={}
    for Tag in Tags:
        path[Tag]=['' for i in range(l)]
    for Tag in Tags:
        dp[0][Tag]=Begin[Tag]+Emission[Tag][observation[0]]
    for i in range(1,l):
        for Tag in Tags:
            dp[1][Tag]=-float('inf')
        for Tag in Tags:
            for preTag in Tags:
                if(observation[i] in Emission[Tag]):
                    temp=dp[0][preTag]+Trans[preTag][Tag]+Emission[Tag][observation[i]]
                else:
                    temp = dp[0][preTag] + Trans[preTag][Tag] - float('inf')
                if(temp>dp[1][Tag]):
                    dp[1][Tag]=temp
                    path[Tag][i]=preTag
        for Tag in Tags:
            dp[0][Tag]=dp[1][Tag]
    #回溯
    temp=-float('inf')
    p=''
    state=[]
    for Tag in Tags:
        if(dp[0][Tag]>temp):
            temp=dp[0][Tag]
            p=Tag
    search(p,path,state,l)
    #state.append(p)
    return state
'''

def CheckOutOfDict(Tags,Emission,word):
    for Tag in Tags:
        if word in Emission[Tag]:
            return False
    return True


def Viterbi(observation,Tags,Trans,Emission,Begin):
    l=len(observation)
    dp=[{} for i in range(2)]
    path={}
    for Tag in Tags:
        path[Tag] = ['' for i in range(l)]
        #dp[0][Tag] = Begin[Tag] + Emission[Tag][observation[0]]
        if(observation[0] in Emission[Tag]):
            dp[0][Tag]=Begin[Tag]+Emission[Tag][observation[0]]
        else:
            dp[0][Tag] = -float('inf')

    for i in range(1,l):
        for Tag in Tags:
            dp[1][Tag] = -float('inf')
        for Tag in Tags:
            path[Tag][i]=Tags[2]
            for preTag in Tags:
                if(observation[i] in Emission[Tag]):
                    temp=dp[0][preTag]+Trans[preTag][Tag]+Emission[Tag][observation[i]]
                else:
                    temp = dp[0][preTag] + Trans[preTag][Tag] - float('inf')
                if(temp>dp[1][Tag]):
                    dp[1][Tag]=temp
                    path[Tag][i]=preTag
        for Tag in Tags:
            dp[0][Tag]=dp[1][Tag]
    #回溯
    temp=-float('inf')
    p=''
    state=[]
    for Tag in Tags:
        if(dp[0][Tag]>=temp):
            temp=dp[0][Tag]
            p=Tag
    search(p,path,state,l)
    return state


def Viterbi2(observation,Tags,Trans,Emission,Begin):
    l=len(observation)
    dp=[{} for i in range(2)]
    path={}
    for Tag in Tags:
        path[Tag] = ['' for i in range(l)]

    isBeginOutDict=CheckOutOfDict(Tags,Emission,observation[0])
    if isBeginOutDict:
        for Tag in Tags:
            dp[0][Tag] = Begin[Tag]
    else:
        for Tag in Tags:
            dp[0][Tag] = Begin[Tag] + Emission[Tag][observation[0]]

    for i in range(1,l):
        for Tag in Tags:
            dp[1][Tag] = -float('inf')
        for Tag in Tags:
            path[Tag][i]=Tags[2]
            for preTag in Tags:
                if(observation[i] in Emission[Tag]):
                    temp=dp[0][preTag]+Trans[preTag][Tag]+Emission[Tag][observation[i]]
                else:
                    temp = dp[0][preTag] + Trans[preTag][Tag] - float('inf')
                #print('从{}转移到{}，发射到{}，概率为{},dp={}'.format(preTag,Tag,observation[i],temp,dp[1][Tag]))
                if(temp>dp[1][Tag]):
                    dp[1][Tag]=temp
                    path[Tag][i]=preTag
            '''
            #如果改词不在词典中
            if(dp[1][Tag]==-float('inf')):
                maxpossible=-float('inf')
                maxpossibletag=''
                for x in Tags:
                    if dp[0][x]+Trans[x][Tag]>maxpossible:
                        maxpossible=dp[0][x]+Trans[x][Tag]
                        maxpossibletag=x
                if(maxpossibletag==''):print('Error,')
                dp[1][Tag]=maxpossible
                path[Tag][i]=maxpossibletag
            '''
        # 如果状态转移结果全部为-inf，则进行修正
        isOutDict=1
        for Tag in Tags:
            if(dp[1][Tag]>-float('inf')):
                isOutDict=0
                break

        if(isOutDict):
            isError = 1
            for Tag in Tags:
                maxpossible = -float('inf')
                maxpossibletag = ''
                for preTag in Tags:
                    if dp[0][preTag]+Trans[preTag][Tag]>maxpossible:
                        maxpossible=dp[0][preTag]+Trans[preTag][Tag]
                        maxpossibletag = preTag
                        isError=0
                dp[1][Tag] = maxpossible
                path[Tag][i] = maxpossibletag
            if isError:
                print('Error!,{}'.format(observation))

        for Tag in Tags:
            dp[0][Tag]=dp[1][Tag]
    #回溯
    temp=-float('inf')
    p=''
    state=[]
    for Tag in Tags:
        if(dp[0][Tag]>=temp):
            temp=dp[0][Tag]
            p=Tag
    search(p,path,state,l)
    return state
