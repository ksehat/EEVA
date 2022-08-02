def CommonStr(str1,str2):
    m = len(str1)
    n = len(str2)
    count = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    cs = []
    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                c = count[i][j] + 1
                count[i+1][j+1] = c
                if c > longest:
                    cs = []
                    longest = c
                    cs.append(str1[i-c+1:i+1])
                elif c == longest:
                    cs.append(str1[i-c+1:i+1])

    return cs


result = CommonStr('str1', 'str2')
for i in result:
    print (i)