# xab = [
#       [X,A,B,C],
#       [index_X,index_A,index_B,index_C,index_4],
#       flag,
#       [enter, index_enter]
#       ]

class XABTrader():

    def __init__(self, df, xab: list, date_pointer: str):
        self.df = df
        self.dp = date_pointer
        self.xab = xab

        # self.stop_loss = xab[3] if self.C >= 4 else C
        # self.flag_sudo_stop_loss = xab[4] if len(xab) >= 5 else 0


    def trade(self):
        df = self.df
        X = self.xab[0][0]
        A = self.xab[0][1]
        B = self.xab[0][2]
        index_X = self.xab[1][0]
        index_A = self.xab[1][1]
        index_B = self.xab[1][2]
        index_4 = self.xab[1][4]
        flag = self.xab[2]
        enter = self.xab[3][0]
        if enter==0:
            if (flag==1 and df['low'][self.dp]<=A): # C is selected but when the enter is True, C or stop loss should be selected.
                self.xab[0][3] = df['low'][self.dp]
                self.xab[1][3] = self.dp
            if flag==0 and df['high'][self.dp]>=A:
                self.xab[0][3] = df['high'][self.dp]
                self.xab[1][3] = self.dp
            if (flag == 0 and df['low'][self.dp] <= B) or (
                    flag == 1 and df['high'][self.dp] >= B) and self.xab[0][3]: # TODO: check it if you want to enter at the same candle as C or not
                self.xab[3][0] = 1 #enter=1
                self.xab[3][1] = self.dp
        if enter==1:







        else:


    def finance(self):


    def