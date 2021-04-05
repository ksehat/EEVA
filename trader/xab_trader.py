# xab = [
#       [X,A,B,C],
#       [index_X,index_A,index_B,index_C,index_4],
#       flag,
#       [warning, index_warning],
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
        C = self.xab[0][3] if xab[0][3] else None
        index_X = self.xab[1][0]
        index_A = self.xab[1][1]
        index_B = self.xab[1][2]
        index_C = self.xab[1][3] if self.xab[1][3] else None
        index_4 = self.xab[1][4]
        flag = self.xab[2]
        warning = self.xab[3][0]
        enter = self.xab[4][0]
        if enter==0:
            if flag==1 and df['low'][self.dp]<=C: # C is selected but when the enter is True, C or stop loss should be selected.
                self.xab[0][3] = df['low'][self.dp]
                self.xab[1][3] = self.dp
            if flag==0 and df['high'][self.dp]>=C:
                self.xab[0][3] = df['high'][self.dp]
                self.xab[1][3] = self.dp
            if (flag == 0 and C >= A) or (
                    flag == 1 and C <= A) and warning == 0:
                self.xab[3][0] = 1 #warning=1
                self.xab[3][1] = self.dp
            if (flag == 0 and df['low'][self.dp] <= B) or (
                    flag == 1 and df['high'][self.dp] >= B) and warning == 1: # TODO: check it if you want to enter at the same candle as C or not
                self.xab[4][0] = 1 #enter=1
                self.xab[4][1] = self.dp

        if enter==1:






        else:


    def finance(self):


    def