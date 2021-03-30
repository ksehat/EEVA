from data_prep.data_hunter import DataHunter
import pandas as pd



class XABCHunter():

    def __init__(self,data,date_pointer):
        self.data = data
        self.date_pointer = date_pointer

    def macd_phase_change(self):
        if df['MACD_ZC1'][self.date_pointer]==1: return True
        else: return False

    def find(self):
        df = self.data[:self.date_pointer]
        ZC_Index = pd.DataFrame({'zcindex': df[df['MACD1_ZC'] == 1].index.values,
                                 'timestamp': df.loc[df['MACD1_ZC'] == 1, 'timestamp'],
                                 'MACD1_Hist': df.loc[df['MACD1_ZC'] == 1, 'MACD1_Hist']},
                                columns=['zcindex', 'timestamp', 'MACD1_Hist']).reset_index(drop=True)
        XABC_list = []
        for row_zcindex, zcindex in ZC_Index.iterrows():
            if row_zcindex + 4 <= len(ZC_Index) - 1:
                if df['MACD1_Hist'][zcindex[0]] >= 0:
                    # region XABC Finder
                    X = max(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'])
                    index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'].idxmax()
                    A = min(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'])
                    index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'low'].idxmin()
                    B = max(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'])
                    index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'high'].idxmax()
                    C = min( df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'] )
                    index_C = df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'].idxmin()
                    if A < X and B < X and B > A and C < A:
                        xabc_flag = 1
                        index_4 = ZC_Index.iloc[row_zcindex + 4, 0]
                        XABC_list.append([[X, A, B, C], [index_X, index_A, index_B, index_C, index_4], xabc_flag])
                    # endregion
                if df['MACD1_Hist'][zcindex[0]] < 0:
                    # region XABC Finder
                    X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                    index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'].idxmin()
                    A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'])
                    index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'high'].idxmax()
                    B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'])
                    index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'low'].idxmin()
                    C = max(df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'])
                    index_C = df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]][
                        'high'].idxmax()
                    if A > X and B > X and B < A and C > A:
                        xabc_flag = 0
                        index_4 = ZC_Index.iloc[row_zcindex + 4, 0]
                        # stop_loss = C
                        # sudo_stop_loss = C
                        XABC_list.append([[X, A, B, C], [index_X, index_A, index_B, index_C, index_4], xabc_flag])
                    # endregion
        return XABC_list


data = DataHunter(symbol='LTCUSDT',start_date='1 Jan 2021', end_date='2021-01-05 00:00:00', step='1h').prepare_data()
for date_pointer in range(len(data)):
    a = XABCHunter(data,date_pointer).find()
    if date_pointer>=a
    print(date_pointer)
    print(a)