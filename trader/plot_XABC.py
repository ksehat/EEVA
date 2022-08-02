import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_figure(df, xab, index_buy, index_sell, enter_price, exit_price):
    index_X = xab[1][0]
    index_A = xab[1][1]
    index_B = xab[1][2]
    index_C = xab[1][3]
    X = xab[0][0]
    A = xab[0][1]
    B = xab[0][2]
    C = xab[0][3]
    width = 1500
    height = 1000
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05)
    fig.add_trace(go.Candlestick(x=df['timestamp'][index_X - 100:index_sell + 100],
                                 open=df['open'][index_X - 100:index_sell + 100],
                                 high=df['high'][index_X - 100:index_sell + 100],
                                 low=df['low'][index_X - 100:index_sell + 100],
                                 close=df['close'][index_X - 100:index_sell + 100]))
    # region
    if X >= A:
        PL = (exit_price - enter_price) / enter_price - 0.002
    else:
        PL = (enter_price - exit_price) / enter_price - 0.002
        # PL = round(PL, 3)

    # fig.add_scatter(go.Scatter(x=df['timestamp'][index_sell],y=exit_price,text=f'{PL*100}%'))
    # endregion
    # fig.add_trace(go.Scatter(x=df['timestamp'][index_X - 10:index_sell + 10],
    #                          y=df['Ichimoku_base_line'][index_X - 10:index_sell + 10]))
    # fig.add_trace(go.Scatter(x=df['timestamp'][index_X - 10:index_sell + 10],
    #                          y=df['Ichimoku_conversion_line'][index_X - 10:index_sell + 10]))
    # fig.add_trace(go.Scatter(
    #     x=[df['timestamp'][index_X], df['timestamp'][index_A], df['timestamp'][index_B],
    #        df['timestamp'][index_C], df['timestamp'][index_buy], df['timestamp'][index_sell]],
    #     y=[X, A, B, C, enter_price, exit_price], mode='markers+text',
    #     marker=dict(size=[10, 11, 12, 13, 10, 10], color=[0, 1, 2, 3, 0, 0]),
    #     text=[None, None, None, None, None, None],
    #     textposition='bottom center'))
    fig.add_trace(go.Scatter(
        x=[df['timestamp'][index_buy], df['timestamp'][index_sell]],
        y=[enter_price, exit_price],
        mode='markers', marker=dict(size=[10, 10], symbol="3", color='black')))
    fig.add_trace(go.Scatter(
        x=[df['timestamp'][index_X], df['timestamp'][index_A], df['timestamp'][index_B],
           df['timestamp'][index_C], df['timestamp'][index_buy], df['timestamp'][index_sell]],
        y=[X, A, B, C, enter_price, exit_price],
        mode='lines', marker=dict(size=[10, 10, 10, 10], symbol="6", color='black')))

    # fig.add_shape(type="line",
    #               x0=df['timestamp'][index_buy], y0=min(df.loc[index_X:index_sell, 'low']),
    #               x1=df['timestamp'][index_buy], y1=max(df.loc[index_X:index_sell, 'high']))
    # fig.add_shape(type="line",
    #               x0=df['timestamp'][index_sell], y0=min(df.loc[index_X:index_sell, 'low']),
    #               x1=df['timestamp'][index_sell], y1=max(df.loc[index_X:index_sell, 'high']))
    # region Fibo line
    if X >= A:
        fibo_line_y = C + 2 * abs(B - C)
        PL_fibo = (fibo_line_y-B)/B
    else:
        fibo_line_y = C - 2 * abs(B - C)
        PL_fibo = (B-fibo_line_y)/B

    fig.add_shape(type="line",
                  x0=df['timestamp'][index_A], y0=fibo_line_y,
                  x1=df['timestamp'][index_sell], y1=fibo_line_y)
    # endregion
    # region B line
    fig.add_shape(type="line",
                  x0=df['timestamp'][index_B], y0=B,
                  x1=df['timestamp'][index_sell], y1=B)
    # endregion
    if index_sell + 100 < len(df):
        fig.add_trace(go.Bar(x=df['timestamp'][index_X - 100:index_sell + 100],
                             y=df['MACD1_Hist'][index_X - 100:index_sell + 100]), row=2, col=1)
    fig.update_layout(height=height, width=width, xaxis_rangeslider_visible=False)
    fig.add_annotation(x=df['timestamp'][index_sell], y=0.999 * exit_price,
                       text=f'{round(PL * 100, 4)}%',
                       showarrow=False, font=dict(size=16, color='black'))
    fig.add_annotation(x=df['timestamp'][index_sell], y=0.999 * fibo_line_y,
                       text=f'{round(PL_fibo * 100, 4)}%',
                       showarrow=False, font=dict(size=16, color='black'))
    fig.show()
