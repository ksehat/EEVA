import websocket, json


def on_message(ws, message):
    print(message)


def on_close(ws):
    print('ws connection is closed!!!')


def on_error(ws, e):
    print(e)


symbol='ethusdt'
data_step = '1m'
socket = f'wss://stream.binance.com:443/ws/{symbol}@kline_{data_step}'
ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close, on_error=on_error)
ws.run_forever()