# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.50  # 30 % of price increase
STOP_LOSSES_THRESHOLD = 0.80
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_scatter(y=df['Close'], mode='lines', name='Close',
                      line={'color': 'green'})
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3})
    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'black', 'width': 2})

    """
    graph.add_scatter(y=df['EMA7'], mode='lines', name='EMA7',
                      line={'color': 'orange', 'width': 2})
    graph.add_scatter(y=df['EMA32'], mode='lines', name='EMA32',
                      line={'color': 'magenta', 'width': 2})
    """

    # graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
    #                   line={'color': 'magenta'})
    # graph.add_scatter(y=df['EMA20'], mode='lines', name='EMA20',
    #                   line={'color': 'red'})

    # graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    """
    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'magenta'})
    graph.add_scatter(y=df['EMA7'], mode='lines', name='EMA7',
                      line={'color': '#ff4455'})

    graph.add_scatter(y=df['EMA20'], mode='lines', name='EMA20', line={'color': 'magenta'})
    """

    # graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s',
    #                   line={'color': '#ff4040', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)

    """
    graph.add_scatter(y=df['S_trend_l3'], mode='lines', name='S_trend_l3',
                      line={'color': '#FF5733', 'width': 4}, row=1, col=1)

    graph.add_scatter(y=df['S_trend_l7'], mode='lines', name='S_trend_l7',
                      line={'color': '#6495ED', 'width': 4}, row=1, col=1)

    graph.add_scatter(y=df['S_trend_l14'], mode='lines', name='S_trend_l14',
                      line={'color': '#50C878', 'width': 4}, row=1, col=1)
    """

    """
    graph.add_scatter(y=df['EMA34'], mode='lines', name='EMA34')

    graph.add_scatter(y=df['L'], mode='lines', name='L', line={'color': 'red'})
    graph.add_scatter(y=df['U'], mode='lines', name='U', line={'color': 'black'})
    """


    """
    df['K'] = ((df['EMA20'] - df['EMA7']) / df['Close'] - 0.02)

    bar_colors = np.array(['#bada55'] * len(df))
    bar_colors[df['K'] < df['K'].shift(1)] = '#ff4040'

    graph.add_bar(y=df['K'], name='K', row=2, col=1, marker={'color': bar_colors})

    df.ta.sma(close=df['K'], length=14, append=True, col_names=('K_EMA7',))
    """

    """
    graph.add_scatter(y=df['K'], mode='lines', name='K',
                      line={'color': 'black'}, row=2, col=1)
    graph.add_scatter(y=df['K2'], mode='lines', name='K2',
                      line={'color': 'blue'}, row=2, col=1)

    graph.add_scatter(y=df['RSI'], mode='lines', name='RSI', row=2, col=1)
    graph.add_scatter(y=df['CTI'], mode='lines', name='CTI',
                      line={'color': '#FF00DF'}, row=2, col=1)
    """

    graph.add_scatter(y=df['RSI'], mode='lines', name='RSI', row=2, col=1)

    """
    graph.add_scatter(y=df['MACD'], mode='lines', name='MACD',
                      line={'color': 'black'}, row=2, col=1)
    graph.add_scatter(y=df['MACD_signal'], mode='lines', name='MACD_signal',
                      line={'color': 'red'}, row=2, col=1)
    """

    graph.show()


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, auto_adjust=True)  # interval="1h",

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.ema(length=7, append=True, col_names=('EMA7',))
    df.ta.ema(length=32, append=True, col_names=('EMA32',))

    """
    df.ta.ema(length=3, append=True, col_names=('EMA3',))
    df.ta.ema(length=7, append=True, col_names=('EMA7',))
    df.ta.ema(length=14, append=True, col_names=('EMA14',))
    df.ta.ema(length=20, append=True, col_names=('EMA20',))
    df.ta.ema(length=34, append=True, col_names=('EMA34',))
    """

    df.ta.rsi(length=RSI_PERIOD, append=True, col_names=('RSI',))
    df.ta.cti(append=True, col_names=('CTI',))
    # df.ta.macd(append=True, period=50, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df.ta.supertrend(append=True, multiplier=4.0, length=32,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    """
    df.ta.supertrend(append=True, multiplier=3.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l3', 'S_trend_s3',))
    df.ta.supertrend(append=True, multiplier=7.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l7', 'S_trend_s7',))
    df.ta.supertrend(append=True, multiplier=34.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l14', 'S_trend_s14',))

    df['S_trend_l3'] -= 10
    df['S_trend_l7'] -= 13
    df['S_trend_l14'] -= 16
    """

    """
    for i in range(k_period * 2, len(df['Close'])):
        # Calculate average volatility for the last X hours
        # volatility = np.std(df['Close'].values[j - k_period:j])

        # Calculate the integral
        diff = df['EMA14'].values[i - k_period * 2:i] - df['EMA7'].values[i - k_period * 2:i]

        # Here we should apply some multiplayer for the recent diff values
        df_diff = pd.DataFrame(diff, columns=['diff'])
        df_diff.ta.ema(close=df_diff['diff'], length=5, append=True, col_names=('EMA', ))
        df_diff = df_diff[k_period:].copy()

        prob_up = df_diff[df_diff['EMA'] > 0].sum()['EMA']
        prob_down = abs(df_diff[df_diff['EMA'] < 0].sum())['EMA']

        k_indicator[i] = prob_up / (prob_up + prob_down)

        # Calculate the integral
        diff = df['EMA34'].values[i - k_period * 2:i] - df['SMA50'].values[i - k_period * 2:i]
        df_diff = pd.DataFrame(diff, columns=['diff'])
        df_diff.ta.ema(close=df_diff['diff'], length=k_period/2, append=True, col_names=('EMA',))
        df_diff = df_diff[k_period:].copy()

        prob_up = df_diff[df_diff['EMA'] > 0].sum()['EMA']
        prob_down = abs(df_diff[df_diff['EMA'] < 0].sum())['EMA']

        k2_indicator[i] = prob_up / (prob_up + prob_down)

    df['K'] = k_indicator.values()
    df['K2'] = k2_indicator.values()
    """

    draw(df)


if __name__ == '__main__':

    for ticker in ['AMD']:
        run_backtest(ticker)  # 1 year is 250 days
