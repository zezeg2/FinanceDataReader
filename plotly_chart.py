# -*- coding: utf-8 -*-

# chart_utils.py
# (c) 2018,2021 FinaceData.KR
import os
import numpy as np
import pandas as pd

__fact_def_params = {  # factory default params
    'width': 800,
    'height': 480,
    'volume': True,
    'bollinger': False,
    'macd': False,
    'stochastic': False,
    'mfi': False,
    'rsi': False,
    'trend_following': False,
    'i_trend_following': False,
    'title': '',
    'ylabel': '',
    'moving_average_type': 'SMA',  # 'SMA', 'WMA', 'EMA'
    'moving_average_lines': (5, 20, 60),
}

__plot_params = dict(__fact_def_params)

# tableau 10 colors for moving_average_lines
tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

plotly_install_msg = '''
FinanceDataReade.chart.plot() dependen on plotly
plotly not installed please install as follows

FinanceDataReade.chart.plot()는 plotly 의존성이 있습니다.
명령창에서 다음과 같이 plotly 설치하세요

pip install plotly
'''


def config(**kwargs):
    global __plot_params

    for key, value in kwargs.items():
        if key.lower() == 'reset' and value:
            __plot_params = dict(__fact_def_params)
        elif key == 'config':
            for k, v in value.items():
                __plot_params[k] = v
        else:
            __plot_params[key] = value


def plot(df, start=None, end=None, **kwargs):
    '''
    plot candle chart with 'df'(DataFrame) from 'start' to 'end'
    * df: DataFrame to plot
    * start(default: None)
    * end(default: None)
    * recent_high: display recent high price befre n-days (if recent_high == -1 then plot recent high yesterday)
    '''
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
        import plotly.subplots as ms
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(plotly_install_msg)

    params = dict(__plot_params)
    for key, value in kwargs.items():
        if key == 'config':
            for key, value in kwargs.items():
                params[key] = value
        else:
            params[key] = value

    df = df.loc[start:end].copy()

    # plot price OHLC candles
    x = np.arange(len(df))
    height = params['height']
    candle = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='red',  # 상승봉 스타일링
        decreasing_line_color='blue',  # 하락봉 스타일링
    )

    # number of row
    nor = 1;

    # 보조 지표 옵션
    # 거래량
    if params['volume']:
        nor += 1
        volume_row = nor
        volume_bar = go.Bar(x=df.index, y=df['Volume'], showlegend=False,
                            marker_color=list(map(lambda x: "red" if x else "blue", df.Volume.diff() >= 0)))

    # 볼린저 밴드
    if params['bollinger']:
        ma_type = params['moving_average_type']
        weights = np.arange(240) + 1

        for n in params['moving_average_lines']:  # moving average lines
            if ma_type.upper() == 'SMA':
                df[f'MA_{n}'] = df.Close.rolling(window=n).mean()
            elif ma_type.upper() == 'WMA':
                df[f'MA_{n}'] = df.Close.rolling(n).apply(
                    lambda prices: np.dot(prices, weights[:n]) / weights[:n].sum())
            elif ma_type.upper() == 'EMA':
                df[f'MA_{n}'] = df.Close.ewm(span=n).mean()
            elif ma_type.upper() == 'NONE':
                pass
            else:
                raise ValueError(f"moving_average_type '{ma_type}' is invalid")

        df['ma20'] = df['Close'].rolling(window=20).mean()  # 20일 이동평균
        df['stddev'] = df['Close'].rolling(window=20).std()  # 20일 이동표준편차
        df['upper'] = df['ma20'] + 2 * df['stddev']  # 상단밴드
        df['lower'] = df['ma20'] - 2 * df['stddev']  # 하단밴드
        upper = go.Scatter(x=df.index, y=df['upper'], line=dict(color='red', width=2), name='upper', showlegend=False)
        ma20 = go.Scatter(x=df.index, y=df['ma20'], line=dict(color='black', width=2), name='ma20', showlegend=False)
        lower = go.Scatter(x=df.index, y=df['lower'], line=dict(color='blue', width=2), name='lower', showlegend=False)

    # MACD
    if params['macd']:
        nor += 1
        macd_row = nor
        df['ma12'] = df['Close'].rolling(window=12).mean()  # 12일 이동평균
        df['ma26'] = df['Close'].rolling(window=26).mean()  # 26일 이동평균
        df['MACD'] = df['ma12'] - df['ma26']  # MACD
        df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  # MACD Signal(MACD 9일 이동평균)
        df['MACD_Oscil'] = df['MACD'] - df['MACD_Signal']  # MACD 오실레이터
        MACD = go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=2), name='MACD', legendgroup='group2',
                          legendgrouptitle_text='MACD')
        MACD_Signal = go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(dash='dashdot', color='green', width=2),
                                 name='MACD_Signal')
        MACD_Oscil = go.Bar(x=df.index, y=df['MACD_Oscil'], marker_color='purple', name='MACD_Oscil')

    # 스토캐스틱
    if params['stochastic']:
        nor += 1
        stochastic_row = nor
        df['ndays_high'] = df['High'].rolling(window=14, min_periods=1).max()  # 14일 중 최고가
        df['ndays_low'] = df['Low'].rolling(window=14, min_periods=1).min()  # 14일 중 최저가
        df['fast_k'] = (df['Close'] - df['ndays_low']) / (df['ndays_high'] - df['ndays_low']) * 100  # Fast %K 구하기
        df['slow_d'] = df['fast_k'].rolling(window=3).mean()  # Slow %D 구하기
        fast_k = go.Scatter(x=df.index, y=df['fast_k'], line=dict(color='skyblue', width=2), name='fast_k',
                            legendgroup='group3', legendgrouptitle_text='%K %D')
        slow_d = go.Scatter(x=df.index, y=df['slow_d'], line=dict(dash='dashdot', color='black', width=2),
                            name='slow_d')

    # MFI
    if params['mfi']:
        nor += 1
        mfi_row = nor
        df['PB'] = (df['Close'] - df['lower']) / (df['upper'] - df['lower'])
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['PMF'] = 0
        df['NMF'] = 0
        for i in range(len(df.Close) - 1):
            if df.TP.values[i] < df.TP.values[i + 1]:
                df.PMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]
                df.NMF.values[i + 1] = 0
            else:
                df.NMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]
                df.PMF.values[i + 1] = 0
        df['MFR'] = (df.PMF.rolling(window=10).sum() /
                     df.NMF.rolling(window=10).sum())
        df['MFI10'] = 100 - 100 / (1 + df['MFR'])
        PB = go.Scatter(x=df.index, y=df['PB'] * 100, line=dict(color='blue', width=2), name='PB', legendgroup='group4',
                        legendgrouptitle_text='PB, MFI')
        MFI10 = go.Scatter(x=df.index, y=df['MFI10'], line=dict(dash='dashdot', color='green', width=2), name='MFI10')

    # RSI
    if params['rsi']:
        nor += 1
        rsi_row = nor
        U = np.where(df['Close'].diff(1) > 0, df['Close'].diff(1), 0)
        D = np.where(df['Close'].diff(1) < 0, df['Close'].diff(1) * (-1), 0)
        AU = pd.DataFrame(U, index=df.index).rolling(window=14).mean()
        AD = pd.DataFrame(D, index=df.index).rolling(window=14).mean()
        RSI = AU / (AD + AU) * 100
        df['RSI'] = RSI
        RSI = go.Scatter(x=df.index, y=df['RSI'], line=dict(color='red', width=2), name='RSI', legendgroup='group5',
                         legendgrouptitle_text='RSI')

    df = df[25:]
    fig = ms.make_subplots(rows=nor, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(candle, row=1, col=1)

    if params['volume']:
        fig.add_trace(volume_bar, row=volume_row, col=1)
    if params['bollinger']:
        fig.add_trace(upper, row=1, col=1)
        fig.add_trace(ma20, row=1, col=1)
        fig.add_trace(lower, row=1, col=1)
    if params['macd']:
        fig.add_trace(MACD, row=macd_row, col=1)
        fig.add_trace(MACD_Signal, row=macd_row, col=1)
        fig.add_trace(MACD_Oscil, row=macd_row, col=1)
    if params['stochastic']:
        fig.add_trace(fast_k, row=stochastic_row, col=1)
        fig.add_trace(slow_d, row=stochastic_row, col=1)
    if params['mfi']:
        fig.add_trace(PB, row=mfi_row, col=1)
        fig.add_trace(MFI10, row=mfi_row, col=1)
        # 추세 추종
        if params['trend_following']:
            for i in range(len(df['Close'])):
                if df['PB'][i] > 0.8 and df['MFI10'][i] > 80:
                    trend_fol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='orange', marker_size=20,
                                           marker_symbol='triangle-up', opacity=0.7, showlegend=False)
                    fig.add_trace(trend_fol, row=1, col=1)
                elif df['PB'][i] < 0.2 and df['MFI10'][i] < 20:
                    trend_fol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='darkblue', marker_size=20,
                                           marker_symbol='triangle-down', opacity=0.7, showlegend=False)
                    fig.add_trace(trend_fol, row=1, col=1)
        # 역추세 추종
        if params['i_trend_following']:
            df['II'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
            df['IIP21'] = df['II'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum() * 100
            for i in range(len(df['Close'])):
                if df['PB'][i] < 0.05 and df['IIP21'][i] > 0:
                    trend_refol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='purple',
                                             marker_size=20,
                                             marker_symbol='triangle-up', opacity=0.7, showlegend=False)  # 보라
                    fig.add_trace(trend_refol, row=1, col=1)
                elif df['PB'][i] > 0.95 and df['IIP21'][i] < 0:
                    trend_refol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='skyblue',
                                             marker_size=20,
                                             marker_symbol='triangle-down', opacity=0.7, showlegend=False)  # 하늘
                    fig.add_trace(trend_refol, row=1, col=1)


    if params['rsi']:
        fig.add_trace(RSI, row=rsi_row, col=1)

    fig.update_layout(
        autosize=True,title=params['title'],
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50), template='seaborn'
    )

    fig.update_xaxes(tickformat='%y년%m월%d일', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True,
                     gridwidth=2, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(tickformat=',d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2,
                     gridcolor='lightgray', showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_traces(xhoverformat='%y년%m월%d일')

    fig.show(config=dict({'scrollZoom': True}))

    if 'save' in kwargs:
        home_path = os.path.expanduser('~')
        if not os.path.exists(home_path + "/figures"):
            os.mkdir(home_path + "/figures")
        fig.write_image(home_path + "/figures/" + "fig_" + params['title'])
