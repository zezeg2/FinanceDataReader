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
    'r_trend_following': False,
    'title': '',
    'ylabel': '',
    'moving_average_type': 'SMA',  # 'SMA', 'WMA', 'EMA'
    'moving_average_lines': (5, 20, 60),
    'save':False
}

color_dict = {0: '#FF7F50', 1: '#8FBC8F', 2: '#708090', 3: '#DDA0DD', 4: '#6A5ACD'}

__plot_params = dict(__fact_def_params)

plotly_install_msg = '''
FinanceDataReade.chart.plot() dependen on plotly
plotly not installed please install as follows

FinanceDataReade.chart.plot()는 plotly 의존성이 있습니다.
명령창에서 다음과 같이 plotly 설치하세요.

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
        import plotly.io as pio
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

    candle = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='red',  # 상승봉 스타일링
        decreasing_line_color='blue',  # 하락봉 스타일링
    )

    ma_type = params['moving_average_type']
    weights = np.arange(240) + 1
    # 이동평균선 기본값 20 입력
    params['moving_average_lines'].insert(0,20)
    moving_average_lines = []
    for value in params['moving_average_lines']:
        if value not in moving_average_lines:
            moving_average_lines.append(value)

    for n in moving_average_lines:  # moving average lines
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

    df['ma_default'] = df['Close'].rolling(window=20).mean()  # 20일 이동평균
    df['stddev_default'] = df['Close'].rolling(window=20).std()  # 20일 이동표준편차
    df['upper_default'] = df['ma_default'] + 2 * df['stddev_default']  # 상단밴드
    df['lower_default'] = df['ma_default'] - 2 * df['stddev_default']  # 하단밴드

    df['PB'] = (df['Close'] - df['lower_default']) / (df['upper_default'] - df['lower_default'])
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

    for ix, n in enumerate(moving_average_lines):
        globals()['ma_{}'.format(n)] = go.Scatter(x=df.index, y=df[f'MA_{n}'], line=dict(color=color_dict[ix], width=1),
                                                  name=f'ma_{n}', showlegend=False)

    # number of row
    row_criteria = (2 if params['volume'] else 1)
    nor = 0;

    # 보조 지표 옵션
    # 거래량
    if params['volume']:
        volume_bar = go.Bar(x=df.index, y=df['Volume'], showlegend=False,
                            marker_color=list(map(lambda x: "red" if x else "blue", df.Volume.diff() >= 0)))
    # 볼린저 밴드
    if params['bollinger']:
        for ix, n in enumerate(moving_average_lines):
            df[f'stddev_{n}'] = df['Close'].rolling(window=20).std()  # n일 이동표준편차
            df[f'upper_{n}'] = df[f'MA_{n}'] + 2 * df[f'stddev_{n}']  # 상단밴드
            df[f'lower_{n}'] = df[f'MA_{n}'] - 2 * df[f'stddev_{n}']  # 하단밴드

            globals()['upper_{}'.format(n)] = go.Scatter(x=df.index, y=df[f'upper_{n}'],
                                                         line=dict(color=color_dict[ix], width=1, dash='dot'), name=f'upper_{n}',
                                                         showlegend=True)
            globals()['lower_{}'.format(n)] = go.Scatter(x=df.index, y=df[f'lower_{n}'],
                                                         line=dict(color=color_dict[ix], width=1, dash='dot' ), name=f'lower_{n}',
                                                         showlegend=True)

    # MACD
    if params['macd']:
        nor += 1
        macd_row = row_criteria + nor
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
        stochastic_row = row_criteria + nor
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
        mfi_row = row_criteria + nor
        PB = go.Scatter(x=df.index, y=df['PB'] * 100, line=dict(color='blue', width=2), name='PB', legendgroup='group4',
                        legendgrouptitle_text='PB, MFI')
        MFI10 = go.Scatter(x=df.index, y=df['MFI10'], line=dict(dash='dashdot', color='green', width=2), name='MFI10')

    # RSI
    if params['rsi']:
        nor += 1
        rsi_row = row_criteria + nor
        U = np.where(df['Close'].diff(1) > 0, df['Close'].diff(1), 0)
        D = np.where(df['Close'].diff(1) < 0, df['Close'].diff(1) * (-1), 0)
        AU = pd.DataFrame(U, index=df.index).rolling(window=14).mean()
        AD = pd.DataFrame(D, index=df.index).rolling(window=14).mean()
        RSI = AU / (AD + AU) * 100
        df['RSI'] = RSI
        RSI = go.Scatter(x=df.index, y=df['RSI'], line=dict(color='red', width=2), name='RSI', legendgroup='group5',
                         legendgrouptitle_text='RSI')

    df = df[25:]

    row_heights = [3 for i in range(row_criteria + nor)]
    row_heights[0] = 7

    fig = ms.make_subplots(rows= row_criteria + nor, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                row_heights=row_heights)
    fig.update_layout(
        width=params['width'],
        height= params['height'],
        title=params['title'],
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor = 'white'
    )
    fig.update_xaxes(tickformat='%y-%m-%d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True,
                          gridwidth=1, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black',
                          mirror=True)
    fig.update_yaxes(title = params['ylabel'], tickformat=',d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True,
                          gridwidth=1,
                          gridcolor='lightgray', showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_traces(xhoverformat='%y년%m월%d일')

    fig.add_trace(candle, row=1, col=1)
    for n in moving_average_lines:
        fig.add_trace(globals()['ma_{}'.format(n)], row=1, col=1)

    if params['volume']:
        fig.add_trace(volume_bar, row=2, col=1)
    if params['bollinger']:
        for n in moving_average_lines:
            fig.add_trace(globals()['upper_{}'.format(n)], row=1, col=1)
            fig.add_trace(globals()['lower_{}'.format(n)], row=1, col=1)
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

    # RSI
    if params['rsi']:
        fig.add_trace(RSI, row=rsi_row, col=1)
    # 추세 추종
    if params['trend_following']:
        for i in range(len(df['Close'])):
            if df['PB'][i] > 0.8 and df['MFI10'][i] > 80:
                trend_fol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='orange',
                                       marker_size=15, marker_symbol='triangle-up', opacity=0.7, showlegend=False)
                fig.add_trace(trend_fol, row=1, col=1)
            elif df['PB'][i] < 0.2 and df['MFI10'][i] < 20:
                trend_fol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='darkblue',
                                       marker_size=15, marker_symbol='triangle-down', opacity=0.7, showlegend=False)
                fig.add_trace(trend_fol, row=1, col=1)
    # 역추세 추종
    if params['r_trend_following']:
        df['II'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
        df['IIP21'] = df['II'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum() * 100
        for i in range(len(df['Close'])):
            if df['PB'][i] < 0.05 and df['IIP21'][i] > 0:
                trend_refol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='purple',
                                         marker_size=15,
                                         marker_symbol='triangle-up', opacity=0.7, showlegend=False)  # 보라
                fig.add_trace(trend_refol, row=1, col=1)
            elif df['PB'][i] > 0.95 and df['IIP21'][i] < 0:
                trend_refol = go.Scatter(x=[df.index[i]], y=[df['Close'][i]], marker_color='skyblue',
                                         marker_size=15,
                                         marker_symbol='triangle-down', opacity=0.7, showlegend=False)  # 하늘
                fig.add_trace(trend_refol, row=1, col=1)

    fig.update_layout(
        width=params['width'],
        height=params['height'],
        title=params['title'],
        xaxis1_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50), template='seaborn'
    )

    fig.show(config=dict({'scrollZoom': True}))


    if params['save']:
        home_path = os.path.expanduser('~')
        if not os.path.exists(home_path + "/figures"):
            os.mkdir(home_path + "/figures")
        pio.write_image(fig, home_path + "/figures/" + "fig_" + params['title'], format='png')
