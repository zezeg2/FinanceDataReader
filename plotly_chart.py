import datetime
import os
import math
import numpy as np
import pandas as pd
from datetime import timedelta

__fact_def_params = {  # factory default params
    'width': 1200,
    'height': 800,
    'us_style': False,
    'title': '',
    'save': False
}

color_list = ['darkmagenta', 'gold', 'limegreen', 'maroon', 'chocolate', 'seagreen', 'coral']

__plot_params = dict(__fact_def_params)

plotly_install_msg = '''
FinanceDataReade.chart.plot() dependen on plotly
plotly not installed please install as follows

FinanceDataReade.chart.plot()는 plotly 의존성이 있습니다.
명령창에서 다음과 같이 plotly 설치하세요.

pip install plotly
'''

try:
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.subplots as ms
    import plotly.io as pio

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(plotly_install_msg)


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


def plot(df: pd.DataFrame, start=None, end=None, tools={}, sub_indexes={}, **kwargs):
    """
    plot candle chart with 'df'(DataFrame) from 'start' to 'end'
    * df: DataFrame to plot
    * start(default: None)
    * end(default: None)
    * recent_high: display recent high price befre n-days (if recent_high == -1 then plot recent high yesterday)
    """

    params = dict(__plot_params)
    for key, value in kwargs.items():
        if key == 'config':
            for k, v in value.items():
                params[k] = v
        else:
            params[key] = value

    sub_indexes = dict(filter(lambda elem: elem[1] != False, sub_indexes.items()))
    if 'Volume' not in df.columns:
        sub_indexes = dict(filter(lambda elem: elem[0] == 'moving_averages', sub_indexes.items()))

    df = df.loc[start:end].copy()

    전일비_등락 = df["Close"].pct_change()
    시종가_비율 = (df["Close"] - df["Open"]) / df["Open"]
    시고가_비율 = (df["High"] - df["Open"]) / df["Open"]

    # 이동 평균선
    weights = np.arange(240) + 1

    default_moving_averages = [5, 20, 60]
    default_ma_type = 'SMA'
    ma_list = sub_indexes.pop('moving_averages') if 'moving_averages' in sub_indexes else default_moving_averages
    moving_average_lines=[];
    for i, ma in enumerate(ma_list):
        if type(ma) == int:
            window = ma
            ma_type = default_ma_type
        elif type(ma) == dict:
            window = ma['window']
            ma_type = ma['type'] if 'type' in ma else default_ma_type
        moving_average_lines.append(window)
        if (ma_type.upper() == 'SMA') | (ma_type.upper() == 'NONE'):
            df[f'MA_{window}'] = df.Close.rolling(window=window).mean()
        elif ma_type.upper() == 'WMA':
            df[f'MA_{window}'] = df.Close.rolling(window).apply(
                lambda prices: np.dot(prices, weights[:n]) / weights[:n].sum())
        elif ma_type.upper() == 'EMA':
            df[f'MA_{window}'] = df.Close.ewm(span=window).mean()
        else:
            raise ValueError(f"moving_average_type '{ma_type}' is invalid")

    df['ma_default'] = df['MA_20'] if 'MA_20' in df.columns else df['Close'].rolling(
        window=20).mean()  # default 20일(단기) 이동평균
    df['stddev_default'] = df['Close'].rolling(window=20).std()  # 20일 이동표준편차
    df['upper_default'] = df['ma_default'] + 2 * df['stddev_default']  # 상단밴드
    df['lower_default'] = df['ma_default'] - 2 * df['stddev_default']  # 하단밴드

    if ('Volume' in df.columns) & (
            ('mfi' in sub_indexes) | ('trend_following' in sub_indexes) | ('r_trend_following' in sub_indexes)):
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
        df['MFR'] = (df.PMF.rolling(window=10).sum() / df.NMF.rolling(window=10).sum())
        df['MFI10'] = 100 - 100 / (1 + df['MFR'])

    # number of row
    nr = 1;
    subplot_titles = ['OHLC']

    for i, index in enumerate(sub_indexes):
        # 거래량
        if index == 'volume':
            nr += 1
            volume_row = nr
            subplot_titles.append("Volume")

        # 볼린저 밴드
        if index == 'bollinger':
            for n in moving_average_lines:
                df[f'stddev_{n}'] = df['Close'].rolling(window=20).std()  # n일 이동표준편차
                df[f'upper_{n}'] = df[f'MA_{n}'] + 2 * df[f'stddev_{n}']  # 상단밴드
                df[f'lower_{n}'] = df[f'MA_{n}'] - 2 * df[f'stddev_{n}']  # 하단밴드


        # MACD
        if index == 'macd':
            subplot_titles.append("MACD")
            nr += 1
            macd_row = nr
            df['ma12'] = df['Close'].rolling(window=12).mean()  # 12일 이동평균
            df['ma26'] = df['Close'].rolling(window=26).mean()  # 26일 이동평균
            df['MACD'] = df['ma12'] - df['ma26']  # MACD
            df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  # MACD Signal(MACD 9일 이동평균)
            df['MACD_Oscil'] = df['MACD'] - df['MACD_Signal']  # MACD 오실레이터

        # 스토캐스틱
        if index == 'stochastic':
            subplot_titles.append("Stochastic")
            nr += 1
            stochastic_row = nr
            df['ndays_high'] = df['High'].rolling(window=14, min_periods=1).max()  # 14일 중 최고가
            df['ndays_low'] = df['Low'].rolling(window=14, min_periods=1).min()  # 14일 중 최저가
            df['fast_k'] = (df['Close'] - df['ndays_low']) / (df['ndays_high'] - df['ndays_low']) * 100  # Fast %K 구하기
            df['slow_d'] = df['fast_k'].rolling(window=3).mean()  # Slow %D 구하기

        # MFI
        if index == 'mfi':
            subplot_titles.append("MFI")
            nr += 1
            mfi_row = nr

        # RSI
        if index == 'rsi':
            subplot_titles.append("RSI")
            nr += 1
            rsi_row = nr
            U = np.where(df['Close'].diff(1) > 0, df['Close'].diff(1), 0)
            D = np.where(df['Close'].diff(1) < 0, df['Close'].diff(1) * (-1), 0)
            AU = pd.DataFrame(U, index=df.index).rolling(window=14).mean()
            AD = pd.DataFrame(D, index=df.index).rolling(window=14).mean()
            RSI = AU / (AD + AU) * 100
            df['RSI'] = RSI

    # 180일 이전 데이터 삭제 및 소수점 3자리 이하 제거
    df = df[(pd.to_datetime(df.index[0]) + timedelta(days=180) <= df.index)].round(3)

    row_heights = [3 for i in range(nr)]
    row_heights[0] = 7

    fig = ms.make_subplots(rows=nr,
                           cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=row_heights,
                           subplot_titles=subplot_titles)
    fig.update_layout(
        hovermode='x unified',
        width=params['width'],
        height=params['height'],
        title=params['title'],
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        template="plotly_white"
    )
    fig.update_xaxes(tickformat='%y-%m-%d',
                     zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='black',
                     showgrid=True,
                     gridwidth=1,
                     gridcolor='lightgray',
                     showline=True, linewidth=2,
                     linecolor='black',
                     ticks="outside",
                     minor=dict(dtick="D1", showgrid=True, ticks="outside"),
                     # x축 비영업일 제외
                     rangebreaks=[
                         dict(values=pd.date_range(df.index[0], df.index[-1]).difference(df.index))
                     ])

    fig.update_yaxes(tickformat=',', zeroline=True, zerolinewidth=1, zerolinecolor='black',
                     ticks="outside",
                     minor_ticks="outside",
                     showgrid=True,
                     gridwidth=1,
                     gridcolor='lightgray', showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_traces(xhoverformat='%y년%m월%d일')

    increase_color = 'red' if not params['us_style'] else 'green'
    decrease_color = 'blue' if not params['us_style'] else 'red'
    candle = go.Candlestick(name="OHLC",
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            increasing_line_color=increase_color,
                            decreasing_line_color=decrease_color,
                            increasing_fillcolor=increase_color,
                            decreasing_fillcolor=decrease_color,
                            increasing_line_width=1,
                            decreasing_line_width=1,
                            showlegend=True,
                            legendgroup='group1',
                            legendgrouptitle_text='Price Chart',
                            text=[f'전일대비: {chg:.1%} 시종: {oc:.1%}, 시고: {oh:.1%}' for chg, oc, oh in
                                  zip(전일비_등락, 시종가_비율, 시고가_비율)])
    fig.add_trace(candle, row=1, col=1)

    for i, ma in enumerate(ma_list):
        def_color = color_list[i % len(color_list)] if i >= len(color_list) else color_list[i]
        if type(ma) == int:
            window = ma
            line_color = def_color
        elif type(ma) == dict:
            window = ma['window']
            line_color = ma['line_color'] if 'line_color' in ma else def_color
        ma_args = dict()
        ma_args['name'] = f'ma{window}'
        ma_args['x'] = df.index
        ma_args['line'] = dict(color=line_color, width=1 if 'line_width' not in ma else ma['line_width'])
        fig.add_trace(go.Scatter(**ma_args, y=df[f'MA_{window}']), row=1, col=1)
        if 'bollinger' in sub_indexes:
            fig.add_trace(go.Scatter(**ma_args, line_dash='dot', y=df[f'upper_{window}']), row=1, col=1)
            fig.add_trace(go.Scatter(**ma_args, line_dash='dot', y=df[f'lower_{window}']), row=1, col=1)

    if sub_indexes.get('volume'):
        marker_color = list(map(lambda x: "red" if x else "blue", df.Volume.diff() >= 0)) if not params[
            'us_style'] else list(map(lambda x: "green" if x else "red", df.Volume.diff() >= 0))
        volume_bar = go.Bar(name="Volume",
                            x=df.index,
                            y=df['Volume'],
                            showlegend=False,
                            marker_color=marker_color)
        fig.add_trace(volume_bar, row=volume_row, col=1)

    if sub_indexes.get('macd'):
        MACD = go.Scatter(name='MACD',
                          x=df.index,
                          y=df['MACD'],
                          line=dict(color='blue', width=2),
                          legendgroup=f'group{macd_row}',
                          legendgrouptitle_text='MACD')
        MACD_Signal = go.Scatter(name='MACD Signal',
                                 x=df.index,
                                 y=df['MACD_Signal'],
                                 line=dict(dash='dashdot', color='green', width=2))
        MACD_Oscil = go.Bar(name='MACD Oscil',
                            x=df.index,
                            y=df['MACD_Oscil'],
                            marker_color='purple')

        fig.add_trace(MACD, row=macd_row, col=1)
        fig.add_trace(MACD_Signal, row=macd_row, col=1)
        fig.add_trace(MACD_Oscil, row=macd_row, col=1)

    if sub_indexes.get('stochastic'):
        fast_k = go.Scatter(name='fast_k',
                            x=df.index,
                            y=df['fast_k'],
                            line=dict(color='skyblue', width=2),
                            legendgroup=f'group{stochastic_row}',
                            legendgrouptitle_text='%K %D')
        slow_d = go.Scatter(name='slow_d',
                            x=df.index,
                            y=df['slow_d'],
                            line=dict(dash='dashdot', color='black', width=2))
        fig.add_trace(fast_k, row=stochastic_row, col=1)
        fig.add_trace(slow_d, row=stochastic_row, col=1)

    if sub_indexes.get('mfi'):
        PB = go.Scatter(name='PB',
                        x=df.index,
                        y=df['PB'] * 100,
                        line=dict(color='blue', width=2),
                        legendgroup=f'group{mfi_row}',
                        legendgrouptitle_text='PB, MFI')
        MFI10 = go.Scatter(name='MFI10',
                           x=df.index,
                           y=df['MFI10'],
                           line=dict(dash='dashdot', color='green', width=2))
        fig.add_trace(PB, row=mfi_row, col=1)
        fig.add_trace(MFI10, row=mfi_row, col=1)

    # RSI
    if sub_indexes.get('rsi'):
        RSI = go.Scatter(name='RSI',
                         x=df.index,
                         y=df['RSI'],
                         line=dict(color='red', width=2),
                         legendgroup=f'group{rsi_row}',
                         legendgrouptitle_text='RSI')
        fig.add_trace(RSI, row=rsi_row, col=1)

    # 추세 추종
    if sub_indexes.get('trend_following'):
        for i in range(len(df['Close'])):
            if df['PB'][i] > 0.8 and df['MFI10'][i] > 80:
                trend_fol = go.Scatter(name="trend_fol_u",
                                       x=[df.index[i]],
                                       y=[df['Close'][i]],
                                       marker_color='orange',
                                       marker_size=15,
                                       marker_symbol='triangle-up',
                                       opacity=0.8,
                                       showlegend=False)
                fig.add_trace(trend_fol, row=1, col=1)
            elif df['PB'][i] < 0.2 and df['MFI10'][i] < 20:
                trend_fol = go.Scatter(name="trend_fol_d",
                                       x=[df.index[i]],
                                       y=[df['Close'][i]],
                                       marker_color='darkblue',
                                       marker_size=15,
                                       marker_symbol='triangle-down',
                                       opacity=0.8,
                                       showlegend=False)
                fig.add_trace(trend_fol, row=1, col=1)
    # 역추세 추종
    if sub_indexes.get('r_trend_following'):
        df['II'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
        df['IIP21'] = df['II'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum() * 100
        for i in range(len(df['Close'])):
            if df['PB'][i] < 0.05 and df['IIP21'][i] > 0:
                trend_refol = go.Scatter(name="trend_refol_u",
                                         x=[df.index[i]],
                                         y=[df['Close'][i]],
                                         marker_color='purple',
                                         marker_size=15,
                                         marker_symbol='triangle-up',
                                         opacity=0.8,
                                         showlegend=False)
                fig.add_trace(trend_refol, row=1, col=1)
            elif df['PB'][i] > 0.95 and df['IIP21'][i] < 0:
                trend_refol = go.Scatter(name="trend_refol_d",
                                         x=[df.index[i]],
                                         y=[df['Close'][i]],
                                         marker_color='skyblue',
                                         marker_size=15,
                                         marker_symbol='triangle-down',
                                         opacity=0.8,
                                         showlegend=False)
                fig.add_trace(trend_refol, row=1, col=1)

    # fig.show(config=dict({'scrollZoom': True}))

    # tools - vlines: vertical lines
    vline_list = tools.pop('vlines') if 'vlines' in tools else {}
    for vline in vline_list:
        vline_args = dict(line_width=1.5, line_dash="dot", line_color="tomato", layer="below")
        if type(vline) in [pd.Timestamp, str, datetime.datetime, datetime.date]:
            vline_args['x'] = str(vline)
        elif type(vline) == dict:
            vline_args.update(vline)
        else:
            raise ValueError("'vlines' must be list of str or list of dict")
        fig.add_vline(**vline_args)

    # tools - vrects: highlighting regions
    vrect_list = tools.pop('vrects') if 'vrects' in tools else {}
    for vrect in vrect_list:
        vrect_args = dict(fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
        if type(vrect) == tuple:
            vrect_args['x0'] = str(vrect[0])
            vrect_args['x1'] = str(vrect[1])
        elif type(vrect) == dict:
            vrect_args.update(vrect)
        else:
            raise ValueError("'vrects' must be list of tuple or list of dict")
        fig.add_vrect(**vrect_args)

    if params['save']:
        home_path = os.path.expanduser('~')
        if not os.path.exists(home_path + "/figures"):
            os.mkdir(home_path + "/figures")
        pio.write_image(fig, home_path + "/figures/" + "fig_" + params['title'] + ".png", format='png')
        fig.write_html(home_path + "/figures/" + "fig_" + params['title'] + '.html')

    return fig


def readAndPlot(symbol: str, start=None, end=None, exchange=None, **kwargs):
    from FinanceDataReader.data import DataReader
    df = DataReader(symbol, start, end, exchange, for_chart=True)
    return plot(df, config=kwargs)
