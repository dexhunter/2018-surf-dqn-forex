from __future__ import division,absolute_import,print_function
from nntrader.marketdata.poloniex import Poloniex
import logging
import numpy as np
from time import time
import pandas as pd
try:
    from numba import jit
except ImportError:
    def identity(ob):
        return ob
    jit = identity


def pricenorm3d(m, features, norm_method, fake_ratio=1.0, with_y=True):
    """normalize the price tensor, whose shape is [features, coins, windowsize]
    @:param m: input tensor, unnormalized and there could be nan in it
    @:param with_y: if the tensor include y (future price)
        logging.debug("price are %s" % (self._latest_price_matrix[0, :, -1]))
    """
    result = m.copy()
    if features[0] != "close":
        raise ValueError("first feature must be close")
    for i, feature in enumerate(features):
        if with_y:
            one_position = 2
        else:
            one_position = 1
        pricenorm2d(result[i], m[0, :, -one_position], norm_method=norm_method,
                    fake_ratio=fake_ratio, one_position=one_position)
    return result


# input m is a 2d matrix, (coinnumber+1) * windowsize
# no jit 2min37s jit 1min50s
@jit
def pricenorm2d(m, reference_column,
                norm_method="absolute", fake_ratio=1.0, one_position=2):
    if norm_method=="absolute":
        output = np.zeros(m.shape)
        for row_number, row in enumerate(m):
            if np.isnan(row[-one_position]) or np.isnan(reference_column[row_number]):
                row[-one_position] = 1.0
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0:
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                row[-one_position] = 1.0
                row[-1] = fake_ratio
            else:
                row = row / reference_column[row_number]
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0 and np.isnan(row[-one_position - index]):
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                if np.isnan(row[-1]):
                    row[-1] = fake_ratio
            output[row_number] = row
        m[:] = output[:]
    elif norm_method=="relative":
        output = m[:, 1:]
        divisor = m[:, :-1]
        output = output / divisor
        pad = np.empty((m.shape[0], 1,))
        pad.fill(np.nan)
        m[:] = np.concatenate((pad, output), axis=1)
        m[np.isnan(m)] = fake_ratio
    else:
        raise ValueError("there is no norm morthod called %s" % norm_method)


# TODO: find the reason of the URLopen error: peers reset
def get_chart_until_success(polo, pair, start, period, end):
    is_connect_success = False
    chart = {}
    while not is_connect_success:
        try:
            chart = polo.marketChart(pair=pair, start=start, period=period, end=end)
            is_connect_success = True
        except Exception as e:
            print(e)
    return chart


def panel2array(panel):
    """convert the panel to datatensor (numpy array) without btc
    """
    without_btc = np.transpose(panel.as_matrix(), axes=(2,0,1))
    return without_btc


def get_panel(selected_coins, length=50, period=1800, features=("close","high","low")):
    """
    @:param selected_coins: list of coin names
    @:param length: total number of periods
    @:param period: trading period
    @:param features: tuple or list of feature names
    @:return : a panel of [coin, time, features]
    """
    t = int(time())
    p = Poloniex()
    data_dict = {}
    # Note that the date is the start time of a period
    for coin in selected_coins:
        if "reversed_" in coin:
            pair = coin.replace("reversed_","") + "_BTC"
        else:
            pair = "BTC_" + coin
        chart = get_chart_until_success(p,
                                        start=t - (length+1) * period,
                                        end=t - period,
                                        pair=pair,
                                        period=period)
        df = pd.DataFrame(chart)[["date"] + list(features)].set_index("date")
        if "reversed_" in coin and len(features) == 3:
            df = 1 / df
            col_list = list(df)
            col_list[2], col_list[1] = col_list[1], col_list[2]
            df.columns = col_list
        data_dict[coin] = df
    panel = pd.Panel(data_dict)
    return panel


def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward


def online_data_generator(selected_coins, length=50, period=1800, features=("close","high","low")):
    """get a online data panel generator, use next() to get a new data panel.
    Note that, unlike get_panel(), only necessary data will be download. The
    `next` should only at least be called one time each period.
    :return : a panel of [coin, time, features]
    """
    panel = get_panel(selected_coins, length, period, features)
    while True:
        new_panel = get_panel(selected_coins, 1, period, features)
        if max(new_panel.major_axis) == max(panel.major_axis) + period:
            panel = panel.drop(min(panel.major_axis), axis=1)
            panel = pd.concat([panel, new_panel], axis=1)
        elif max(new_panel.major_axis) == max(panel.major_axis):
            logging.warning("try to generate input multiple times in a period")
        else:
            message = "there might be periods missing, new date is %s while old is %s" % \
                      (max(new_panel.major_axis), max(panel.major_axis))
            logging.error(message)
            raise ValueError(message)
        yield panel
