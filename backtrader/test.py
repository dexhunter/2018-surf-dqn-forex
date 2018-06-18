from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt

from datetime import datetime
import os.path
import sys
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from backtrader.utils.py3 import with_metaclass
from scipy.spatial.distance import cdist, euclidean

class PercFilter(bt.metabase.ParamsBase):
    params = (('base', 1),)
    def __init__(self, data):
        self._refclose = None
        self._refopen = None
        self._refhigh = None
        self._reflow = None

    def __call__(self, data, *args, **kwargs):
        if self._refclose is None:
            self._refclose = data.close[0]
        if self._refopen is None:
            self._refopen = data.open[0]
        if self._refhigh is None:
            self._refhigh = data.high[0]
        if self._reflow is None:
            self._reflow = data.low[0]

        #pc = 100.0 * (data.close[0] / self._refclose - 1.0)
        #data.close[0] = self.p.base + pc
        data.close[0] /= self._refclose
        data.open[0] /= self._refopen
        data.high[0] /= self._refhigh
        data.low[0] /= self._reflow

        return False # no change to stream structure/length

class TestStrategy(bt.Strategy):

        def log(self, txt, dt=None):
            ''' Logging function fot this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

        def __init__(self):
            # Keep a reference to the "close" line in the data[0] dataseries
            self.dataclose = self.datas[0].close
            self.dataopen = self.datas[0].open
            self.order = None
            self.buyprice = None
            self.buycomm = None
            self.b = None # portfolio weight vector
            self.last_b = None
            self.eps = 5 # reversion threshold
            self.W = 5 # length of window
            self.history = np.ones(len(self.datas))
            #self.count = 0

        def start(self):
            self.order = None
            self.mystats = open('mystats.csv', 'wb')
            self.mystats.write('datetime, pv\n')

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                if order.isbuy():
                    pass
                    #self.log('BUY Executed, Price: %.2f, Cost: %.2f, Comm: %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
                    #self.buyprice = order.executed.price
                    #self.buycomm = order.executed.comm

                elif order.issell():
                    pass
                    #self.log('SELL Executed, Price: %.2f, Cost: %.2f, Comm:%.2f' % (order.executed.price, order.executed.value, order.executed.comm))

                self.bar_executed = len(self)

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                pass
                #self.log('Order Canceled/Margin/Rejected')
                #print('now open: ', self.dataopen[0], 'last close: ', self.dataclose[-1])

            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return

            #self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


        def euclidean_proj_simplex(self, v, s=1):
            '''Compute the Euclidean projection on a positive simplex
            :param v: n-dimensional vector to project
            :param s: int, radius of the simple
            return w numpy array, Euclidean projection of v on the simplex
            Original author: John Duchi
            '''
            assert s>0, "Radius s must be positive (%d <= 0)" % s

            n, = v.shape # raise ValueError if v is not 1D
            # check if already on the simplex
            if v.sum() == s and np.alltrue( v>= 0):
                return v

            # get the array of cumulaive sums of a sorted copy of v
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            # get the number of >0 components of the optimal solution
            rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
            # compute the Lagrange multiplier associated to the simplex constraint
            theta = (cssv[rho] - s) / (rho + 1.)
            w = (v-theta).clip(min=0)
            return w


        def l1_median_VaZh(self, X, eps=1e-5):
            '''calculate the L1_median of X with the l1median_VaZh method
            '''
            y = np.mean(X, 0)

            while True:
                D = cdist(X, [y])
                nonzeros = (D != 0)[:, 0]

                Dinv = 1 / D[nonzeros]
                Dinvs = np.sum(Dinv)
                W = Dinv / Dinvs
                T = np.sum(W * X[nonzeros], 0)
                num_zeros = len(X) - np.sum(nonzeros)
                if num_zeros == 0:
                    y1 = T
                elif num_zeros == len(X):
                    return y
                else:
                    R = (T - y) * Dinvs
                    r = np.linalg.norm(R)
                    rinv = 0 if r==0 else num_zeros/r
                    y1 = max(0, 1-rinv)*T + min(1, rinv)*y

                if euclidean(y, y1) < eps:
                    return y1

                y = y1

        def update(self, data, last_b, eps, W):
            t1 = data.shape[0]
            if t1 < W+2:
                x_t1 = data[t1-1, :]
            else:
                x_t1 = self.l1_median_VaZh(data[(t1-W):(t1-1),:]) / data[t1-1,:]

            if np.linalg.norm(x_t1 - x_t1.mean())**2 == 0:
                tao = 0
            else:
                tao = min(0, (x_t1.dot(last_b)-eps) / np.linalg.norm(x_t1 - x_t1.mean())**2)
            if self.b is None:
                self.b = np.ones(data.shape[1])/data.shape[1]
            else:
                self.b -= tao * (x_t1 - x_t1.mean() * np.ones(x_t1.shape))
                self.b = self.euclidean_proj_simplex(self.b)
            return self.b


        def next(self):
            # Simply log the closing price of the series from the reference
            #self.log('Close[0], %.2f' % self.dataclose[0])
            #self.log('current portfolio value is %.2f' % self.stats.broker.value[0])
            #for i in range(len(self.datas)):
            #    print(self.datas[i].close[0])
            #self.log('Close[-1], %.2f' % self.dataclose[-1])
            #self.history.append(self.dataclose)
            #print(self.datas[0].datetime.date(0), 'buflen', self.buflen())
            #len(self.datas) # 7(num of datafeed)
            N = len(self.datas)

            nx = np.ones(len(self.datas))

            for i in range(len(self.datas)):
                nx[i] = self.datas[i].close[0]

            self.history = np.vstack((self.history, nx))
            #print(self.history.shape)

            if self.last_b is None:
                self.last_b = np.ones(len(self.datas)) / len(self.datas)
            else:
                self.last_b = self.b

            b = self.update(self.history, self.last_b, self.eps, self.W)
            #print(b)


            for i in range(len(self.datas)):
                self.order_target_percent(data=self.datas[i], target=b[i])

            #print(self.data.datetime.date(0).strftime('%Y-%m-%d'), self.count)
            #print(self.data.datetime.date(0).strftime('%Y-%m-%d'), self.stats.broker.value[0])
            #self.count += 1


            self.mystats.write(self.data.datetime.date(0).strftime('%Y-%m-%d'))
            self.mystats.write(',%.2f' % self.stats.broker.value[0])
            self.mystats.write('\n')

            #print('------------')
            #print(self.broker.get_value([self.data0, self.data1, self.data2]))


            '''
            if self.order:
                return

            if not self.position:

                if self.dataclose[0] < self.dataclose[-1]:
                    """if the current close price is smaller than the previous"""
                    self.log('Buy Created, %.2f' % self.dataclose[0])
                    self.order = self.buy()
            else:
                #print('position', self.position)
                #print('bar_executed', self.bar_executed) # How many bars have been passed to strategy (according to Data)
                if len(self) >= self.bar_executed + 5:
                    self.log('Sell Created, %.2f' % self.dataclose[0])

                    self.order = self.sell()
            '''


            #self.log('Open, %.2f' % self.dataopen[0])
            #print(type(self.datas)) #list
            #j = 0
            #for i in self.datas:
            #    print(self.datas[j])
            #    j += 1
            #print(self.dataopen[0])
            #print(self.dataopen[1])
            #print(self.dataopen[2])
            #pass
            #print(self.history)

if __name__ == "__main__":



    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    datapath = "orcl-1995-2014.txt"

    start=datetime(2003,1,1)
    end=datetime(2004,12,31)

    #download_data = web.DataReader("AMZN", "google", start, end)
    #download_data.to_csv("amzn.csv")

   # data = bt.feeds.YahooFinanceCSVData(
   #         dataname='amzn.csv',
   #         fromdate=start,
   #         todate=end,
   #         reverse=False)

    #df = pd.read_csv("amzn.csv", index_col=0)
    df = {}
    name = ["amzn", "intc", "msft", "adbe", "cvs", "ms", "mmm"]
    #print(df.head())
    #dataframe['date'] = datetime.strptime(dataframe['date'], '%Y-%m-%d')
    #df.set_index('Date', inplace=True)
    #print(df['Date'])
    #df.rename_axis('Date')
    data = {}
    for i in range(len(name)):
        df[i] = pd.read_csv("./data/"+name[i]+".csv")
        df[i]['Date'] = pd.to_datetime(df[i]['Date'])
        df[i].set_index('Date', inplace=True)

    #df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    #print(df.head())

    #dataframe = web.DataReader("AMZN", 'google', start, end)

        data[i] = bt.feeds.PandasData(dataname=df[i],fromdate=start, todate=end)
        #data[i].addfilter(PercFilter)
        print(df[i].shape)
        cerebro.adddata(data[i])

    cerebro.broker.setcash(1000000.0)

    cerebro.broker.setcommission(commission=0.001)
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    #cerebro.plot()


