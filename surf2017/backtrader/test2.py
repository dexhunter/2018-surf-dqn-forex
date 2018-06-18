from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt
from datetime import datetime
import os.path
import sys
#import pandas_datareader.data as web
import pandas as pd
import numpy as np
from backtrader.utils.py3 import with_metaclass
from scipy.spatial.distance import cdist, euclidean
import numpy as np
from tdagent.algorithms.cornk import CORNK

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
			self.history = np.ones(len(self.datas))
			self.last_b = None
			self.b = None
			self.td= CORNK()
			
			

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

		def start(self):
			self.order = None # sentinel to avoid operrations on pending order
		
		
			
			
		def next(self):
		
			N = len(self.datas)

			nx = np.ones(len(self.datas))

			for i in range(len(self.datas)):
				nx[i] = self.datas[i].close[0]

			self.history = np.vstack((self.history, nx))
			
			
			if self.last_b is None:
				self.last_b = np.ones(len(self.datas)) / len(self.datas)
			else:
				self.last_b = self.b
			
			


			self.b=self.td.decide_by_history(self.history[-1:, :].transpose()[None, :, :], self.last_b)
			
			for i in range(len(self.datas)):
				self.order_target_percent(data=self.datas[i], target=self.b[i])
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
			#	 print(self.datas[j])
			#	 j += 1
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
	end=datetime(2005,12,31)

	#download_data = web.DataReader("AMZN", "google", start, end)
	#download_data.to_csv("amzn.csv")

   # data = bt.feeds.YahooFinanceCSVData(
   #		 dataname='amzn.csv',
   #		 fromdate=start,
   #		 todate=end,
   #		 reverse=False)

	#df = pd.read_csv("amzn.csv", index_col=0)
	df = {}
	name = ["GBP_USD",'EUR_USD']
	#print(df.head())
	#dataframe['date'] = datetime.strptime(dataframe['date'], '%Y-%m-%d')
	#df.set_index('Date', inplace=True)
	#print(df['Date'])
	#df.rename_axis('Date')
	data = {}
	for i in range(len(name)):
		df[i] = pd.read_csv(name[i]+".csv")
		df[i]['Date'] = pd.to_datetime(df[i]['Date'])
		df[i].set_index('Date', inplace=True)
		#print(df[i].head())

	#df['Date'] = pd.to_datetime(df['Date'])
	#df.set_index('Date', inplace=True)
	#print(df.head())

	#dataframe = web.DataReader("AMZN", 'google', start, end)
				
		data[i] = bt.feeds.PandasData(dataname=df[i])
		data[i].addfilter(PercFilter)

		cerebro.adddata(data[i]) 

	cerebro.broker.setcash(100000.0)

	cerebro.broker.setcommission(commission=0.0003)
	# Print out the starting conditions
	print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Run over everything
	cerebro.run()

	# Print out the final result
	print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
	