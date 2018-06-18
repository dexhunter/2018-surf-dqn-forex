from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import datetime	 # For datetime objects
import os.path	# To manage paths
import sys	# To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist, euclidean

# from backtrader.utils.py3 import with_metaclass

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

# Create a Stratey
class TestStrategy(bt.Strategy):
	'''params = (
		('maperiod', 15),
	)'''

	def log(self, txt, dt=None):
		''' Logging function fot this strategy'''
		dt = dt or self.datas[0].datetime.date(0)
		print('%s, %s' % (dt.isoformat(), txt))

	def __init__(self, eps=10, alpha=0.5, data_phi=None, b=None):
		# Keep a reference to the "close" line in the data[0] dataseries
		self.dataclose = self.datas[0].close
		self.dataopen = self.datas[0].open

		# To keep track of pending orders
		self.order = None
		self.buyprice = None
		self.buycomm = None
		self.b = None # portfolio weight vector
		self.last_b = None
		self.eps = 5 # reversion threshold
		self.W = 5 # length of window
		self.history = np.ones(len(self.datas))

		super(TestStrategy, self).__init__()
		self.eps = eps
		self.alpha = alpha
		self.data_phi = data_phi
		self.b = b

		'''# Adding a MovingAverageSimple indicator
		self.sma = bt.indicators.SimpleMovingAverage(
			self.datas[0], period=self.params.maperiod)'''

	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			# Buy/Sell order submitted/accepted to/by broker - Nothing to do
			return

		# Check if an order has been completed
		# Attention: broker could reject order if not enough cash
		if order.status in [order.Completed]:
			if order.isbuy():
				self.log(
					'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
					(order.executed.price,
					 order.executed.value,
					 order.executed.comm))

				self.buyprice = order.executed.price
				self.buycomm = order.executed.comm

			else: # Sell
				self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
						 (order.executed.price,
						  order.executed.value,
						  order.executed.comm))

			self.bar_executed = len(self)

		elif order.status in [order.Canceled, order.Margin, order.Rejected]:
			self.log('Order Canceled/Margin/Rejected')

		# Write down: no pending order
		self.order = None

	def notify_trade(self, trade):
		if not trade.isclosed:
			return

		self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
				(trade.pnl, trade.pnlcomm))

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

	def l1_median_VaZh(self, XX, eps=1e-5):
		'''calculate the L1_median of XX with the l1median_VaZh method
		'''
		y = np.mean(XX, 0)

		while True:
			D = cdist(XX, [y])
			nonzeros = (D != 0)[:, 0]

			Dinv = 1 / D[nonzeros]
			Dinvs = np.sum(Dinv)
			W = Dinv / Dinvs
			T = np.sum(W * XX[nonzeros], 0)
			num_zeros = len(XX) - np.sum(nonzeros)
			if num_zeros == 0:
				y1 = T
			elif num_zeros == len(XX):
				return y
			else:
				R = (T - y) * Dinvs
				r = np.linalg.norm(R)
				rinv = 0 if r==0 else num_zeros/r
				y1 = max(0, 1-rinv)*T + min(1, rinv)*y

			if euclidean(y, y1) < eps:
				return y1

			y = y1

	def decide_by_history(self):

		self.record_history(x)
		nx = self.get_last_rpv(x)

		if self.b is None:
			self.b = np.ones(nx.size) / nx.size
		last_b = self.b
		if self.data_phi is None:
			self.data_phi = np.ones((1,nx.size))
		else:
			self.data_phi = self.alpha + (1-self.alpha)*self.data_phi/nx

		ell = max(0, self.eps - self.data_phi.dot(last_b))

		x_bar = self.data_phi.mean()
		denominator = np.linalg.norm(self.data_phi - x_bar)**2

		if denominator == 0:
			lam = 0
		else:
			lam = ell / denominator

		self.data_phi = np.squeeze(self.data_phi)
		b = last_b + lam * (self.data_phi - x_bar)

		b = self.euclidean_proj_simplex(b)
		self.b = b
		return self.b

	def next(self):
		'''# Simply log the closing price of the series from the reference
		self.log('Close, %.2f' % self.dataclose[0])

		# check if an order is pending ... if yes, we cannot send a 2nd one
		if self.order:
			return

		# Check if we are in the market
		if not self.position:

			# Not yet ... we MIGHT BUY if ...
			if self.dataclose[0] > self.sma[0]:
				# current close less than previous close

				# buy (with all possible default parameters)
				self.log('BUY CREAT, %.2f' % self.dataclose[0])

				# Keep track of the created order to avoid a 2nd order
				self.order = self.buy()

		else:

			# Already in the market ... we might sell
			if self.dataclose[0] < self.sma[0]:
				# sell (with all possible defualt parameters)
				self.log('SELL CREATE, %.2f' % self.dataclose[0])

				# Keep track of the created order to avoid a 2nd order
				self.order = self.sell()'''

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

		b = self.decide_by_history(self, nx, last_b)
		print(b)


		#for i in range(len(self.datas)):
		#	self.order_target_percent(data=self.datas[i], target=b[i])

if __name__ == '__main__':
	# Create a cerebro entity
	cerebro = bt.Cerebro()

	# Add a strategy
	cerebro.addstrategy(TestStrategy)

	'''# Datas are in a subfolder of the samples. Need to find where the script is
	# because it could have been called from anywhere
	datapath = "E:/OANDA historical data/backtrader/datas/yhoo-1996-2015.txt"

	# Create a Data Feed
	data = bt.feeds.YahooFinanceData(
		dataname=datapath,
		# Do not pass values before this date
		fromdate=datetime.datetime(2000, 1, 1),
		# Do not pass values before this date
		todate=datetime.datetime(2000, 12, 31),
		# Do not pass values after this date
		reverse=False)'''

	df = {}
	name = ['EUR_USD', 'GBP_USD']

    # , 'GBP_USD', 'AUD_USD', 'CAD_USD', 'CHF_USD', 'JPY_USD', 'NZD_USD'

	data = {}
	for i in range(len(name)):
		df[i] = pd.read_csv("../data/"+name[i]+".CSV")
		df[i]['Date'] = pd.to_datetime(df[i]['Date'])
		df[i].set_index('Date', inplace=True)
		print(df[i].head())

	#dataframe = web.DataReader("AMZN", 'google', start, end)

		data[i] = bt.feeds.PandasData(dataname=df[i])
		#data[i].addfilter(PercFilter)

		cerebro.adddata(data[i])

	cerebro.broker.setcash(1000000.0)

	cerebro.broker.setcommission(commission=0.001)

	'''# Add the Data Feed to Cerebro
	cerebro.adddata(data)

	# Set our desired cash start
	cerebro.broker.setcash(1000.0)


	# Add a FixedSize sizer according to the stake
	cerebro.addsizer(bt.sizers.FixedSize, stake=10)

	# Set the commission - 0.1% ... divide by 100 to remove the %
	cerebro.broker.setcommission(commission=0.0)'''

	# Print out the starting conditions
	print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Run over everything
	cerebro.run()

	# Print out the final result
	print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

	'''# Plot the result
	cerebro.plot()'''
