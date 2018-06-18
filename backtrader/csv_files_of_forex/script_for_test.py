
#!/usr/bin/env python

import pandas as mypanda
from pandas import Series, DataFrame

import argparse

import common.config

import common.args

from view import CandlePrinter

from datetime import datetime


openprice = []
closeprice = []
highprice = []
lowprice = []
volumevalue = []
timelist = []


def main(fromtimedate, totimedate, openprice, highprice, lowprice, closeprice, volumevalue, timelist):

	"""

	Create an API context, and use it to fetch candles for an instrument.



	The configuration for the context is parsed from the config file provided

	as an argumentV

	"""



	parser = argparse.ArgumentParser()



	#

	# The config object is initialized by the argument parser, and contains

	# the REST APID host, port, accountID, etc.

	#

	common.config.add_argument(parser)



	parser.add_argument(

		"instrument",

		type=common.args.instrument,

		help="The instrument to get candles for"

	)



	parser.add_argument(

		"--mid", 

		action='store_true',

		help="Get midpoint-based candles"

	)



	parser.add_argument(

		"--bid", 

		action='store_true',

		help="Get bid-based candles"

	)



	parser.add_argument(

		"--ask", 

		action='store_true',

		help="Get ask-based candles"

	)



	parser.add_argument(

		"--smooth", 

		action='store_true',

		help="'Smooth' the candles"

	)



	parser.set_defaults(mid=False, bid=False, ask=False)



	parser.add_argument(

		"--granularity",

		default='M5',

		help="The candles granularity to fetch"

	)



	parser.add_argument(

		"--count",

		default=None,

		help="The number of candles to fetch"

	)



	date_format = "%Y-%m-%d %H:%M:%S"



	parser.add_argument(

		"--from-time",

		default=fromtimedate,

		type=common.args.date_time(),

		help="The start date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"

	)



	parser.add_argument(

		"--to-time",

		default=totimedate,

		type=common.args.date_time(),

		help="The end date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"

	)



	parser.add_argument(

		"--alignment-timezone",

		default=None,

		help="The timezone to used for aligning daily candles"

	)



	args = parser.parse_args()



	account_id = args.config.active_account



	#

	# The v20 config object creates the v20.Context for us based on the

	# contents of the config file.

	#

	api = args.config.create_context()



	kwargs = {}



	if args.granularity is not None:

		kwargs["granularity"] = args.granularity



	if args.smooth is not None:

		kwargs["smooth"] = args.smooth



	if args.count is not None:

		kwargs["count"] = args.count



	if args.from_time is not None:

		kwargs["fromTime"] = api.datetime_to_str(args.from_time)



	if args.to_time is not None:

		kwargs["toTime"] = api.datetime_to_str(args.to_time)



	if args.alignment_timezone is not None:

		kwargs["alignmentTimezone"] = args.alignment_timezone



	price = "mid"



	if args.mid:

		kwargs["price"] = "M" + kwargs.get("price", "")

		price = "mid"



	if args.bid:

		kwargs["price"] = "B" + kwargs.get("price", "")

		price = "bid"



	if args.ask:

		kwargs["price"] = "A" + kwargs.get("price", "")

		price = "ask"



	#

	# Fetch the candles

	#

	response = api.instrument.candles(args.instrument, **kwargs)

	candles = response.get("candles", 200)

	for candle in response.get("candles", 200):
	
		openprice.append(1 / candle.mid.o)
		closeprice.append(1 / candle.mid.c)
		highprice.append(1 / candle.mid.h)
		lowprice.append(1 / candle.mid.l)
		volumevalue.append(candle.volume)
		timelist.append(candle.time)

		
for i in range(41):
	fromtimebyme = 1438358400 + 60*5*5000*i
	totimebyme = 1438358400 + 60*5*5000*(i+1)
	fromtimeunix = datetime.fromtimestamp(int('%d' %(fromtimebyme)))
	fromtimedate = fromtimeunix.strftime("%Y-%m-%d %H:%M:%S")
	totimeunix = datetime.fromtimestamp(int('%d' %(totimebyme)))
	totimedate = totimeunix.strftime("%Y-%m-%d %H:%M:%S")
	
	main(fromtimedate, totimedate, openprice, highprice, lowprice, closeprice, volumevalue, timelist)
	
'''openprice_column = mypanda.Series(openprice, name = 'openprice')
closeprice_column = mypanda.Series(closeprice, name = 'closeprice')
highprice_column = mypanda.Series(highprice, name = 'highprice')
lowprice_column = mypanda.Series(lowprice, name = 'lowprice')
volumevalue_column = mypanda.Series(volumevalue, name = 'volume')
timelist_column = mypanda.Series(timelist, name = 'time')'''

save = mypanda.DataFrame({'Date':timelist,
						  'Open':openprice,
						  'High':highprice,
						  'Low':lowprice,
						  'Close':closeprice,
						  'Volume':volumevalue})
						  
save.to_csv('CAD_USD.CSV', index = False, sep = ',')