#import pandas_datareader as pdr
#print(pdr.DataReader("GOOG", "yahoo"))

import  pandas_datareader.data as web
from datetime import datetime

def d(symbol):
    start = datetime(2003,1,1)
    end = datetime(2005,12,31)
    df = web.DataReader(symbol, 'google', start, end)
    df.to_csv(symbol+".csv")
    return df

if __name__ == "__main__":
    print(d('amzn').head()) #amazon
    print(d('intc').head()) #intel
    print(d('msft').head()) #microsoft
    print(d('adbe').head()) #adobe
    print(d('cvs').head()) #carriage
    print(d('ms').head()) #morgran stanley
    print(d('mmm').head()) #3m
