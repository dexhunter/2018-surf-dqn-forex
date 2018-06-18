from oandapyV20 import API
import sqlite3
import v20
from datetime import datetime
import time

hostname = "api-fxpractice.oanda.com"
port = 443
ssl = True
ID = None #your ID
token = None # your token
start = "2017-07-07T22:00:00Z"
end = "2017-07-13T22:00:00Z"
instrument = "EUR_USD"
granularity = "M5"

def init_db(path):
    #with sqlite3.connect(DATABASE_DIR) as connection:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS MARKET \
                (time TEXT, instrument VARCHAR(10), type CHAR(4), high FLOAT, \
                close FLOAT, low FLOAT, open FLOAT, volume INTEGER, \
                complete BOOLEAN)''')
    print "Table created successfully!"
    conn.close()

def insert_test(path):
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO MARKET VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
             ("2017-07-13T06:50:00.000000000Z", "EUR_USD", "ask", 1.14505, 1.14477, 1.14468, 1.14502, 91, True))
    conn.commit()
    print "records created successfully!"
    conn.close()

def insert(database_path, time, h, c, l, o, vol, complete, instrument="EUR_USD", price="mid"):
    conn = sqlite3.connect(database_path)
    conn.execute("INSERT INTO MARKET VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (time, instrument, price, h, c, l, o, vol, complete))
    conn.commit()
    print "success"
    conn.close()

def select(path):
    conn = sqlite3.connect(path)
    cursor = conn.execute("SELECT time, instrument, high, volume, complete FROM MARKET")
    for row in cursor:
        print "time = ", row[0]
        print "instrument = ", row[1]
        print "high = ", row[2]
        print "volume = ", row[3]
        print "complete = ", row[4], "\n"
    conn.close()

def get_data_test(database_path):
    api = v20.Context(
            hostname,
            port,
            ssl,
            application="test",
            token=token)
    #kwargs = {}
    #test = None
    #if test is None:
    #    kwargs['price'] = "M"
    #    kwargs['granularity']= "M5"
    #    kwargs['smooth'] = True
    #    kwargs['fromTime'] = start
    #    kwargs['toTime'] = end
    response = api.instrument.candles("EUR_USD", price="M", granularity="M5", smooth=True, fromTime=start, toTime=end)

    if response.status != 200:
        print(response)
        print(response.body)
        return

    candles = response.get("candles", 200)
    print(candles)

    for candle in response.get("candles", 200):
        print(candle.time) #unicode
        candletime = datetime.strptime(candle.time, "%Y-%m-%dT%H:%M:%S.000000000Z")
        unixtime = int(time.mktime(candletime.timetuple()))
        print(unixtime)
        print((candle.mid)) #candlestickdata
        print(type(candle.mid.l)) #float
        #print(candle.price) #unicode
        print(type(candle.volume)) #int
        print(type(candle.complete)) #bool
        print(type(candle.ask)) #None
        insert(database_path, candle.time, candle.mid.h, candle.mid.c, candle.mid.h, candle.mid.l, candle.mid.o, candle.volume, candle.complete)



if __name__ == "__main__":
    path = 'test.db'
    init_db(path)
    insert_test(path)
    select(path)
    get_data_test(path) #passed

