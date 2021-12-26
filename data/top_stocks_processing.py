import requests # for alphavantage querying
from datetime import datetime, date, timedelta
import json
import csv
import time
import pandas as pd
import os
import numpy as np

#####################################################
# DOWNLOAD - PROCESS - SAVE ON CSV
# STOCK TIME SERIES REQUESTED FROM ALPHAVANTAGE API
#####################################################



# NOTE: AlphaVantage accounts both for splits and dividends in the adjusted close
# first available date in alphavantage is: 1999-11-01



api_key = "ZB9MBSQH94LY8WH8"                # api_key for alphavantage
output_size = "full"                        # maximum length of time series available
function = "TIME_SERIES_DAILY_ADJUSTED"     # type of time series





def get_json(symbol, output_size=output_size, function=function, api_key=api_key):

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={output_size}&apikey={api_key}"
    r = requests.get(url)
    data = r.json()

    return data


# Monday is 0 and Sunday is 6. ---> Saturday==5, Sunday==6

def get_real_date():
    sdate = date(1999, 11, 1)  # start date
    edate = date(2021, 6, 11)  # end date

    delta = edate - sdate  # as timedelta

    day_list = []
    myday_list = []
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        if day.weekday() not in [5, 6]:
            myday_list.append(str(day))
            day_list.append(str("{}+{}+{}+{}+{}").format(i - 1, day.year, day.month, day.day, day.weekday()))

    return myday_list, day_list


# NOTE: stocks listed after the initial date are to be started later (counter is not zero!!!)

def fill_the_gap(new_stock_price, stock_price, new_stock_dates, real_dates, dates_stock, j_long, j_short):
    upto = j_long
    while real_dates[upto] != dates_stock[j_short]:
        upto += 1

    diff = upto - j_long
    baseline = stock_price[j_short - 1]
    diff_price = stock_price[j_short] - baseline
    for j in range(diff):
        new_stock_price[j_long + j] = baseline + ((j + 1) / (diff + 1)) * diff_price
        new_stock_dates[j_long + j] = real_dates[j_long + j]

    return new_stock_price, new_stock_dates, upto


def find_beginning(first_date, real_dates):
    for j in range(len(real_dates)):
        if real_dates[j] == first_date:
            break

    return j


def fill_missing_days(data, real_dates):
    # get stock dates sorted in chrono order
    dates_stock = list(data["Time Series (Daily)"].keys())
    dates_stock.reverse()

    # get stock price sorted in chrono order
    stock_price = [float(data["Time Series (Daily)"][key]["5. adjusted close"]) for key in
                   data["Time Series (Daily)"].keys()]
    stock_price.reverse()

    new_stock_price = [-1] * len(real_dates)
    new_stock_dates = [''] * len(real_dates)
    start = find_beginning(dates_stock[0], real_dates)
    j_long = start
    j_short = 0
    while j_long < len(real_dates):
        if real_dates[j_long] == dates_stock[j_short]:
            new_stock_price[j_long] = stock_price[j_short]
            new_stock_dates[j_long] = real_dates[j_long]
            j_short += 1
            j_long += 1

        else:  # real_dates[j] != dates_stock[j]:
            new_stock_price, new_stock_dates, j_long = fill_the_gap( \
                new_stock_price, stock_price, new_stock_dates, \
                real_dates, dates_stock, j_long, j_short)

    new_stock_price, new_stock_dates = new_stock_price[start:], new_stock_dates[start:]

    return new_stock_price, new_stock_dates


# remember to myday_list.remove('1999-11-01') : after getting returns the first day is not usable
def remove_first_date(new_stock_dates, returns, real_dates):
    new_stock_dates.remove(new_stock_dates[0])

    try:
        real_dates.remove('1999-11-01')
    except:
        pass

    return new_stock_dates, returns, real_dates


def get_returns(new_stock_price, new_stock_dates, real_dates):
    returns = [(new_stock_price[j] - new_stock_price[j - 1]) / (new_stock_price[j - 1]) \
               for j in range(1, len(new_stock_price))]
    new_stock_dates, returns, real_dates = remove_first_date(new_stock_dates, returns, real_dates)
    return new_stock_dates, returns, real_dates


def fill_not_listed(returns, new_stock_dates, real_dates):
    ready_returns = []
    if len(new_stock_dates) != len(real_dates):
        ready_returns = [0.0] * len(real_dates)
        ind = real_dates.index(new_stock_dates[0])
        for j in range(ind, len(real_dates)):
            ready_returns[j] = returns[j - ind]

    if ready_returns == []:
        ready_returns = returns

    return ready_returns




def solve_stock(symbol):
    myday_list, day_list = get_real_date()
    data=get_json(symbol)
    new_stock_price, new_stock_dates = fill_missing_days(data, myday_list)
    new_stock_dates, returns, real_dates = get_returns(new_stock_price,  new_stock_dates, myday_list)
    ready_returns = fill_not_listed(returns, new_stock_dates, real_dates)

    return ready_returns, new_stock_dates, real_dates, day_list[1:]


def solve_return(symbol):
    myday_list, day_list = get_real_date()
    data=get_json(symbol)
    new_stock_price, new_stock_dates = fill_missing_days(data, myday_list)
    new_stock_dates, new_stock_price, real_dates = remove_first_date(new_stock_dates, new_stock_price, myday_list)
    ready_price = fill_not_listed(new_stock_price, new_stock_dates, real_dates)


    return ready_price, new_stock_dates, real_dates, day_list[1:]




def write_on_csv(name_of_csv, header, data):
    with open(name_of_csv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

    return

def read_from_csv(name_of_csv):
    df = pd.read_csv(name_of_csv)
    return df

def emit_sound(duration, freq):
    duration = duration  # seconds
    freq = freq  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))






if __name__ == "__main__":

    name_of_stock_csv="topStocks.csv"
    number_of_stocks=160


    with open('500tickers_by_cap.json') as json_file:
        ticker_dict = json.load(json_file)

    ticker_list = ticker_dict["symbols"]
    ticker_list = ticker_list[:number_of_stocks]

    print(ticker_list)

    myday_list, day_list = get_real_date()
    print(len(myday_list))

    list_of_list = []

    header = day_list[1:]
    header.insert(0, 'ticker')
    counter = 1
    for ticker in ticker_list:
        counter += 1
        if counter%5 == 0:
            print(f"Up to now we have processed {counter/len(ticker_list)*100} % of the total list of stocks")
            print("Sleeping for 60 secs to dodge API limits...")
            time.sleep(60)
        try:
            ready_returns, new_stock_dates, real_dates, day_list = solve_stock(ticker)
            ready_returns.insert(0, ticker)
            list_of_list.append(ready_returns)
            print(f"Ok with ticker {ticker}")
        except:
            print(f"Something Wrong with ticker: {ticker} with length {len(ready_returns)}")
            pass

    write_on_csv(name_of_stock_csv, header, list_of_list)

    reader=read_from_csv(name_of_stock_csv)
    reader=reader.drop(labels='ticker', axis=1)
    reader=reader.to_numpy(copy=True)
    reader = reader.transpose()
    np.save("financial.npy", reader, allow_pickle=False)



    emit_sound()  # emits sound when the program is completed
