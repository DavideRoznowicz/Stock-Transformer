from top_stocks_processing import read_from_csv, get_real_date, find_beginning, fill_the_gap, get_returns, remove_first_date, fill_not_listed, write_on_csv
import yfinance as yf
import numpy as np
import datetime
from fredapi import Fred

###########################################################
# DOWNLOAD - PROCESS - SAVE ON CSV
# STOCK TIME SERIES REQUESTED FROM FRED-YAHOO FINANCE API
###########################################################


# deal with yfinance and fred only
fred = Fred(api_key='abd2151ea5e1008c3f20682e8ef52fe1')

############## down for fred - interest rates

def get_start(time_start, time_end, date_list, stock_price, datetime_format_list):
    start_ind = date_list.index(time_start)
    end_ind = date_list.index(time_end)

    new_stock_price = []
    new_stock_dates = []
    new_stock_datetime_format = []
    for j in range(start_ind, end_ind + 1):
        new_stock_price.append(stock_price[j])
        new_stock_dates.append(date_list[j])
        new_stock_datetime_format.append(datetime_format_list[j])

    return new_stock_dates, new_stock_price, new_stock_datetime_format


def get_rid_of_weekend(datetime_format_list, date_list, stock_price):
    new_stock_price = []
    new_stock_dates = []
    new_stock_datetime_format = []
    for j in range(len(datetime_format_list)):
        if datetime_format_list[j].weekday() not in [5, 6]:
            new_stock_price.append(stock_price[j])
            new_stock_dates.append(date_list[j])
            new_stock_datetime_format.append(datetime_format_list[j])

    return new_stock_dates, new_stock_price, new_stock_datetime_format




############## down for yahoo

def fill_missing_days_yahoo(stock_price, dates_stock, real_dates):
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




def solve_stock_yahoo(symbol, stock_cl, dt_list):
    myday_list, day_list = get_real_date()
    new_stock_price, new_stock_dates = fill_missing_days_yahoo(stock_cl, dt_list, myday_list)
    new_stock_dates, returns, real_dates = get_returns(new_stock_price,  new_stock_dates, myday_list)
    ready_price = fill_not_listed(returns, new_stock_dates, real_dates)

    return ready_price, new_stock_dates, real_dates, day_list[1:]



if __name__ == "__main__":

    name_of_stock_csv="alternative.csv"
    number_of_stocks=7
    time_start = '1999-11-01'
    time_end = '2021-06-11'
    yahoo_end = '2021-06-12'

    # alternative data
    ticker_yahoo = ['GC=F', '^TNX',
                    'CL=F', 'SI=F',
                    'HG=F', 'NG=F',
                    'PL=F']

    # GOLD, 10-yr treasury, wti, silver, copper, natural gas, palladium



    ############## now let's download from yahoo finance

    ticker_yahoo = ticker_yahoo[:number_of_stocks]

    print(ticker_yahoo)



    myday_list, day_list = get_real_date()
    print(len(myday_list))



    list_of_list = []

    header = day_list[1:]
    header.insert(0, 'ticker')

    for ticker in ticker_yahoo:

        try:
            stock = yf.download(ticker, start=time_start, end=yahoo_end)
            stock_cl = stock['Adj Close'].values
            dt_list = list(stock.axes[0])
            dt_list = [str(dt_list[k].date()) for k in range(len(dt_list))]
            ready_returns, new_stock_dates, real_dates, day_list = solve_stock_yahoo(ticker, stock_cl, dt_list)
            ready_returns.insert(0, ticker)
            list_of_list.append(ready_returns)
            print(f"Ok with ticker {ticker}")
        except:
            print(f"Something Wrong with ticker: {ticker} with length {len(ready_returns)}")
            pass



    ############## now let's download from fred

    ticker = "DFF"  # interest rate

    data = fred.get_series(ticker)
    dt_list = list(data.axes[0])
    datetime_format_list = [dt_list[k].date() for k in range(len(dt_list))]
    dt_list = [str(dt_list[k].date()) for k in range(len(dt_list))]
    stock_price = list(np.array(data))
    dt_list.append(time_end)
    stock_price.append(stock_price[-1])
    datetime_format_list.append(datetime.date(2021, 6, 11))

    date_list, stock_price, datetime_format_list = get_start(time_start, time_end, dt_list, stock_price,
                                                             datetime_format_list)
    date_list, stock_price, datetime_format_list = get_rid_of_weekend(datetime_format_list, date_list, stock_price)

    myday_list, day_list = get_real_date()
    date_list, stock_price, real_dates = remove_first_date(date_list, stock_price, myday_list)
    stock_price = stock_price[1:]

    header = day_list[1:]
    header.insert(0, 'ticker')
    stock_price.insert(0, ticker)
    list_of_list.append(stock_price)
    print(f"Ok with ticker {ticker}")

    print(header[:5])
    print(len(header), len(stock_price))
    for v in list_of_list:
        print("start: ", v[:5])
        print("end: ", v[-5:])
        print(len(header), len(v))



    ############# NOW CSV WRITE
    write_on_csv(name_of_stock_csv, header, list_of_list)

    reader=read_from_csv(name_of_stock_csv)
    reader=reader.drop(labels='ticker', axis=1)
    reader=reader.to_numpy(copy=True)
    reader=reader.transpose()
    np.save("contextual.npy", reader, allow_pickle=False)