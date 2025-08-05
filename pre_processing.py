import pandas as pd

AAPL_PATH = "./Dataset/AAPL.csv"
TSLA_PATH = "./Dataset/TSLA.csv"
GOOG_PATH = "./Dataset/GOOG.csv"
MSFT_PATH = "./Dataset/MSFT.csv"
AMZN_PATH = "./Dataset/AMZN.csv"
ETH_PATH = "/mnt/data/ethusdt_15m.parquet"


# Apple stock's pre processing


def _apple_data():
    # Drop NULL rows
    AAPL = pd.read_csv(AAPL_PATH)
    AAPL.dropna(inplace=True)
    # Change Dtype of Columns
    AAPL["Date"] = pd.to_datetime(AAPL["Date"])
    AAPL["Volume"] = AAPL["Volume"].str.replace(',', '')
    AAPL = AAPL.astype({"Open": float, "Volume": float})
    # Sort the Database by Date
    AAPL = AAPL.sort_values(by='Date', ignore_index=True, ascending=True)
    # Drop rows having Date < '2015-01-01'
    # AAPL = AAPL[AAPL["Date"] >= '2015-01-01'].reset_index(drop=True)

    return AAPL

# Tesla stock's pre processing


def _tesla_data():
    TSLA = pd.read_csv(TSLA_PATH)
    # Drop NULL rows
    TSLA.dropna(inplace=True)
    # Change Dtype of Columns
    TSLA["Date"] = pd.to_datetime(TSLA["Date"])
    TSLA["Volume"] = TSLA["Volume"].str.replace(',', '')
    TSLA = TSLA.astype({"Open": float, "Volume": float})
    # Sort the Database by Date
    TSLA = TSLA.sort_values(by='Date', ignore_index=True, ascending=True)
    # Drop rows having Date < '2015-01-01'
    # TSLA = TSLA[TSLA["Date"] >= '2015-01-01'].reset_index(drop=True)

    return TSLA

# Google stock's pre processing


def _google_data():
    GOOG = pd.read_csv(GOOG_PATH)
    # Drop NULL rows
    GOOG.dropna(inplace=True)
    # Change Dtype of Columns
    GOOG["Date"] = pd.to_datetime(GOOG["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj. Close", "Volume"]:
        GOOG[col] = GOOG[col].astype(str).str.replace(',', '')
    GOOG = GOOG.astype({"Open": float, "High": float, "Low": float,
                       "Close": float, "Adj. Close": float, "Volume": float})
    # Sort the Database by Date
    GOOG = GOOG.sort_values(by='Date', ignore_index=True, ascending=True)
    # Drop rows having Date < '2015-01-01'
    # GOOG = GOOG[GOOG["Date"] >= '2015-01-01'].reset_index(drop=True)

    return GOOG

# Microsoft stock's pre processing


def _mircrosoft_data():
    MSFT = pd.read_csv(MSFT_PATH)
    # Drop NULL rows
    MSFT.dropna(inplace=True)
    # Change Dtype of Columns
    MSFT["Date"] = pd.to_datetime(MSFT["Date"])
    MSFT["Open"] = MSFT["Open"].str.replace(',', '')
    MSFT["Volume"] = MSFT["Volume"].str.replace(',', '')
    MSFT = MSFT.astype({"Open": float, "Volume": float})
    # Sort the Database by Date
    MSFT = MSFT.sort_values(by='Date', ignore_index=True, ascending=True)
    # Drop rows having Date < '2015-01-01'
    # MSFT = MSFT[MSFT["Date"] >= '2015-01-01'].reset_index(drop=True)

    return MSFT

# Amazon stock's pre processing


def _amazon_data():
    AMZN = pd.read_csv(AMZN_PATH)
    # Drop NULL rows
    AMZN.dropna(inplace=True)
    # Change Dtype of Columns
    AMZN["Date"] = pd.to_datetime(AMZN["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj. Close", "Volume"]:
        AMZN[col] = AMZN[col].astype(str).str.replace(',', '')
    AMZN = AMZN.astype({"Open": float, "High": float, "Low": float,
                       "Close": float, "Adj. Close": float, "Volume": float})
    # Sort the Database by Date
    AMZN = AMZN.sort_values(by='Date', ignore_index=True, ascending=True)
    # Drop rows having Date < '2015-01-01'
    # AMZN = AMZN[AMZN["Date"] >= '2015-01-01'].reset_index(drop=True)

    return AMZN

def _eth_data():
    # 读取parquet格式数据
    ETH = pd.read_parquet(ETH_PATH).reset_index()
    
    print("ETH数据中的列名:", ETH.columns.tolist())
    
    # 1. 处理缺失值 - 删除空值行
    ETH.dropna(inplace=True)
    
    # 2. 数据类型转换
    # 将timestamp转换为datetime类型（对应股票数据的Date列）
    ETH["timestamp"] = pd.to_datetime(ETH["timestamp"])
    
    # 处理数值列中的可能存在的逗号（虽然parquet通常不会有，但保持与股票处理一致）
    for col in ["open", "high", "low", "close", "volume"]:
        # 先转换为字符串处理，再转回浮点型
        ETH[col] = ETH[col].astype(str).str.replace(',', '')
        ETH[col] = ETH[col].astype(float)
    
    # 3. 按时间戳升序排序
    ETH = ETH.sort_values(by='timestamp', ignore_index=True, ascending=True)
    
    # 4. 可选：过滤2015年1月1日之前的数据（与股票数据处理对应）
    # ETH = ETH[ETH["timestamp"] >= '2015-01-01'].reset_index(drop=True)
    
    # 5. 保留与股票数据相同的列，并调整列名以匹配（timestamp对应date）
    # 选择需要保留的列
    ETH_processed = ETH[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 重命名列名以匹配股票数据格式（timestamp -> Date）
    ETH_processed.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    return ETH_processed


def processed_data_dict():
    return {
        # "AAPL": _apple_data(), 
        # "TSLA": _tesla_data(),
        # "GOOG": _google_data(),
        # "MSFT": _mircrosoft_data(), 
        # "AMZN": _amazon_data(),
        "ETH": _eth_data()
    }

# if __name__ == "__main__":
#     # data visualisation
#     plt.figure(figsize=(16, 8))
#     plt.title('Apple Stock Price History')
#     plt.plot(AAPL['Date'], AAPL['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)

#     plt.figure(figsize=(16, 8))
#     plt.title('Tesla Stock Price History')
#     plt.plot(TSLA['Date'], TSLA['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)

#     plt.figure(figsize=(16, 8))
#     plt.title('Google Stock Price History')
#     plt.plot(GOOG['Date'], GOOG['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)

#     plt.figure(figsize=(16, 8))
#     plt.title('Microsoft Stock Price History')
#     plt.plot(MSFT['Date'], MSFT['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)

#     plt.figure(figsize=(16, 8))
#     plt.title('Amazon Stock Price History')
#     plt.plot(AMZN['Date'], AMZN['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)
    
#     plt.show()

