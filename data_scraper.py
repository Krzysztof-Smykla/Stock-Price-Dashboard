import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

# ================================
# Fetch S&P 500 tickers
# ================================

# wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# tables = pd.read_html(wiki_url, header=0)

import requests

wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
} # Some websites block requests that don't have a user-agent header, so we include one to mimic a browser.

response = requests.get(wiki_url, headers=headers)

tables = pd.read_html(response.text)
sp500_table = tables[0]

tickers = sp500_table["Symbol"].tolist()

# Convert special tickers for yfinance (BRK.B → BRK-B)
tickers = [ticker.replace(".", "-") for ticker in tickers]
tickers = sorted(tickers)

print(f"Total S&P 500 tickers fetched: {len(tickers)}")

# Save ticker list
os.makedirs("data", exist_ok=True)
pd.DataFrame(tickers, columns=["Ticker"]).to_csv("data/sp500_tickers.csv", index=False)

# NOTE: Downloading all historical data from yfinance for 500 tickers at once can trigger rate limits or timeouts.
# The same thing can happen when scraping data from Wikipedia without proper handling. 
# Large requests or requests with too many parallel threads are very likely to get blocked by the server. 
# To avoid this, we can implement a batch download approach with delays between batches or download all data in one call to the yfinance API.
# This is more efficient and less likely to trigger rate limits.

# Quick fixes for API limits ("HTTP Error 403: Forbidden" or "HTTP Error 429: Too Many Requests"):
# 1. Reduce batch size (e.g., 25 tickers per batch).
# 2. Add delays between batches (e.g., 5 seconds).
# 3. Restart router or use a VPN to change IP address if blocked.

# ================================
# Safe Batch Download
# ================================

BATCH_SIZE = 25          # Safe batch size
SLEEP_BETWEEN_BATCHES = 5  # Seconds between batches
MAX_RETRIES = 3

def download_batch(batch):
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\nDownloading batch: {batch}")
            
            df = yf.download(
                tickers=batch,
                period="5y",
                interval="1d",
                group_by="ticker",
                threads=False,  # safer than True
                progress=False
            )
            
            return df
        
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(3)

    print("Batch failed after retries.")
    return None


# ================================
# Iterate Through Batches
# ================================

for i in range(0, len(tickers), BATCH_SIZE):
    
    batch = tickers[i:i+BATCH_SIZE]
    data = download_batch(batch)
    
    if data is not None:
        for ticker in batch:
            try:
                ticker_df = data[ticker].dropna(how="all")
                
                if not ticker_df.empty:
                    ticker_df.to_csv(f"data/{ticker}.csv")
                    print(f"Saved {ticker}")
                else:
                    print(f"No data for {ticker}")
            
            except Exception as e:
                print(f"Failed to save {ticker}: {e}")
    
    print(f"Sleeping {SLEEP_BETWEEN_BATCHES} seconds to avoid rate limits...")
    time.sleep(SLEEP_BETWEEN_BATCHES)

print("\n All downloads complete.")
