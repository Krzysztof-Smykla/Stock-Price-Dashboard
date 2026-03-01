import os
import time
import logging
import random
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================
# Setup Logging
# ============================================
def configure_logging():
    """Configure logging for the data scraper."""
    logging.basicConfig(
        filename="sp500_downloader.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("Starting S&P 500 data pipeline...")

# ============================================
# Create Robust Session (for Wikipedia)
# ============================================

def create_session():
    """Create a requests session with retry logic to handle transient errors."""
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ])
    })

    return session


# ============================================
# 3️⃣ Fetch S&P 500 Tickers Safely
# ============================================

def fetch_sp500_tickers():
    """Scrape the list of S&P 500 tickers from Wikipedia with retry logic."""

    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    session = create_session()
    response = session.get(wiki_url)
    response.raise_for_status()

    tables = pd.read_html(response.text)
    sp500_table = tables[0]

    tickers = sp500_table["Symbol"].tolist()

    # Convert dot tickers for Yahoo
    tickers = [t.replace(".", "-") for t in tickers]

    tickers = sorted(tickers)

    logging.info(f"Fetched {len(tickers)} tickers.")
    return tickers


# ============================================
# 4️⃣ Safe Batch Downloader
# ============================================

def download_batch(batch):
    try:
        df = yf.download(
            tickers=batch,
            period="5y",
            interval="1d",
            group_by="ticker",
            threads=False,
            progress=False
        )
        return df

    except Exception as e:
        logging.error(f"Batch failed: {e}")
        return None
    
# NOTE: Downloading all historical data from yfinance for 500 tickers at once can trigger rate limits or timeouts.
# The same thing can happen when scraping data from Wikipedia without proper handling. 
# Large requests or requests with too many parallel threads are very likely to get blocked by the server. 
# To avoid this, we can implement a batch download approach with delays between batches or download all data in one call to the yfinance API.
# This is more efficient and less likely to trigger rate limits.

# Quick fixes for API limits ("HTTP Error 403: Forbidden" or "HTTP Error 429: Too Many Requests"):
# 1. Reduce batch size (e.g., 25 tickers per batch).
# 2. Add delays between batches (e.g., 5 seconds).
# 3. Restart router or use a VPN to change IP address if blocked.

# ============================================
# 5️⃣ Main Execution
# ============================================

def main():

    os.makedirs("data", exist_ok=True)

    tickers = fetch_sp500_tickers()

    # Save ticker list
    pd.DataFrame(tickers, columns=["Ticker"]).to_csv(
        "data/sp500_tickers.csv", index=False
    )

    BATCH_SIZE = 15
    SLEEP_BETWEEN_BATCHES = 7

    for i in range(0, len(tickers), BATCH_SIZE):

        batch = tickers[i:i + BATCH_SIZE]

        print(f"\nDownloading batch {i//BATCH_SIZE + 1}...")
        logging.info(f"Downloading batch: {batch}")

        data = download_batch(batch)

        if data is not None:

            for ticker in batch:
                filepath = f"data/{ticker}.csv"

                # Resume capability
                if os.path.exists(filepath):
                    logging.info(f"Skipping existing {ticker}")
                    continue

                try:
                    ticker_df = data[ticker].dropna(how="all")

                    if not ticker_df.empty:
                        ticker_df.to_csv(filepath)
                        logging.info(f"Saved {ticker}")
                        print(f"Saved {ticker}")
                    else:
                        logging.warning(f"No data for {ticker}")

                except Exception as e:
                    logging.error(f"Error saving {ticker}: {e}")

        print(f"Sleeping {SLEEP_BETWEEN_BATCHES} seconds...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("\nAll downloads completed.")
    logging.info("Download process completed successfully.")


if __name__ == "__main__":
    main()
