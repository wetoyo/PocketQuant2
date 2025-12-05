import pandas as pd
import requests
from io import StringIO

def get_snp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Header to look like a browser and bypass 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        # Read table from the response text
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        print(df)
        # Extract 'Symbol' column
        tickers = df['Symbol'].tolist()
        
        # Save to file
        with open('snp500.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(tickers))
            
        print(f"Success! Saved {len(tickers)} tickers to snp500.txt")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_snp500_tickers()