import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_financial_news(ticker):
    #to fetch recent financial news articles for a given stock ticker
    api_key = os.getenv("NEWS_API_KEY")
    #using aplha vantage 
    url= f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}" 


    print (f"Fetching news data for {ticker}...")
    response= requests.get(url)
    if response.status_code !=200:
        print("API Request Failed")
        return []

    data = response.json()
    articles =  data.get("feed",[])
    return articles


def process_articles(articles):
    #strucutring raw api json data intoa clean pandas frame
    extracted_data =[]
    for article in articles[:20]:  # limiting ot top 20 articles per ticker for MVP speed
        extracted_data.append({
            "title": article.get("title"),
            "summary": article.get("summary"),
            "url": article.get("url"),
            "time_published": article.get("time_published"),
            "banner_image": article.get("banner_image")
        })
    return pd.DataFrame(extracted_data)

if __name__ == "__main__":
    # Test with a high-activity ticker
    target_ticker = "NVDA"
    raw_news = fetch_financial_news(target_ticker)


    if raw_news:
        df= process_articles(raw_news)
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/{target_ticker}_news.csv",index=False)
        print(f"Successfully saved(len{df}) articles to data/{target_ticker}_news.csv")