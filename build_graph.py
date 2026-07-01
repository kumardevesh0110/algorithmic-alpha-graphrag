import os
import sys
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class FinancialGraph:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_ticker_data(self, ticker):
        csv_path = f"data/{ticker}_processed.csv"
        if not os.path.exists(csv_path):
            print(f"No processed data found for {ticker}. Did you run sentiment_engine.py first?")
            return

        df = pd.read_csv(csv_path)
        
        with self.driver.session() as session:
            session.run("MERGE (t:Ticker {name: $ticker})", ticker=ticker)
            print(f"Injecting {len(df)} financial news nodes into Neo4j for {ticker}...")
            
            for idx, row in df.iterrows():
                query = """
                MERGE (a:Article {url: $url})
                ON CREATE SET a.title = $title, a.summary = $summary, 
                              a.sentiment = $sentiment, a.confidence = $confidence
                
                WITH a
                MATCH (t:Ticker {name: $ticker})
                MERGE (a)-[:MENTIONS {impact_score: $confidence}]->(t)
                """
                session.run(query, 
                            url=row['url'], 
                            title=row['title'], 
                            summary=row['summary'], 
                            sentiment=row['sentiment'], 
                            confidence=float(row['confidence']),
                            ticker=ticker)
                
        print(f"Graph generation complete for {ticker}!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_ticker = sys.argv[1].upper()
    else:
        target_ticker = "NVDA"
        
    graph = FinancialGraph()
    graph.import_ticker_data(target_ticker)
    graph.close()