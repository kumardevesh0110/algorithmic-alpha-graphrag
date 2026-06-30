import pandas as pd
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

#loading the Pre-Trained Transformer Model and Tokenizer
print("Loading FinBERT Transformer model(this may take a minute on first run)...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    """It passes financial text through FinBERT to calculate sentiment probabilities."""
    #if text is empyt or not a string , return Neutral
    if not isinstance(text, str) or len(text.strip())==0:
        return "Neutral",0.0
    
    #tokenizing and converting to pytorch tensors
    inputs = tokenizer(text,padding=True,truncation =True,max_length=512, return_tensors="pt")

    #running a forward pass  through FinBERT (no_grad disables backpropogation to save memory)
    with torch.no_grad():
        outputs = model(**inputs)

    #applying softmax to raw logits to prob. distribution
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    #FinBERT's internal label ordering: Index 0= Positive , 1= Negative , 2=Neutral
    labels = ["Positive", " Negative", "Neutral"]
    max_index = torch.argmax(probabilities).item()

    sentiment = labels[max_index]
    confidence = probabilities[0][max_index].item() #extracting exact probabilites score

    return sentiment , confidence

def process_stored_news(ticker):
    # thsi loads csv news, runs sentiments  analysis 
    csv_path = f"data/{ticker}_news.csv"
    if not os.path.exists(csv_path):
        print(f" Error: No data found for {ticker}. Did you run ingest.py first?")
        return 
    
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df)} articles. runing FinBERT inference...")

    sentiments = []
    confidences = []

    #loop through every article summary and analyze it
    for idx , row in df.iterrows():
        sentiment, confidence = analyze_sentiment(row["summary"])
        sentiments.append(sentiment)
        confidences.append(confidence)

    #adding our new mathematical  features back into the table
    df['sentiment']= sentiments
    df['confidence']= confidences

    #saving our engineered data set to a new file
    output_path = f"data/{ticker}_processed.csv"
    df.to_csv(output_path, index = False)
    print(f"Analysis complete! features saved to {output_path}")

if __name__ == "__main__":
    #processing the data that we pulled
    
    #first checking if the user provided a ticke in the terminal 
    #sys.argv[0] is the script name, sysargv[1] is the first argument
    if len(sys.argv)>1:
        target_ticker = sys.argv[1].upper()
    else:
        #default
        target_ticker = "NVDA"

    process_stored_news(target_ticker)
    
    
