from datetime import datetime, timedelta
import time
from ib_async import IB, Stock
import sys
import json

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

def load_simple_ticker_list(filename='sp500_tickers.txt'):
    """Load tickers from simple text file (one per line)"""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"❌ File {filename} not found.")
        return []

def get10YData(ib, ticker):
    """Get 10 years of news data for a single ticker"""
    print(f"\n🔍 Starting news collection for {ticker}")
    
    # Local variables for this ticker only
    end_date = datetime.now()
    all_articles = []
    seen_articles = set()
    
    try:
        newsProviders = ib.reqNewsProviders()
        codes = '+'.join(provider.code for provider in newsProviders)  # Fixed: use provider.code

        news_stock = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(news_stock)
        
        for week in range(52 * 10):  # Fixed: 10 years = 520 weeks
            start_date = end_date - timedelta(days=7)
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S.0')
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S.0')
            print(f"  Week {week+1}/520: {start_str} to {end_str}")
            
            try:
                headlines = ib.reqHistoricalNews(news_stock.conId, codes, start_str, end_str, 300)
                week_articles = 0
                
                for headline in headlines:
                    if headline.articleId in seen_articles:
                        continue
                    seen_articles.add(headline.articleId)
                    
                    conf = None
                    if 'C:' in headline.headline:
                        try:
                            conf_str = headline.headline.split('C:')[1].split('}')[0]
                            conf = float(conf_str)
                        except Exception:
                            conf = None
                    
                    if conf is not None and conf >= 0.9:
                        try:
                            article = ib.reqNewsArticle(headline.providerCode, headline.articleId)
                            clean_headline = headline.headline.split('}', 1)[-1] if '}' in headline.headline else headline.headline
                            print(f"    {clean_headline}")
                            all_articles.append({
                                'datetime': str(headline.time),  # Convert to string immediately
                                'headline': clean_headline,
                                'confidence': conf,
                                'article': str(article) if article else None
                            })
                            week_articles += 1
                        except Exception as e:
                            print(f"    ❌ Error getting article: {e}")
                
                print(f"    → {week_articles} articles found")
                
                # Save backup every 50 weeks
                if (week + 1) % 50 == 0:
                    with open(f'news_data/backups/{ticker}_backup_week_{week+1}.json', 'w') as f:
                        json.dump(all_articles, f, indent=2)
                    print(f"    💾 Backup saved: {len(all_articles)} total articles")
                
            except Exception as e:
                print(f"    ❌ Error getting headlines for week {week+1}: {e}")
            
            # Move to the previous week
            end_date = start_date
            time.sleep(1)  # Avoid pacing violations
            
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
        return []
    
    return all_articles

def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', ports[3], clientId=998)
        print(f"{'='*5}Connected{'='*5}")
        
        tickers = load_simple_ticker_list()
        print(f"📈 Processing {len(tickers)} tickers")
        
        for idx, ticker in enumerate(tickers):
            print(f"\n[{idx+1}/{len(tickers)}] Processing {ticker}")
            
            ticker_articles = get10YData(ib, ticker)
            
            # Save individual ticker file
            with open(f'news_data/{ticker}.json', 'w') as f:
                json.dump(ticker_articles, f, indent=2)
            
            print(f"✅ {ticker}: Saved {len(ticker_articles)} articles")
            
            # Brief pause between tickers
            time.sleep(5)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        ib.disconnect()
        print("🔌 Disconnected")

if __name__ == "__main__":
    main()