from datetime import datetime, timedelta
import time
from ib_async import IB, Stock
import sys
from typing import Any

# Gets news on a stock with IKBR

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

ib = IB()
ib.connect('127.0.0.1', ports[3], clientId=1)

newsProviders = ib.reqNewsProviders()
codes = '+'.join(np.code for np in newsProviders)

news_stock = Stock('MSFT', 'SMART', 'USD')
ib.qualifyContracts(news_stock)

# Get news week by week for 10 years
end_date = datetime.now()
all_articles = []
seen_articles = set()  # Track article IDs to avoid duplicates

for week in range(52 * 1):  # 10 years
    start_date = end_date - timedelta(days=7)
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S.0')
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S.0')
    print(f"Week {week+1}: {start_str} to {end_str}")
    
    headlines = ib.reqHistoricalNews(news_stock.conId, codes, start_str, end_str, 300)
    for headline in headlines:
        # Skip if we've already seen this article
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
            article = ib.reqNewsArticle(headline.providerCode, headline.articleId)
            clean_headline = headline.headline.split('}', 1)[-1] if '}' in headline.headline else headline.headline
            print(headline.time, clean_headline)
            all_articles.append({
                'datetime': headline.time,
                'headline': clean_headline,
                'confidence': conf,
                'article': article
            })
    
    # Move to the previous week (no gaps or overlaps)
    end_date = start_date
    time.sleep(1)  # Avoid pacing violations

# Save to file
import json
with open('msft_news_10y.json', 'w') as f:
    json.dump(all_articles, f, indent=2, default=str)

ib.disconnect()