from datetime import datetime, timedelta
import time
from ib_async import *
import sys

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

ib = IB()
ib.connect('127.0.0.1', ports[3], clientId=1)

newsProviders = ib.reqNewsProviders()
codes = '+'.join(np.code for np in newsProviders)

news_stock = Stock('MSFT', 'SMART', 'USD')
ib.qualifyContracts(news_stock)

# Get news from the past 10 years
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)
end_str = end_date.strftime('%Y%m%d %H:%M:%S') # Today
start_str = start_date.strftime('%Y%m%d %H:%M:%S') # 10yrs from now

all_articles = []

# IBKR may limit the number of headlines per request, so loop by chunks
current_end = end_str
while True:
    headlines = ib.reqHistoricalNews(news_stock.conId, codes, start_str, current_end, 300)
    if not headlines:
        break
    for headline in headlines:
        conf = None
        if 'C:' in headline.headline:
            try:
                conf_str = headline.headline.split('C:')[1].split('}')[0]
                conf = float(conf_str)
            except Exception:
                conf = None
        # Save all articles, not just high confidence
        if conf is not None and conf >= 0.8:
            article = ib.reqNewsArticle(headline.providerCode, headline.articleId)
            clean_headline = headline.headline.split('}', 1)[-1] if '}' in headline.headline else headline.headline
            print(headline.time, clean_headline)
            all_articles.append({
                'datetime': headline.time,
                'headline': clean_headline,
                'confidence': conf,
                'article': article
            })
            
    # Move the end date back to just before the oldest headline time to get older news
    last_time = headlines[-1].time
    if isinstance(last_time, datetime):
        current_end = (last_time - timedelta(seconds=1)).strftime('%Y%m%d %H:%M:%S')
        print(current_end)
    else:
        current_end = (datetime.fromtimestamp(last_time / 1000) - timedelta(seconds=1)).strftime('%Y%m%d %H:%M:%S')

    time.sleep(1)  # Avoid pacing violations

# Save to file
import json
with open('msft_news_10y.json', 'w') as f:
    json.dump(all_articles, f, indent=2)
    
ib.disconnect()