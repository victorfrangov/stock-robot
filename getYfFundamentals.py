import yfinance as yf
import pandas as pd
import os

# Gets fundamentals with yf
def parse_yf_data(ticker: str, 
                  balance_sheet: pd.DataFrame, 
                  cashflow: pd.DataFrame, 
                  income_stmt: pd.DataFrame, 
                  info: pd.DataFrame, 
                  ratings: pd.DataFrame) -> None:
    """Cleans fundamentals data collected from yfinance on a single stock. This method is meant to be looped over multiple stocks.
    Data still needs processing before being fed into training

    Args:
        ticker (str): The ticker symbol
        balance_sheet (pd.DataFrame): Balance sheet
        cashflow (pd.DataFrame): Cashflow
        income_stmt (pd.DataFrame): Income statement
        info (pd.DataFrame): Info on the stock
        ratings (pd.DataFrame): Analyst ratings
    """
    balance_sheet_clean = balance_sheet.loc[['TotalDebt',
                                             'TangibleBookValue',
                                             'TotalEquityGrossMinorityInterest',
                                             'StockholdersEquity',
                                             'TotalLiabilitiesNetMinorityInterest',
                                             'CurrentLiabilities',
                                             'TotalAssets',
                                             'CurrentAssets',
                                             'AccountsReceivable',
                                             'CashAndCashEquivalents',
                                             'OrdinarySharesNumber']]
    
    cashflow_clean = cashflow.loc[['FreeCashFlow',
                                   'CapitalExpenditure',
                                   'OperatingCashFlow',
                                   'ChangeInWorkingCapital',
                                   'DepreciationAndAmortization',
                                   'NetIncomeFromContinuingOperations'
                                   ]]
    
    income_stmt_clean = income_stmt.loc[['EBITDA',
                                         'DilutedEPS',
                                         'NetIncome',
                                         'TaxProvision',
                                         'OperatingIncome',
                                         'OperatingExpense',
                                         'GrossProfit',
                                         'CostOfRevenue',
                                         'TotalRevenue'
                                         ]]
    
    info_clean = info.loc[['enterpriseValue',
                           'beta',
                           'trailingPE',
                           'forwardPE',
                           'marketCap',
                           'sharesShort',
                           'shortRatio',
                           'priceToBook',
                           'trailingEps',
                           'forwardEps',
                           'targetHighPrice',
                           'targetLowPrice',
                           'targetMeanPrice',
                           'targetMedianPrice',
                           'earningsGrowth',
                           'revenueGrowth'
                           ]]
    
    ratings.index = pd.to_datetime(ratings.index)
    ratings_clean = ratings[ratings.index >= '2015-01-01']
    
    path = f'fundamentals/{ticker}'
    
    os.makedirs(path, exist_ok=True)
    
    # Add a column to identify the source of each row
    balance_sheet_clean['Source'] = 'BalanceSheet'
    cashflow_clean['Source'] = 'Cashflow'
    income_stmt_clean['Source'] = 'IncomeStatement'

    # Concatenate all cleaned DataFrames
    fundamentals_concat = pd.concat([balance_sheet_clean, cashflow_clean, income_stmt_clean])

    # Move 'Source' to the first column for clarity
    fundamentals_concat = fundamentals_concat.reset_index()
    cols = ['Source'] + [col for col in fundamentals_concat.columns if col != 'Source']
    fundamentals_concat = fundamentals_concat[cols]

    # Save to a single Excel file/sheet
    fundamentals_concat.to_excel(f'{path}/fundamentals.xlsx', index=False, sheet_name='Fundamentals')
    info_clean.to_excel(f'{path}/info.xlsx')
    ratings_clean.to_excel(f'{path}/ratings.xlsx')

def load_simple_ticker_list(filename='sp500_tickers.txt'):
    """Load tickers from simple text file (one per line)"""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"❌ File {filename} not found.")
        return []

tickers = load_simple_ticker_list()

for idx, ticker in enumerate(tickers, 1):
    try:
        b_sheet = pd.DataFrame(yf.Ticker(ticker).get_balance_sheet())
        c_flow = pd.DataFrame(yf.Ticker(ticker).get_cashflow())
        i_stmt = pd.DataFrame(yf.Ticker(ticker).get_income_stmt())
        info = yf.Ticker(ticker).get_info()
        info = pd.DataFrame(list(info.items()), columns=['Field', 'Value']).set_index('Field')
        ratings = pd.DataFrame(yf.Ticker(ticker).get_upgrades_downgrades())

        parse_yf_data(ticker, b_sheet, c_flow, i_stmt, info, ratings)
    except Exception as e:
        print(f'Error processing {ticker}: {e}')
    print(f'[{idx}/{len(tickers)}]: {ticker} processed')

