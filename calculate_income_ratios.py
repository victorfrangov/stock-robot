import pandas as pd
import numpy as np

def calculate_income_statement_ratios(income_stmt_df):
    """
    Calculate key profitability ratios from income statement data
    
    Args:
        income_stmt_df: DataFrame with income statement data (columns are years)
    
    Returns:
        Dictionary of calculated ratios
    """
    
    # Get the data (assuming most recent year is first column)
    current_year = income_stmt_df.iloc[:, 0]  # Most recent year
    previous_year = income_stmt_df.iloc[:, 1] if income_stmt_df.shape[1] > 1 else None
    
    # Extract key values
    total_revenue = current_year.get('TotalRevenue', 0)
    gross_profit = current_year.get('GrossProfit', 0)
    operating_income = current_year.get('OperatingIncome', 0)
    net_income = current_year.get('NetIncome', 0)
    diluted_eps = current_year.get('DilutedEPS', 0)
    
    # Previous year values for growth calculations
    if previous_year is not None:
        prev_revenue = previous_year.get('TotalRevenue', 0)
        prev_eps = previous_year.get('DilutedEPS', 0)
    else:
        prev_revenue = total_revenue  # No growth if no previous data
        prev_eps = diluted_eps
    
    # Calculate ratios
    ratios = {}
    
    # === MARGIN RATIOS ===
    if total_revenue > 0:
        ratios['gross_margin'] = (gross_profit / total_revenue) * 100
        ratios['operating_margin'] = (operating_income / total_revenue) * 100
        ratios['net_margin'] = (net_income / total_revenue) * 100
    else:
        ratios['gross_margin'] = 0
        ratios['operating_margin'] = 0
        ratios['net_margin'] = 0
    
    # === GROWTH RATIOS ===
    if prev_revenue > 0:
        ratios['revenue_growth'] = ((total_revenue - prev_revenue) / prev_revenue) * 100
    else:
        ratios['revenue_growth'] = 0
    
    if prev_eps > 0:
        ratios['eps_growth'] = ((diluted_eps - prev_eps) / prev_eps) * 100
    elif prev_eps < 0 and diluted_eps > 0:
        ratios['eps_growth'] = 100  # Turned profitable
    else:
        ratios['eps_growth'] = 0
    
    return ratios

# Example usage with your data
def add_ratios_to_features(ticker_data):
    """
    Add income statement ratios to your feature set
    """
    
    # Assuming you have income statement data loaded
    income_stmt = ticker_data['income_statement']  # Your DataFrame
    
    # Calculate ratios
    ratios = calculate_income_statement_ratios(income_stmt)
    
    # Add to your features dictionary
    features = {
        # Your existing features...
        
        # Add the new ratios
        'gross_margin': ratios['gross_margin'],
        'operating_margin': ratios['operating_margin'], 
        'net_margin': ratios['net_margin'],
        'revenue_growth': ratios['revenue_growth'],
        'eps_growth': ratios['eps_growth'],
        
        # Additional useful ratios
        'margin_expansion': ratios['operating_margin'] - ratios.get('prev_operating_margin', ratios['operating_margin']),
        'profitability_score': (ratios['gross_margin'] + ratios['operating_margin'] + ratios['net_margin']) / 3,
        'growth_score': (ratios['revenue_growth'] + ratios['eps_growth']) / 2
    }
    
    return features

# More comprehensive ratio calculation
def calculate_comprehensive_income_ratios(income_stmt_df):
    """
    Calculate comprehensive set of income statement ratios with multi-year trends
    """
    
    ratios = {}
    
    # Get multiple years of data if available
    years = income_stmt_df.columns
    current_data = income_stmt_df.iloc[:, 0]
    
    # Current year ratios
    total_revenue = current_data.get('TotalRevenue', 0)
    gross_profit = current_data.get('GrossProfit', 0)
    operating_income = current_data.get('OperatingIncome', 0)
    net_income = current_data.get('NetIncome', 0)
    ebitda = current_data.get('EBITDA', 0)
    rd_expense = current_data.get('ResearchAndDevelopment', 0)
    sga_expense = current_data.get('SellingGeneralAndAdministration', 0)
    cost_of_revenue = current_data.get('CostOfRevenue', 0)
    
    if total_revenue > 0:
        # === PROFITABILITY MARGINS ===
        ratios['gross_margin'] = (gross_profit / total_revenue) * 100
        ratios['operating_margin'] = (operating_income / total_revenue) * 100
        ratios['net_margin'] = (net_income / total_revenue) * 100
        ratios['ebitda_margin'] = (ebitda / total_revenue) * 100
        
        # === EXPENSE RATIOS ===
        ratios['cost_ratio'] = (cost_of_revenue / total_revenue) * 100
        ratios['rd_intensity'] = (rd_expense / total_revenue) * 100
        ratios['sga_ratio'] = (sga_expense / total_revenue) * 100
        
        # === EFFICIENCY METRICS ===
        ratios['operating_efficiency'] = (operating_income / gross_profit) * 100 if gross_profit > 0 else 0
        ratios['expense_control'] = ((sga_expense + rd_expense) / total_revenue) * 100
        
    # === MULTI-YEAR GROWTH RATES ===
    if len(years) >= 2:
        for i in range(1, min(4, len(years))):  # Up to 3 years back
            prev_data = income_stmt_df.iloc[:, i]
            
            prev_revenue = prev_data.get('TotalRevenue', 0)
            prev_net_income = prev_data.get('NetIncome', 0)
            prev_eps = prev_data.get('DilutedEPS', 0)
            curr_eps = current_data.get('DilutedEPS', 0)
            
            year_suffix = f'_{i}y'
            
            # Growth calculations
            if prev_revenue > 0:
                ratios[f'revenue_growth{year_suffix}'] = ((total_revenue - prev_revenue) / prev_revenue) * 100
            
            if prev_net_income != 0:
                ratios[f'net_income_growth{year_suffix}'] = ((net_income - prev_net_income) / abs(prev_net_income)) * 100
            
            if prev_eps != 0:
                ratios[f'eps_growth{year_suffix}'] = ((curr_eps - prev_eps) / abs(prev_eps)) * 100
    
    # === TREND ANALYSIS ===
    if len(years) >= 3:
        # Calculate margin trends (improving vs declining)
        margins_3y = []
        revenues_3y = []
        
        for i in range(3):
            year_data = income_stmt_df.iloc[:, i]
            rev = year_data.get('TotalRevenue', 0)
            op_income = year_data.get('OperatingIncome', 0)
            
            if rev > 0:
                margins_3y.append((op_income / rev) * 100)
                revenues_3y.append(rev)
        
        if len(margins_3y) >= 3:
            # Margin trend (positive = improving)
            ratios['margin_trend'] = margins_3y[0] - margins_3y[2]  # Current vs 3 years ago
            ratios['margin_stability'] = np.std(margins_3y)  # Lower = more stable
            
        # Revenue growth acceleration/deceleration
        if len(revenues_3y) >= 3:
            growth_1y = (revenues_3y[0] - revenues_3y[1]) / revenues_3y[1] * 100
            growth_2y = (revenues_3y[1] - revenues_3y[2]) / revenues_3y[2] * 100
            ratios['growth_acceleration'] = growth_1y - growth_2y
    
    return ratios

# Integration with your existing feature engineering
def integrate_income_ratios(feature_dict, income_stmt_df):
    """
    Integrate income statement ratios into your existing feature dictionary
    """
    
    # Calculate all ratios
    income_ratios = calculate_comprehensive_income_ratios(income_stmt_df)
    
    # Add to existing features
    feature_dict.update(income_ratios)
    
    # Create composite scores
    feature_dict['profitability_composite'] = (
        income_ratios.get('gross_margin', 0) * 0.3 +
        income_ratios.get('operating_margin', 0) * 0.4 +
        income_ratios.get('net_margin', 0) * 0.3
    )
    
    feature_dict['growth_composite'] = (
        income_ratios.get('revenue_growth_1y', 0) * 0.5 +
        income_ratios.get('eps_growth_1y', 0) * 0.5
    )
    
    feature_dict['efficiency_composite'] = (
        income_ratios.get('operating_efficiency', 0) * 0.6 +
        (100 - income_ratios.get('expense_control', 100)) * 0.4  # Lower expense control is better
    )
    
    return feature_dict

# Example usage in your data pipeline
"""
# In your main data processing loop:
for ticker in tickers:
    # Get your data
    income_data = yf.Ticker(ticker).get_income_stmt()
    
    # Your existing features
    features = your_existing_feature_function(ticker_data)
    
    # Add income statement ratios
    features = integrate_income_ratios(features, income_data)
    
    # Now you have all the margin and growth ratios!
"""