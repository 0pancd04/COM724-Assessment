"""
What-If Scenario Analysis Module
Allows users to test different trading scenarios and see potential outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .logger import setup_logger
from .database import crypto_db
from .analysis_storage import analysis_storage
from .model_comparison import model_comparison

# Setup logging
logger = setup_logger("whatif_scenarios", "logs/whatif_scenarios.log")

class WhatIfAnalyzer:
    """Handles what-if scenario analysis for cryptocurrency trading"""
    
    def __init__(self):
        self.db = crypto_db
        self.storage = analysis_storage
    
    def analyze_price_change(self, ticker: str, 
                            current_price: Optional[float] = None,
                            target_prices: List[float] = None,
                            quantities: List[float] = None) -> Dict:
        """
        Analyze profit/loss for different price scenarios
        
        Args:
            ticker: Cryptocurrency ticker
            current_price: Current price (if None, fetches latest)
            target_prices: List of target prices to analyze
            quantities: List of quantities to analyze
        
        Returns:
            Scenario analysis results
        """
        try:
            # Get current price if not provided
            if current_price is None:
                df = self.db.get_ohlcv_data(ticker, interval="1d")
                if df.empty:
                    return {"error": f"No data available for {ticker}"}
                current_price = df['Close'].iloc[-1]
            
            # Ensure current_price is positive
            if current_price <= 0:
                return {"error": "Current price must be greater than 0"}
            
            # Default scenarios if not provided
            if target_prices is None:
                # Generate scenarios: -50%, -20%, -10%, +10%, +20%, +50%, +100%
                target_prices = [
                    current_price * 0.5,
                    current_price * 0.8,
                    current_price * 0.9,
                    current_price * 1.1,
                    current_price * 1.2,
                    current_price * 1.5,
                    current_price * 2.0
                ]
            
            if quantities is None:
                quantities = [1, 10, 100, 1000]
            
            scenarios = []
            
            for quantity in quantities:
                for target_price in target_prices:
                    investment = current_price * quantity if quantity > 0 else 0
                    future_value = target_price * quantity if quantity > 0 else 0
                    profit_loss = future_value - investment
                    profit_loss_pct = ((profit_loss / investment) * 100) if investment > 0 else 0
                    
                    scenario = {
                        "ticker": ticker,
                        "quantity": quantity,
                        "current_price": current_price,
                        "target_price": target_price,
                        "price_change_pct": ((target_price - current_price) / current_price) * 100 if current_price > 0 else 0,
                        "investment": investment,
                        "future_value": future_value,
                        "profit_loss": profit_loss,
                        "profit_loss_pct": profit_loss_pct
                    }
                    scenarios.append(scenario)
            
            if not scenarios:
                return {"error": "No valid scenarios could be generated"}
            
            result = {
                "ticker": ticker,
                "current_price": current_price,
                "scenarios": scenarios,
                "best_scenario": max(scenarios, key=lambda x: x["profit_loss"]) if scenarios else None,
                "worst_scenario": min(scenarios, key=lambda x: x["profit_loss"]) if scenarios else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in database
            self.storage.store_whatif_scenario(
                ticker=ticker,
                scenario_name="price_change_analysis",
                parameters={
                    "current_price": current_price,
                    "target_prices": target_prices,
                    "quantities": quantities
                },
                results=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in price change analysis: {e}")
            return {"error": str(e)}
    
    def analyze_trading_strategy(self, ticker: str,
                                investment_amount: float,
                                buy_price: Optional[float] = None,
                                sell_price: Optional[float] = None,
                                holding_period_days: int = 30,
                                stop_loss_pct: float = 10,
                                take_profit_pct: float = 20) -> Dict:
        """
        Analyze a specific trading strategy
        
        Args:
            ticker: Cryptocurrency ticker
            investment_amount: Amount to invest
            buy_price: Entry price (if None, uses current)
            sell_price: Exit price (if None, predicts)
            holding_period_days: How long to hold
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        
        Returns:
            Trading strategy analysis
        """
        try:
            # Validate input parameters
            if investment_amount <= 0:
                return {"error": "Investment amount must be greater than 0"}
            
            # Get current/historical data
            df = self.db.get_ohlcv_data(ticker, interval="1d")
            if df.empty:
                return {"error": f"No data available for {ticker}"}
            
            current_price = df['Close'].iloc[-1]
            
            if buy_price is None:
                buy_price = current_price
            
            # Validate buy price
            if buy_price <= 0:
                return {"error": "Buy price must be greater than 0"}
            
            # Calculate quantities
            quantity = investment_amount / buy_price if buy_price > 0 else 0
            
            # Calculate stop loss and take profit levels
            stop_loss_price = buy_price * (1 - (stop_loss_pct / 100)) if stop_loss_pct > 0 else 0
            take_profit_price = buy_price * (1 + (take_profit_pct / 100)) if take_profit_pct > 0 else float('inf')
            
            # If sell price not provided, use prediction
            if sell_price is None:
                # Try to get prediction from models
                try:
                    from .forecasting import load_model, forecast_arima
                    model = load_model(ticker, "arima")
                    forecast = forecast_arima(model, holding_period_days)
                    sell_price = forecast.iloc[-1]
                except:
                    # Fallback to simple trend estimation
                    returns = df['Close'].pct_change().dropna()
                    avg_daily_return = returns.mean() if len(returns) > 0 else 0
                    expected_return = (1 + avg_daily_return) ** holding_period_days - 1
                    sell_price = buy_price * (1 + expected_return)
            
            # Calculate outcomes
            outcomes = []
            
            # Scenario 1: Hit stop loss
            if sell_price <= stop_loss_price:
                final_price = stop_loss_price
                outcome = "Stop Loss Hit"
            # Scenario 2: Hit take profit
            elif sell_price >= take_profit_price:
                final_price = take_profit_price
                outcome = "Take Profit Hit"
            # Scenario 3: Normal exit
            else:
                final_price = sell_price
                outcome = "Normal Exit"
            
            exit_value = quantity * final_price if quantity > 0 and final_price > 0 else 0
            profit_loss = exit_value - investment_amount
            roi = (profit_loss / investment_amount) * 100 if investment_amount > 0 else 0
            
            # Calculate risk metrics
            max_loss = investment_amount * (stop_loss_pct / 100) if stop_loss_pct > 0 else 0
            max_gain = investment_amount * (take_profit_pct / 100) if take_profit_pct > 0 else 0
            risk_reward_ratio = max_gain / max_loss if max_loss > 0 else float('inf')
            
            result = {
                "ticker": ticker,
                "strategy": {
                    "investment_amount": investment_amount,
                    "buy_price": buy_price,
                    "quantity": quantity,
                    "holding_period_days": holding_period_days,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price
                },
                "predicted_outcome": {
                    "sell_price": sell_price,
                    "final_price": final_price,
                    "outcome": outcome,
                    "exit_value": exit_value,
                    "profit_loss": profit_loss,
                    "roi_pct": roi
                },
                "risk_metrics": {
                    "max_loss": max_loss,
                    "max_gain": max_gain,
                    "risk_reward_ratio": risk_reward_ratio
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in database
            self.storage.store_whatif_scenario(
                ticker=ticker,
                scenario_name="trading_strategy",
                parameters={
                    "investment_amount": investment_amount,
                    "buy_price": buy_price,
                    "holding_period_days": holding_period_days,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct
                },
                results=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trading strategy analysis: {e}")
            return {"error": str(e)}
    
    def analyze_portfolio_allocation(self, tickers: List[str],
                                    total_investment: float,
                                    allocations: Optional[Dict[str, float]] = None,
                                    rebalance_period_days: int = 30) -> Dict:
        """
        Analyze portfolio allocation scenarios
        
        Args:
            tickers: List of cryptocurrency tickers
            total_investment: Total amount to invest
            allocations: Dict of ticker -> percentage allocation
            rebalance_period_days: How often to rebalance
        
        Returns:
            Portfolio analysis results
        """
        try:
            # Validate input parameters
            if total_investment <= 0:
                return {"error": "Total investment must be greater than 0"}
            
            if not tickers:
                return {"error": "At least one ticker must be provided"}
            
            # Default equal allocation if not provided
            if allocations is None:
                equal_alloc = 100.0 / len(tickers) if len(tickers) > 0 else 0
                allocations = {ticker: equal_alloc for ticker in tickers}
            
            # Validate allocations sum to 100
            total_alloc = sum(allocations.values()) if allocations else 0
            if abs(total_alloc - 100) > 0.01:
                return {"error": f"Allocations must sum to 100%, got {total_alloc}%"}
            
            portfolio = []
            total_current_value = 0
            total_predicted_value = 0
            
            for ticker in tickers:
                allocation_pct = allocations.get(ticker, 0)
                investment = total_investment * (allocation_pct / 100) if allocation_pct > 0 else 0
                
                # Get current price
                df = self.db.get_ohlcv_data(ticker, interval="1d")
                if df.empty:
                    continue
                
                current_price = df['Close'].iloc[-1]
                if current_price <= 0:
                    logger.warning(f"Invalid price for {ticker}: {current_price}")
                    continue
                
                quantity = investment / current_price if current_price > 0 else 0
                
                # Calculate historical volatility
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
                
                # Simple return prediction based on historical average
                avg_return = returns.mean() * rebalance_period_days if len(returns) > 0 else 0
                predicted_price = current_price * (1 + avg_return)
                predicted_value = quantity * predicted_price if quantity > 0 and predicted_price > 0 else 0
                
                position = {
                    "ticker": ticker,
                    "allocation_pct": allocation_pct,
                    "investment": investment,
                    "current_price": current_price,
                    "quantity": quantity,
                    "current_value": investment,
                    "predicted_price": predicted_price,
                    "predicted_value": predicted_value,
                    "expected_return": avg_return * 100,
                    "volatility": volatility * 100
                }
                
                portfolio.append(position)
                total_current_value += investment
                total_predicted_value += predicted_value
            
            # Calculate portfolio metrics
            portfolio_return = ((total_predicted_value - total_current_value) / 
                              total_current_value) * 100 if total_current_value > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            risk_free_rate = 0.02  # 2% annual
            portfolio_volatility = np.sqrt(sum(
                (p["allocation_pct"]/100)**2 * (p["volatility"]/100)**2 
                for p in portfolio
            )) * 100 if portfolio else 0
            
            sharpe_ratio = ((portfolio_return - risk_free_rate) / portfolio_volatility 
                          if portfolio_volatility > 0 else 0)
            
            result = {
                "portfolio": portfolio,
                "summary": {
                    "total_investment": total_investment,
                    "total_current_value": total_current_value,
                    "total_predicted_value": total_predicted_value,
                    "expected_return_pct": portfolio_return,
                    "portfolio_volatility_pct": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "rebalance_period_days": rebalance_period_days
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in database
            self.storage.store_whatif_scenario(
                ticker=','.join(tickers),
                scenario_name="portfolio_allocation",
                parameters={
                    "tickers": tickers,
                    "total_investment": total_investment,
                    "allocations": allocations,
                    "rebalance_period_days": rebalance_period_days
                },
                results=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio allocation analysis: {e}")
            return {"error": str(e)}
    
    def analyze_dca_strategy(self, ticker: str,
                            periodic_investment: float,
                            frequency_days: int = 7,
                            total_periods: int = 52) -> Dict:
        """
        Analyze Dollar Cost Averaging (DCA) strategy
        
        Args:
            ticker: Cryptocurrency ticker
            periodic_investment: Amount to invest each period
            frequency_days: Days between investments
            total_periods: Number of investment periods
        
        Returns:
            DCA strategy analysis
        """
        try:
            # Validate input parameters
            if periodic_investment <= 0:
                return {"error": "Periodic investment must be greater than 0"}
            
            if frequency_days <= 0:
                return {"error": "Frequency days must be greater than 0"}
            
            if total_periods <= 0:
                return {"error": "Total periods must be greater than 0"}
            
            # Get historical data
            df = self.db.get_ohlcv_data(ticker, interval="1d")
            if df.empty:
                return {"error": f"No data available for {ticker}"}
            
            # Simulate DCA strategy on historical data
            total_investment = periodic_investment * total_periods
            
            # Use last N periods of historical data
            required_days = frequency_days * total_periods
            if len(df) < required_days:
                return {"error": f"Insufficient historical data for {total_periods} periods"}
            
            historical_df = df.tail(required_days)
            
            # Simulate purchases
            purchases = []
            total_quantity = 0
            
            for i in range(0, required_days, frequency_days):
                if i < len(historical_df):
                    price = historical_df['Close'].iloc[i]
                    if price > 0:
                        quantity = periodic_investment / price
                        total_quantity += quantity
                        
                        purchases.append({
                            "period": len(purchases) + 1,
                            "date": historical_df.index[i].strftime("%Y-%m-%d"),
                            "price": price,
                            "investment": periodic_investment,
                            "quantity": quantity,
                            "cumulative_investment": periodic_investment * (len(purchases) + 1),
                            "cumulative_quantity": total_quantity
                        })
            
            if not purchases:
                return {"error": "No valid purchases could be simulated"}
            
            # Calculate final value
            final_price = df['Close'].iloc[-1]
            final_value = total_quantity * final_price if final_price > 0 else 0
            
            # Calculate average purchase price
            avg_purchase_price = total_investment / total_quantity if total_quantity > 0 else 0
            
            # Compare with lump sum investment
            lump_sum_price = historical_df['Close'].iloc[0]
            lump_sum_quantity = total_investment / lump_sum_price if lump_sum_price > 0 else 0
            lump_sum_value = lump_sum_quantity * final_price if final_price > 0 else 0
            
            result = {
                "ticker": ticker,
                "strategy": {
                    "periodic_investment": periodic_investment,
                    "frequency_days": frequency_days,
                    "total_periods": total_periods,
                    "total_investment": total_investment
                },
                "dca_results": {
                    "purchases": purchases[:10],  # Show first 10 for brevity
                    "total_quantity": total_quantity,
                    "avg_purchase_price": avg_purchase_price,
                    "final_value": final_value,
                    "profit_loss": final_value - total_investment,
                    "roi_pct": ((final_value - total_investment) / total_investment) * 100 if total_investment > 0 else 0
                },
                "lump_sum_comparison": {
                    "purchase_price": lump_sum_price,
                    "quantity": lump_sum_quantity,
                    "final_value": lump_sum_value,
                    "profit_loss": lump_sum_value - total_investment,
                    "roi_pct": ((lump_sum_value - total_investment) / total_investment) * 100 if total_investment > 0 else 0
                },
                "advantage": "DCA" if final_value > lump_sum_value else "Lump Sum",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in database
            self.storage.store_whatif_scenario(
                ticker=ticker,
                scenario_name="dca_strategy",
                parameters={
                    "periodic_investment": periodic_investment,
                    "frequency_days": frequency_days,
                    "total_periods": total_periods
                },
                results=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in DCA strategy analysis: {e}")
            return {"error": str(e)}

# Initialize what-if analyzer
whatif_analyzer = WhatIfAnalyzer()