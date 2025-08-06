import React, { useState } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const WhatIfScenarios = () => {
  const selectedTicker = useCryptoStore(state => state.selectedTicker);
  const whatIfResults = useCryptoStore(state => state.whatIfResults);
  const actions = useCryptoStore(state => state.actions);
  const loading = useCryptoStore(state => state.loading);
  
  // Debug: Log the whatIfResults directly
  React.useEffect(() => {
    console.log('WhatIfResults from selector:', whatIfResults);
    console.log('WhatIfResults type:', typeof whatIfResults);
    console.log('WhatIfResults keys:', whatIfResults ? Object.keys(whatIfResults) : 'null/undefined');
  }, [whatIfResults]);
  const [activeScenario, setActiveScenario] = useState('price-change');
  
  // Debug: Log store state changes
  React.useEffect(() => {
    console.log('Store whatIfResults changed:', whatIfResults);
  }, [whatIfResults]);
  
  // Price Change Scenario State
  const [priceChangeParams, setPriceChangeParams] = useState({
    ticker: selectedTicker,
    target_prices: '50000,55000,60000',
    quantities: '0.1,0.5,1.0'
  });
  
  // Trading Strategy Scenario State
  const [tradingParams, setTradingParams] = useState({
    ticker: selectedTicker,
    investment_amount: 10000,
    buy_price: '',
    sell_price: '',
    holding_period_days: 30,
    stop_loss_pct: 10,
    take_profit_pct: 20
  });
  
  // Portfolio Allocation State
  const [portfolioParams, setPortfolioParams] = useState({
    tickers: 'BTC,ETH,ADA,DOT',
    total_investment: 50000,
    allocations: '40,30,20,10',
    rebalance_period_days: 30
  });
  
  // DCA Strategy State
  const [dcaParams, setDcaParams] = useState({
    ticker: selectedTicker,
    periodic_investment: 1000,
    frequency_days: 7,
    total_periods: 52
  });
  
  const runScenario = async () => {
    try {
      let params = {};
      switch (activeScenario) {
        case 'price-change':
          params = {
            ...priceChangeParams,
            target_prices: priceChangeParams.target_prices, // Keep as comma-separated string
            quantities: priceChangeParams.quantities // Keep as comma-separated string
          };
          break;
        case 'trading-strategy':
          params = tradingParams;
          break;
        case 'portfolio':
          params = {
            ...portfolioParams,
            tickers: portfolioParams.tickers, // Keep as comma-separated string
            allocations: portfolioParams.allocations // Keep as comma-separated string
          };
          break;
        case 'dca':
          params = dcaParams;
          break;
      }
      
      console.log('Component: Calling runWhatIfScenario with:', { activeScenario, params });
      await actions.runWhatIfScenario(activeScenario, params);
    } catch (error) {
      console.error('Failed to run scenario:', error);
      // Show error message to user
      alert('Failed to run scenario analysis: ' + error.message);
    }
  };
  
  const results = whatIfResults?.[activeScenario] || null;
  
  // Debug logging
  React.useEffect(() => {
    console.log('Component Debug:', {
      activeScenario,
      whatIfResults,
      'whatIfResults[activeScenario]': whatIfResults?.[activeScenario],
      results,
      'results exists': !!results,
      'results.scenarios': results?.scenarios,
      'results.summary': results?.summary,
      loading
    });
  }, [results, activeScenario, whatIfResults, loading]);
  
  const prepareChartData = () => {
    if (!results) return null;
    
    switch (activeScenario) {
      case 'price-change':
        if (!results.scenarios || !Array.isArray(results.scenarios)) return null;
        return {
          series: [{
            name: 'P&L',
            data: results.scenarios.map(s => s.profit_loss || s.pnl || 0)
          }],
          options: {
            chart: { type: 'bar', height: 350 },
            xaxis: {
              categories: results.scenarios.map(s => `$${s.target_price || 0}`),
              title: { text: 'Target Price' }
            },
            yaxis: { title: { text: 'Profit/Loss ($)' } },
            colors: ['#00E396'],
            dataLabels: { enabled: true }
          }
        };
        
      case 'dca':
        if (!results.accumulation_data || !Array.isArray(results.accumulation_data)) return null;
        return {
          series: [
            {
              name: 'Total Investment',
              data: results.accumulation_data.map(d => ({
                x: new Date(d.date),
                y: d.total_invested || 0
              }))
            },
            {
              name: 'Portfolio Value',
              data: results.accumulation_data.map(d => ({
                x: new Date(d.date),
                y: d.portfolio_value || 0
              }))
            }
          ],
          options: {
            chart: { type: 'area', height: 350 },
            xaxis: { type: 'datetime' },
            yaxis: {
              title: { text: 'Value ($)' },
              labels: { formatter: (val) => `$${val.toFixed(0)}` }
            },
            stroke: { curve: 'smooth' },
            colors: ['#008FFB', '#00E396']
          }
        };
        
      case 'portfolio':
        if (!results.allocation_percentages || !Array.isArray(results.allocation_percentages)) return null;
        return {
          series: results.allocation_percentages,
          options: {
            chart: { type: 'donut', height: 350 },
            labels: results.tickers || [],
            colors: ['#008FFB', '#00E396', '#FEB019', '#FF4560', '#775DD0']
          }
        };
        
      default:
        return null;
    }
  };
  
  const chartData = prepareChartData();
  
  const scenarios = [
    { id: 'price-change', name: '💰 Price Change', icon: '📊' },
    { id: 'trading-strategy', name: '📈 Trading Strategy', icon: '🎯' },
    { id: 'portfolio', name: '💼 Portfolio Allocation', icon: '🏦' },
    { id: 'dca', name: '⏰ DCA Strategy', icon: '💵' }
  ];
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        🔮 What-If Scenarios Analysis
      </h2>
      
      {/* Scenario Tabs */}
      <div className="flex flex-wrap gap-2 mb-6 border-b">
        {scenarios.map(scenario => (
          <button
            key={scenario.id}
            onClick={() => setActiveScenario(scenario.id)}
            className={`
              px-4 py-2 font-medium transition-all
              ${activeScenario === scenario.id
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
              }
            `}
          >
            <span className="mr-2">{scenario.icon}</span>
            {scenario.name}
          </button>
        ))}
      </div>
      
      {/* Scenario Input Forms */}
      <div className="mb-6">
        {activeScenario === 'price-change' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Ticker Symbol
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={priceChangeParams.ticker}
                onChange={(e) => setPriceChangeParams({...priceChangeParams, ticker: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Target Prices (comma-separated)
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={priceChangeParams.target_prices}
                onChange={(e) => setPriceChangeParams({...priceChangeParams, target_prices: e.target.value})}
                placeholder="50000,55000,60000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Quantities (comma-separated)
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={priceChangeParams.quantities}
                onChange={(e) => setPriceChangeParams({...priceChangeParams, quantities: e.target.value})}
                placeholder="0.1,0.5,1.0"
              />
            </div>
          </div>
        )}
        
        {activeScenario === 'trading-strategy' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Investment Amount ($)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={tradingParams.investment_amount}
                onChange={(e) => setTradingParams({...tradingParams, investment_amount: parseFloat(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Stop Loss (%)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={tradingParams.stop_loss_pct}
                onChange={(e) => setTradingParams({...tradingParams, stop_loss_pct: parseFloat(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Take Profit (%)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={tradingParams.take_profit_pct}
                onChange={(e) => setTradingParams({...tradingParams, take_profit_pct: parseFloat(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Holding Period (days)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={tradingParams.holding_period_days}
                onChange={(e) => setTradingParams({...tradingParams, holding_period_days: parseInt(e.target.value)})}
              />
            </div>
          </div>
        )}
        
        {activeScenario === 'portfolio' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tickers (comma-separated)
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={portfolioParams.tickers}
                onChange={(e) => setPortfolioParams({...portfolioParams, tickers: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Allocations % (comma-separated)
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={portfolioParams.allocations}
                onChange={(e) => setPortfolioParams({...portfolioParams, allocations: e.target.value})}
                placeholder="40,30,20,10"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Total Investment ($)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={portfolioParams.total_investment}
                onChange={(e) => setPortfolioParams({...portfolioParams, total_investment: parseFloat(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Rebalance Period (days)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={portfolioParams.rebalance_period_days}
                onChange={(e) => setPortfolioParams({...portfolioParams, rebalance_period_days: parseInt(e.target.value)})}
              />
            </div>
          </div>
        )}
        
        {activeScenario === 'dca' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Ticker Symbol
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={dcaParams.ticker}
                onChange={(e) => setDcaParams({...dcaParams, ticker: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Periodic Investment ($)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={dcaParams.periodic_investment}
                onChange={(e) => setDcaParams({...dcaParams, periodic_investment: parseFloat(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Frequency (days)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={dcaParams.frequency_days}
                onChange={(e) => setDcaParams({...dcaParams, frequency_days: parseInt(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Total Periods
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={dcaParams.total_periods}
                onChange={(e) => setDcaParams({...dcaParams, total_periods: parseInt(e.target.value)})}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* Run Scenario Button */}
      <div className="mb-6">
        <button
          onClick={runScenario}
          disabled={loading}
          className={`
            px-6 py-3 rounded-md font-medium transition-colors
            ${loading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
            }
          `}
        >
          {loading ? 'Analyzing...' : 'Run Scenario Analysis'}
        </button>
      </div>
      
      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            Analyzing Scenarios...
          </h3>
          <p className="text-gray-500">
            Please wait while we process your what-if analysis
          </p>
        </div>
      )}
      
      {/* Results Display */}
      {!results && !loading && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔮</div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            No Analysis Results Yet
          </h3>
          <p className="text-gray-500 mb-4">
            Configure your parameters above and click "Run Scenario Analysis" to see results
          </p>
        </div>
      )}
      
      {results && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {results.summary && Object.entries(results.summary).slice(0, 3).map(([key, value]) => (
              <div key={key} className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600 capitalize mb-1">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-2xl font-bold text-gray-800">
                  {typeof value === 'number' 
                    ? value > 1000 
                      ? `$${value.toFixed(0)}` 
                      : value.toFixed(2)
                    : value}
                </p>
              </div>
            ))}
          </div>
          
          {/* Chart Display */}
          {chartData && (
            <div className="mb-6">
              <ReactApexChart
                options={chartData.options}
                series={chartData.series}
                type={chartData.options.chart.type}
                height={350}
              />
            </div>
          )}
          
          {/* Detailed Results Table */}
          {results.scenarios && Array.isArray(results.scenarios) && results.scenarios.length > 0 && (
            <div className="overflow-x-auto">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">Scenario Details</h3>
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(results.scenarios[0]).map(key => (
                      <th key={key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {key.replace(/_/g, ' ')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {results.scenarios.map((scenario, idx) => (
                    <tr key={idx}>
                      {Object.values(scenario).map((value, vidx) => (
                        <td key={vidx} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {typeof value === 'number' 
                            ? value > 1000 
                              ? `$${value.toFixed(0)}` 
                              : value.toFixed(2)
                            : value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          {/* Recommendations */}
          {results.recommendations && Array.isArray(results.recommendations) && results.recommendations.length > 0 && (
            <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
              <h3 className="font-semibold text-blue-900 mb-2">Recommendations</h3>
              <ul className="list-disc list-inside text-blue-700 space-y-1">
                {results.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default WhatIfScenarios;
