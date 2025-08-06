import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const TradingSignals = () => {
  const { selectedTicker, signals, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState('week');
  const [dataSource, setDataSource] = useState('yfinance');
  const [modelType, setModelType] = useState('arima');
  
  useEffect(() => {
    loadTradingSignals();
  }, [selectedTicker, dataSource, modelType, timeframe]);
  
  // Convert timeframe to periods
  const getPeriods = (timeframe) => {
    switch(timeframe) {
      case 'week': return 7;
      case '2weeks': return 14; 
      case 'month': return 30;
      default: return 7;
    }
  };
  
  const loadTradingSignals = async () => {
    if (!selectedTicker) return;
    
    setLoading(true);
    try {
      const periods = getPeriods(timeframe);
      await actions.fetchSignals(selectedTicker, modelType, periods, dataSource);
    } catch (error) {
      console.error('Failed to load trading signals:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const signalData = signals[selectedTicker] || null;
  
  // Debug logging
  React.useEffect(() => {
    if (signalData) {
      console.log('SignalData structure:', signalData);
      console.log('SignalData.signals type:', typeof signalData.signals, signalData.signals);
      console.log('SignalData.price_data type:', typeof signalData.price_data, signalData.price_data);
    }
  }, [signalData]);
  
  const prepareSignalChart = () => {
    if (!signalData?.price_data || !signalData?.signals || 
        !Array.isArray(signalData.signals) || !Array.isArray(signalData.price_data)) return null;
    
    const buySignals = [];
    const sellSignals = [];
    
    // Process signals - these are for future dates from raw_signals
    if (signalData.raw_signals && signalData.price_data.length > 0) {
      const lastPrice = signalData.price_data[signalData.price_data.length - 1].price;
      
      Object.entries(signalData.raw_signals).forEach(([date, signal]) => {
        const signalPrice = lastPrice * (1 + Math.random() * 0.02 - 0.01); // Slight price variation
        
        if (signal === 'BUY') {
          buySignals.push({
            x: new Date(date),
            y: signalPrice
          });
        } else if (signal === 'SELL') {
          sellSignals.push({
            x: new Date(date),
            y: signalPrice
          });
        }
      });
    }
    
    return {
      series: [
        {
          name: 'Price',
          type: 'line',
          data: signalData.price_data.map(d => ({
            x: new Date(d.date),
            y: d.price
          }))
        },
        {
          name: 'Buy Signal',
          type: 'scatter',
          data: buySignals
        },
        {
          name: 'Sell Signal',
          type: 'scatter',
          data: sellSignals
        }
      ],
      options: {
        chart: {
          height: 400,
          type: 'line',
          toolbar: { show: true }
        },
        stroke: {
          width: [2, 0, 0],
          curve: 'smooth'
        },
        markers: {
          size: [0, 8, 8],
          shape: ['circle', 'triangle', 'inverted-triangle']
        },
        xaxis: {
          type: 'datetime'
        },
        yaxis: {
          title: { text: 'Price' },
          labels: {
            formatter: (val) => val?.toFixed(2)
          }
        },
        colors: ['#008FFB', '#00E396', '#FF4560'],
        legend: {
          position: 'top'
        },
        title: {
          text: 'Trading Signals',
          align: 'left'
        }
      }
    };
  };
  
  const signalChart = React.useMemo(() => {
    try {
      return prepareSignalChart();
    } catch (error) {
      console.error('Error preparing signal chart:', error);
      return null;
    }
  }, [signalData]);
  
  const getSignalIcon = (signal) => {
    if (signal === 'BUY') {
      return (
        <svg className="w-6 h-6 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
        </svg>
      );
    } else if (signal === 'SELL') {
      return (
        <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    }
    return (
      <svg className="w-6 h-6 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
      </svg>
    );
  };
  
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        📈 Trading Signals & Predictions - {selectedTicker}
      </h2>
      
      {/* Controls Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Data Source Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Data Source
          </label>
          <select
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="yfinance">Yahoo Finance</option>
            <option value="binance">Binance</option>
          </select>
        </div>

        {/* Model Type Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Model Type
          </label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="arima">ARIMA</option>
            <option value="sarima">SARIMA</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
            <option value="lstm">LSTM</option>
          </select>
        </div>

        {/* Timeframe Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Timeframe
          </label>
          <div className="flex gap-2">
            {['week', '2weeks', 'month'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${timeframe === tf
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }
                `}
              >
                {tf === 'week' ? '1W' : tf === '2weeks' ? '2W' : '1M'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Status Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-6">
        <div className="flex items-center justify-between text-sm">
          <span className="text-blue-700">
            <span className="font-medium">Active Configuration:</span> {dataSource === 'yfinance' ? 'Yahoo Finance' : 'Binance'} • {modelType.toUpperCase()} Model
          </span>
          <div className="flex items-center gap-3">
            <span className="text-blue-600 text-xs">
              {signalData?.from_cache ? '📊 Cached Data' : '🔄 Live Data'}
            </span>
            <button
              onClick={loadTradingSignals}
              disabled={loading}
              className="text-blue-600 hover:text-blue-800 disabled:opacity-50 transition-colors"
              title="Refresh signals"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Generating trading signals...</p>
          </div>
        </div>
      )}
      
      {!loading && signalData && signalData.current_signal !== undefined && (
        <>
          {/* Current Recommendation */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className={`
              p-4 rounded-lg border-2
              ${signalData.current_signal === 'BUY' ? 'border-green-500 bg-green-50' :
                signalData.current_signal === 'SELL' ? 'border-red-500 bg-red-50' :
                'border-gray-300 bg-gray-50'}
            `}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-700">Current Signal</h3>
                {getSignalIcon(signalData.current_signal)}
              </div>
              <p className={`text-2xl font-bold
                ${signalData.current_signal === 'BUY' ? 'text-green-600' :
                  signalData.current_signal === 'SELL' ? 'text-red-600' :
                  'text-gray-600'}
              `}>
                {signalData.current_signal || 'HOLD'}
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Confidence: <span className={`font-semibold ${getConfidenceColor(signalData.confidence)}`}>
                  {(signalData.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>
            
            <div className="p-4 rounded-lg border border-gray-200">
              <h3 className="font-semibold text-gray-700 mb-2">Expected Return</h3>
              <p className={`text-2xl font-bold ${
                signalData.expected_return > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {signalData.expected_return > 0 ? '+' : ''}{signalData.expected_return?.toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Target: ${signalData.target_price?.toFixed(2)}
              </p>
            </div>
            
            <div className="p-4 rounded-lg border border-gray-200">
              <h3 className="font-semibold text-gray-700 mb-2">Risk Level</h3>
              <div className="flex items-center">
                <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                  <div
                    className={`h-3 rounded-full ${
                      signalData.risk_level === 'High' ? 'bg-red-500' :
                      signalData.risk_level === 'Medium' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${signalData.risk_score * 100}%` }}
                  />
                </div>
                <span className="font-semibold text-gray-700">
                  {signalData.risk_level}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">
                Stop Loss: ${signalData.stop_loss?.toFixed(2)}
              </p>
            </div>
          </div>
          
          {/* Technical Indicators */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">RSI (14)</p>
              <p className="text-lg font-semibold">
                {signalData.indicators?.rsi?.toFixed(2) || 'N/A'}
              </p>
              <p className="text-xs text-gray-500">
                {signalData.indicators?.rsi > 70 ? 'Overbought' :
                 signalData.indicators?.rsi < 30 ? 'Oversold' : 'Neutral'}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">MACD</p>
              <p className="text-lg font-semibold">
                {signalData.indicators?.macd?.signal > 0 ? 'Bullish' : 'Bearish'}
              </p>
              <p className="text-xs text-gray-500">
                Signal: {signalData.indicators?.macd?.signal?.toFixed(4)}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Moving Avg</p>
              <p className="text-lg font-semibold">
                {signalData.indicators?.ma_trend || 'N/A'}
              </p>
              <p className="text-xs text-gray-500">
                MA20: ${signalData.indicators?.ma20?.toFixed(2)}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Volume</p>
              <p className="text-lg font-semibold">
                {signalData.indicators?.volume_trend || 'Normal'}
              </p>
              <p className="text-xs text-gray-500">
                {signalData.indicators?.volume_change > 0 ? '+' : ''}
                {signalData.indicators?.volume_change?.toFixed(1)}%
              </p>
            </div>
          </div>
          
          {/* Signal Chart */}
          {signalChart && (
            <div className="mb-6">
              <ReactApexChart
                options={signalChart.options}
                series={signalChart.series}
                type="line"
                height={400}
              />
            </div>
          )}
          
          {/* Prediction Table */}
          <div className="overflow-x-auto">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">Future Predictions</h3>
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Period
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predicted Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Change
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Signal
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {signalData.predictions?.map((pred, idx) => (
                  <tr key={idx}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {pred.period}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      ${pred.price.toFixed(2)}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                      pred.change > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {pred.change > 0 ? '+' : ''}{pred.change.toFixed(2)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        pred.signal === 'BUY' ? 'bg-green-100 text-green-800' :
                        pred.signal === 'SELL' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {pred.signal}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {(pred.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      
      {!loading && !signalData && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-gray-600">No trading signals available</p>
            <p className="text-gray-500 text-sm">Select a ticker to generate signals</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingSignals;
