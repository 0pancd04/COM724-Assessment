import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const TradingSignals = () => {
  const { selectedTicker, tradingSignals, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState('week');
  
  useEffect(() => {
    loadTradingSignals();
  }, [selectedTicker]);
  
  const loadTradingSignals = async () => {
    if (!selectedTicker) return;
    
    setLoading(true);
    try {
      await actions.fetchTradingSignals(selectedTicker);
    } catch (error) {
      console.error('Failed to load trading signals:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const signals = tradingSignals[selectedTicker];
  
  const prepareSignalChart = () => {
    if (!signals?.price_data || !signals?.signals) return null;
    
    const buySignals = [];
    const sellSignals = [];
    
    signals.signals.forEach((signal, idx) => {
      if (signal === 1) {
        buySignals.push({
          x: new Date(signals.price_data[idx].date),
          y: signals.price_data[idx].price
        });
      } else if (signal === -1) {
        sellSignals.push({
          x: new Date(signals.price_data[idx].date),
          y: signals.price_data[idx].price
        });
      }
    });
    
    return {
      series: [
        {
          name: 'Price',
          type: 'line',
          data: signals.price_data.map(d => ({
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
  
  const signalChart = prepareSignalChart();
  
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
      
      {/* Timeframe Selector */}
      <div className="flex gap-2 mb-6">
        {['week', '2weeks', 'month'].map(tf => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            className={`
              px-4 py-2 rounded-md font-medium transition-colors
              ${timeframe === tf
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }
            `}
          >
            {tf === 'week' ? '1 Week' : tf === '2weeks' ? '2 Weeks' : '1 Month'}
          </button>
        ))}
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Generating trading signals...</p>
          </div>
        </div>
      )}
      
      {!loading && signals && (
        <>
          {/* Current Recommendation */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className={`
              p-4 rounded-lg border-2
              ${signals.current_signal === 'BUY' ? 'border-green-500 bg-green-50' :
                signals.current_signal === 'SELL' ? 'border-red-500 bg-red-50' :
                'border-gray-300 bg-gray-50'}
            `}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-700">Current Signal</h3>
                {getSignalIcon(signals.current_signal)}
              </div>
              <p className={`text-2xl font-bold
                ${signals.current_signal === 'BUY' ? 'text-green-600' :
                  signals.current_signal === 'SELL' ? 'text-red-600' :
                  'text-gray-600'}
              `}>
                {signals.current_signal || 'HOLD'}
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Confidence: <span className={`font-semibold ${getConfidenceColor(signals.confidence)}`}>
                  {(signals.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>
            
            <div className="p-4 rounded-lg border border-gray-200">
              <h3 className="font-semibold text-gray-700 mb-2">Expected Return</h3>
              <p className={`text-2xl font-bold ${
                signals.expected_return > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {signals.expected_return > 0 ? '+' : ''}{signals.expected_return?.toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Target: ${signals.target_price?.toFixed(2)}
              </p>
            </div>
            
            <div className="p-4 rounded-lg border border-gray-200">
              <h3 className="font-semibold text-gray-700 mb-2">Risk Level</h3>
              <div className="flex items-center">
                <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                  <div
                    className={`h-3 rounded-full ${
                      signals.risk_level === 'High' ? 'bg-red-500' :
                      signals.risk_level === 'Medium' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${signals.risk_score * 100}%` }}
                  />
                </div>
                <span className="font-semibold text-gray-700">
                  {signals.risk_level}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">
                Stop Loss: ${signals.stop_loss?.toFixed(2)}
              </p>
            </div>
          </div>
          
          {/* Technical Indicators */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">RSI (14)</p>
              <p className="text-lg font-semibold">
                {signals.indicators?.rsi?.toFixed(2) || 'N/A'}
              </p>
              <p className="text-xs text-gray-500">
                {signals.indicators?.rsi > 70 ? 'Overbought' :
                 signals.indicators?.rsi < 30 ? 'Oversold' : 'Neutral'}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">MACD</p>
              <p className="text-lg font-semibold">
                {signals.indicators?.macd?.signal > 0 ? 'Bullish' : 'Bearish'}
              </p>
              <p className="text-xs text-gray-500">
                Signal: {signals.indicators?.macd?.signal?.toFixed(4)}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Moving Avg</p>
              <p className="text-lg font-semibold">
                {signals.indicators?.ma_trend || 'N/A'}
              </p>
              <p className="text-xs text-gray-500">
                MA20: ${signals.indicators?.ma20?.toFixed(2)}
              </p>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Volume</p>
              <p className="text-lg font-semibold">
                {signals.indicators?.volume_trend || 'Normal'}
              </p>
              <p className="text-xs text-gray-500">
                {signals.indicators?.volume_change > 0 ? '+' : ''}
                {signals.indicators?.volume_change?.toFixed(1)}%
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
                {signals.predictions?.map((pred, idx) => (
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
    </div>
  );
};

export default TradingSignals;
