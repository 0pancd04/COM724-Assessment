import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const CorrelationAnalysis = () => {
  const { selectedTicker, correlationResults, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [selectedTickers, setSelectedTickers] = useState(['BTC', 'ETH', 'ADA', 'DOT']);
  const [feature, setFeature] = useState('Close');
  
  useEffect(() => {
    loadCorrelation();
  }, [selectedTickers, feature]);
  
  const loadCorrelation = async () => {
    setLoading(true);
    try {
      await actions.fetchCorrelationAnalysis({
        tickers: selectedTickers.join(','),
        feature: feature
      });
    } catch (error) {
      console.error('Failed to load correlation analysis:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const prepareHeatmap = () => {
    if (!correlationResults?.correlation_matrix) return null;
    
    const matrix = correlationResults.correlation_matrix;
    const tickers = correlationResults.tickers || selectedTickers;
    
    const series = tickers.map((ticker, i) => ({
      name: ticker,
      data: tickers.map((_, j) => ({
        x: tickers[j],
        y: matrix[i]?.[j] || 0
      }))
    }));
    
    return {
      series,
      options: {
        chart: {
          type: 'heatmap',
          height: 400,
          toolbar: { show: true }
        },
        dataLabels: {
          enabled: true,
          formatter: (val) => val?.toFixed(2)
        },
        colors: ['#FF4560'],
        title: {
          text: 'Correlation Matrix',
          align: 'left'
        },
        xaxis: {
          type: 'category',
          categories: tickers
        },
        plotOptions: {
          heatmap: {
            colorScale: {
              ranges: [
                { from: -1, to: -0.5, color: '#FF4560', name: 'Strong Negative' },
                { from: -0.5, to: 0, color: '#FEB019', name: 'Weak Negative' },
                { from: 0, to: 0.5, color: '#00E396', name: 'Weak Positive' },
                { from: 0.5, to: 1, color: '#008FFB', name: 'Strong Positive' }
              ]
            }
          }
        }
      }
    };
  };
  
  const heatmapData = prepareHeatmap();
  
  const getCorrelationStrength = (value) => {
    const absValue = Math.abs(value);
    if (absValue >= 0.7) return 'Strong';
    if (absValue >= 0.4) return 'Moderate';
    return 'Weak';
  };
  
  const getCorrelationColor = (value) => {
    if (value >= 0.7) return 'text-green-600';
    if (value >= 0.4) return 'text-blue-600';
    if (value >= 0) return 'text-gray-600';
    if (value >= -0.4) return 'text-orange-600';
    return 'text-red-600';
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        🔗 Correlation Analysis
      </h2>
      
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Select Cryptocurrencies (comma-separated)
          </label>
          <input
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={selectedTickers.join(',')}
            onChange={(e) => setSelectedTickers(e.target.value.split(',').map(t => t.trim()))}
            placeholder="BTC,ETH,ADA,DOT"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Feature
          </label>
          <select
            className="w-full px-3 py-2 glass-dropdown"
            value={feature}
            onChange={(e) => setFeature(e.target.value)}
          >
            <option value="Close">Close Price</option>
            <option value="Volume">Volume</option>
            <option value="Returns">Returns</option>
          </select>
        </div>
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Analyzing correlations...</p>
          </div>
        </div>
      )}
      
      {!loading && correlationResults && (
        <>
          {/* Heatmap */}
          {heatmapData && (
            <div className="mb-6">
              <ReactApexChart
                options={heatmapData.options}
                series={heatmapData.series}
                type="heatmap"
                height={400}
              />
            </div>
          )}
          
          {/* Top Correlations */}
          {correlationResults.top_correlations && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Top Correlations for {selectedTicker}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Positive Correlations */}
                <div>
                  <h4 className="font-medium text-green-600 mb-2">
                    📈 Positive Correlations
                  </h4>
                  <div className="space-y-2">
                    {correlationResults.top_correlations.positive?.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-green-50 rounded">
                        <span className="font-medium">{item.ticker}</span>
                        <div className="text-right">
                          <span className={`text-lg font-bold ${getCorrelationColor(item.correlation)}`}>
                            {item.correlation.toFixed(3)}
                          </span>
                          <p className="text-xs text-gray-600">
                            {getCorrelationStrength(item.correlation)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Negative Correlations */}
                <div>
                  <h4 className="font-medium text-red-600 mb-2">
                    📉 Negative Correlations
                  </h4>
                  <div className="space-y-2">
                    {correlationResults.top_correlations.negative?.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-red-50 rounded">
                        <span className="font-medium">{item.ticker}</span>
                        <div className="text-right">
                          <span className={`text-lg font-bold ${getCorrelationColor(item.correlation)}`}>
                            {item.correlation.toFixed(3)}
                          </span>
                          <p className="text-xs text-gray-600">
                            {getCorrelationStrength(item.correlation)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Statistics */}
          {correlationResults.statistics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Average Correlation</p>
                <p className="text-xl font-bold">
                  {correlationResults.statistics.avg_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Max Correlation</p>
                <p className="text-xl font-bold">
                  {correlationResults.statistics.max_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Min Correlation</p>
                <p className="text-xl font-bold">
                  {correlationResults.statistics.min_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Std Deviation</p>
                <p className="text-xl font-bold">
                  {correlationResults.statistics.std_correlation?.toFixed(3)}
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default CorrelationAnalysis;
