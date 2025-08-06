import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const CorrelationAnalysis = () => {
  const { selectedTicker, correlationData, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [selectedTickers, setSelectedTickers] = useState(['BTC', 'ETH', 'ADA', 'DOT']);
  const [feature, setFeature] = useState('Close');
  const [dataSource, setDataSource] = useState('yfinance');
  
  useEffect(() => {
    loadCorrelation();
  }, [selectedTickers, feature, dataSource]);
  
  const loadCorrelation = async () => {
    setLoading(true);
    try {
      await actions.fetchCorrelationAnalysis({
        tickers: selectedTickers.join(','),
        feature: feature,
        source: dataSource
      });
    } catch (error) {
      console.error('Failed to load correlation analysis:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const prepareHeatmap = () => {
    if (!correlationData?.report?.top_positive_pairs) return null;
    
    // Extract unique tickers from positive and negative pairs
    const pairs = [
      ...correlationData.report.top_positive_pairs,
      ...correlationData.report.top_negative_pairs
    ];
    const uniqueTickers = Array.from(new Set(pairs.flatMap(p => p.pair)));
    
    // Create empty matrix
    const matrix = {};
    uniqueTickers.forEach(t1 => {
      matrix[t1] = {};
      uniqueTickers.forEach(t2 => {
        matrix[t1][t2] = t1 === t2 ? 1 : null;
      });
    });
    
    // Fill in known correlations
    pairs.forEach(({ pair, correlation }) => {
      const [t1, t2] = pair;
      matrix[t1][t2] = correlation;
      matrix[t2][t1] = correlation; // Matrix is symmetric
    });
    
    const series = uniqueTickers.map(ticker => ({
      name: ticker,
      data: uniqueTickers.map(t2 => ({
        x: t2,
        y: matrix[ticker][t2] || 0
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
          text: `Correlation Matrix (${feature})`,
          align: 'left',
          style: {
            fontSize: '16px',
            fontWeight: 'bold'
          }
        },
        xaxis: {
          type: 'category',
          categories: uniqueTickers,
          labels: {
            style: {
              fontSize: '12px',
              fontWeight: 'bold'
            }
          }
        },
        yaxis: {
          labels: {
            style: {
              fontSize: '12px',
              fontWeight: 'bold'
            }
          }
        },
        plotOptions: {
          heatmap: {
            radius: 2,
            enableShades: true,
            colorScale: {
              ranges: [
                { from: -1, to: -0.5, color: '#FF4560', name: 'Strong Negative' },
                { from: -0.5, to: 0, color: '#FEB019', name: 'Weak Negative' },
                { from: 0, to: 0.5, color: '#00E396', name: 'Weak Positive' },
                { from: 0.5, to: 1, color: '#008FFB', name: 'Strong Positive' }
              ]
            }
          }
        },
        tooltip: {
          theme: 'dark',
          y: {
            formatter: (val) => val?.toFixed(4)
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
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
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
            Data Source
          </label>
          <select
            className="w-full px-3 py-2 glass-dropdown"
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
          >
            <option value="yfinance">📈 Yahoo Finance</option>
            <option value="binance">🟡 Binance</option>
          </select>
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
            <option value="Open">Open Price</option>
            <option value="High">High Price</option>
            <option value="Low">Low Price</option>
            <option value="Volume">Volume</option>
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
      
      {!loading && correlationData && (
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
          {correlationData?.report?.top_positive_pairs && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Top Correlations ({feature})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Positive Correlations */}
                <div>
                  <h4 className="font-medium text-green-600 mb-2">
                    📈 Strongest Positive Correlations
                  </h4>
                  <div className="space-y-2">
                    {correlationData.report.top_positive_pairs.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-green-50 rounded">
                        <span className="font-medium">{item.pair.join(' → ')}</span>
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
                
                {/* Negative/Lowest Correlations */}
                <div>
                  <h4 className="font-medium text-red-600 mb-2">
                    📉 Lowest Correlations
                  </h4>
                  <div className="space-y-2">
                    {correlationData.report.top_negative_pairs.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-red-50 rounded">
                        <span className="font-medium">{item.pair.join(' → ')}</span>
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
          {correlationData?.report?.statistics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Average Correlation</p>
                <p className="text-2xl font-bold text-blue-600">
                  {correlationData.report.statistics.avg_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Max Correlation</p>
                <p className="text-2xl font-bold text-green-600">
                  {correlationData.report.statistics.max_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Min Correlation</p>
                <p className="text-2xl font-bold text-red-600">
                  {correlationData.report.statistics.min_correlation?.toFixed(3)}
                </p>
              </div>
              <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Std Deviation</p>
                <p className="text-2xl font-bold text-purple-600">
                  {correlationData.report.statistics.std_correlation?.toFixed(3)}
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
