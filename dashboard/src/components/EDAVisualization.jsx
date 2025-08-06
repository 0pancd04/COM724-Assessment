import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import axios from 'axios';
import { toast } from 'react-toastify';
import useCryptoStore from '../stores/cryptoStore';
import { API_ENDPOINTS } from '../config/api';

const EDAVisualization = () => {
  const [dataSource, setDataSource] = useState('yfinance');
  const { selectedTicker } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [activeChart, setActiveChart] = useState('price_trends');
  const [edaData, setEdaData] = useState(null);
  const [rawData, setRawData] = useState(null);
  
  useEffect(() => {
    if (selectedTicker) {
      loadEDAData();
    }
  }, [selectedTicker, dataSource]);
  
  const loadEDAData = async () => {
    setLoading(true);
    try {
      // Fetch comprehensive EDA data from new endpoint
      const response = await axios.get(`${API_ENDPOINTS.EDA}/${selectedTicker}`, {
        params: { source: dataSource }
      });
      
      if (response.data && response.data.success) {
        setEdaData(response.data);
        
        // Set raw data if available
        if (response.data.raw_data) {
          setRawData(response.data.raw_data);
        } else {
          // Fallback: try to fetch raw data separately
          await fetchTickerData();
        }
        
        toast.success(`📊 Loaded EDA for ${selectedTicker} from ${dataSource} (${response.data.statistics?.total_records || 0} records)`);
      } else {
        // Handle case where no data is available
        toast.warn(`⚠️ ${response.data.message || 'No EDA data available'}`);
        setEdaData(response.data);
        
        // Still try to fetch raw data for basic charts
        await fetchTickerData();
      }
    } catch (error) {
      console.error('Failed to load EDA data:', error);
      toast.error('Failed to load EDA analysis. Trying to load basic data...');
      
      // Fallback: try to load basic ticker data
      await fetchTickerData();
    } finally {
      setLoading(false);
    }
  };
  
  const processChartData = (data) => {
    // Process the data for ApexCharts
    // The backend returns HTML charts, but we'll create our own from the report data
    if (data.report && data.report.num_records) {
      // Fetch raw ticker data for chart generation
      fetchTickerData();
    }
  };
  
  const fetchTickerData = async () => {
    try {
      const response = await axios.get(`${API_ENDPOINTS.DATABASE}/ticker-data/${selectedTicker}`, {
        params: { source: dataSource, limit: 90 }
      });
      
      if (response.data && response.data.data) {
        setRawData(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch ticker data:', error);
    }
  };
  
  // Chart configurations with glassmorphism styling
  const getChartOptions = (type) => {
    const baseOptions = {
      chart: {
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
          }
        },
        background: 'transparent',
        fontFamily: 'Inter, sans-serif',
      },
      theme: {
        mode: 'light',
        palette: 'palette2'
      },
      grid: {
        borderColor: '#e0e0e0',
        strokeDashArray: 4,
      }
    };
    
    switch(type) {
      case 'candlestick':
        return {
          ...baseOptions,
          chart: { ...baseOptions.chart, type: 'candlestick', height: 450 },
          title: { text: `${selectedTicker} Price Action`, align: 'left', style: { fontSize: '18px', fontWeight: 'bold' } },
          xaxis: { type: 'datetime' },
          yaxis: { 
            tooltip: { enabled: true },
            labels: { formatter: (val) => `$${val?.toFixed(2)}` }
          },
          plotOptions: {
            candlestick: {
              colors: {
                upward: '#10b981',
                downward: '#ef4444'
              }
            }
          }
        };
        
      case 'volume':
        return {
          ...baseOptions,
          chart: { ...baseOptions.chart, type: 'bar', height: 200 },
          title: { text: 'Trading Volume', align: 'left' },
          xaxis: { type: 'datetime' },
          yaxis: {
            labels: { formatter: (val) => `${(val / 1e6).toFixed(1)}M` }
          },
          colors: ['#8b5cf6'],
          fill: {
            type: 'gradient',
            gradient: {
              shade: 'dark',
              type: 'vertical',
              shadeIntensity: 0.5,
              inverseColors: false,
              opacityFrom: 0.8,
              opacityTo: 0.3,
            }
          }
        };
        
      case 'ma':
        return {
          ...baseOptions,
          chart: { ...baseOptions.chart, type: 'line', height: 400 },
          title: { text: 'Moving Averages', align: 'left' },
          stroke: { curve: 'smooth', width: 2 },
          xaxis: { type: 'datetime' },
          yaxis: { labels: { formatter: (val) => `$${val?.toFixed(2)}` } },
          legend: { position: 'top', horizontalAlign: 'left' },
          colors: ['#3b82f6', '#10b981', '#f59e0b']
        };
        
      case 'volatility':
        return {
          ...baseOptions,
          chart: { ...baseOptions.chart, type: 'area', height: 300 },
          title: { text: 'Price Volatility (20-day)', align: 'left' },
          fill: {
            type: 'gradient',
            gradient: {
              shadeIntensity: 1,
              opacityFrom: 0.7,
              opacityTo: 0.3,
              stops: [0, 90, 100]
            }
          },
          stroke: { curve: 'smooth', width: 2 },
          xaxis: { type: 'datetime' },
          yaxis: { labels: { formatter: (val) => `${val?.toFixed(1)}%` } },
          colors: ['#ef4444']
        };
        
      default:
        return baseOptions;
    }
  };
  
  // Generate chart series from raw data
  const generateChartSeries = (type) => {
    if (!rawData || rawData.length === 0) return [];
    
    switch(type) {
      case 'candlestick':
        return [{
          name: 'OHLC',
          data: rawData.map(item => ({
            x: new Date(item.timestamp),
            y: [item.open, item.high, item.low, item.close]
          }))
        }];
        
      case 'volume':
        return [{
          name: 'Volume',
          data: rawData.map(item => ({
            x: new Date(item.timestamp),
            y: item.volume
          }))
        }];
        
      case 'ma':
        const prices = rawData.map(item => item.close);
        const ma7 = calculateMA(rawData, 7);
        const ma20 = calculateMA(rawData, 20);
        const ma50 = calculateMA(rawData, 50);
        
        return [
          { name: 'MA7', data: ma7 },
          { name: 'MA20', data: ma20 },
          { name: 'MA50', data: ma50 }
        ];
        
      case 'volatility':
        const volatility = calculateVolatility(rawData, 20);
        return [{
          name: 'Volatility %',
          data: volatility
        }];
        
      default:
        return [];
    }
  };
  
  // Helper functions for calculations
  const calculateMA = (data, period) => {
    const result = [];
    for (let i = period - 1; i < data.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close;
      }
      result.push({
        x: new Date(data[i].timestamp),
        y: sum / period
      });
    }
    return result;
  };
  
  const calculateVolatility = (data, period = 20) => {
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      returns.push((data[i].close - data[i - 1].close) / data[i - 1].close);
    }
    
    const result = [];
    for (let i = period - 1; i < returns.length; i++) {
      const periodReturns = returns.slice(i - period + 1, i + 1);
      const mean = periodReturns.reduce((a, b) => a + b, 0) / period;
      const variance = periodReturns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
      const stdDev = Math.sqrt(variance) * Math.sqrt(252) * 100; // Annualized
      
      result.push({
        x: new Date(data[i + 1].timestamp),
        y: stdDev
      });
    }
    return result;
  };
  
  const chartTypes = [
    { id: 'price_trends', name: '📈 Price Trends', icon: '📊', gradient: 'gradient-blue', description: 'Candlestick chart showing OHLC data' },
    { id: 'volume', name: '📊 Volume', icon: '📊', gradient: 'gradient-purple', description: 'Trading volume analysis' },
    { id: 'moving_avg', name: '📉 Moving Averages', icon: '📈', gradient: 'gradient-green', description: '7, 20, and 50-day moving averages' },
    { id: 'volatility', name: '🌊 Volatility', icon: '📊', gradient: 'gradient-orange', description: '20-day rolling volatility' },
    { id: 'distribution', name: '📊 Distribution', icon: '📈', gradient: 'gradient-red', description: 'Price distribution histograms' },
    { id: 'correlation', name: '🔗 Correlations', icon: '📊', gradient: 'gradient-teal', description: 'Feature correlation analysis' },
  ];
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gradient">
          📊 Exploratory Data Analysis
        </h2>
        <div className="flex items-center gap-2">
          <span className="px-4 py-2 glass-card text-sm font-semibold">
            {selectedTicker}
          </span>
          <select
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
            className="px-4 py-2 glass-dropdown text-sm font-medium"
          >
            <option value="yfinance">📈 Yahoo Finance</option>
            <option value="binance">🟡 Binance</option>
          </select>
          {edaData?.statistics?.total_records && (
            <span className="px-4 py-2 glass-card text-sm">
              {edaData.statistics.total_records} Records
            </span>
          )}
        </div>
      </div>
      
      {/* Chart Type Selector */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {chartTypes.map(type => (
          <button
            key={type.id}
            onClick={() => setActiveChart(type.id)}
            className={`
              p-4 rounded-xl transition-all duration-300 transform hover:scale-105
              ${activeChart === type.id
                ? `${type.gradient} text-white shadow-xl`
                : 'glass-card hover:shadow-lg'
              }
            `}
          >
            <div className="text-2xl mb-2">{type.icon}</div>
            <div className="font-semibold text-sm">{type.name}</div>
            <div className="text-xs opacity-75 mt-1">{type.description}</div>
          </button>
        ))}
      </div>
      
      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="w-20 h-20 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-lg text-gray-600">Loading EDA analysis...</p>
          </div>
        </div>
      )}
      
      {/* Charts Display */}
      {!loading && rawData && (
        <div className="glass-card p-6 animate-fadeIn">
          {activeChart === 'price_trends' && (
            <ReactApexChart
              options={getChartOptions('candlestick')}
              series={generateChartSeries('candlestick')}
              type="candlestick"
              height={450}
            />
          )}
          
          {activeChart === 'volume' && (
            <ReactApexChart
              options={getChartOptions('volume')}
              series={generateChartSeries('volume')}
              type="bar"
              height={300}
            />
          )}
          
          {activeChart === 'moving_avg' && (
            <ReactApexChart
              options={getChartOptions('ma')}
              series={generateChartSeries('ma')}
              type="line"
              height={400}
            />
          )}
          
          {activeChart === 'volatility' && (
            <ReactApexChart
              options={getChartOptions('volatility')}
              series={generateChartSeries('volatility')}
              type="area"
              height={300}
            />
          )}
          
          {activeChart === 'distribution' && rawData && (
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gradient">Price Distribution Analysis</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ReactApexChart
                  options={{
                    chart: { type: 'histogram', toolbar: { show: true } },
                    title: { text: 'Close Price Distribution' },
                    xaxis: { title: { text: 'Price ($)' } },
                    yaxis: { title: { text: 'Frequency' } },
                    colors: ['#3b82f6']
                  }}
                  series={[{
                    name: 'Frequency',
                    data: rawData.map(item => item.close)
                  }]}
                  type="histogram"
                  height={300}
                />
                <ReactApexChart
                  options={{
                    chart: { type: 'boxPlot', toolbar: { show: true } },
                    title: { text: 'OHLC Box Plot' },
                    xaxis: { categories: ['Open', 'High', 'Low', 'Close'] },
                    colors: ['#10b981']
                  }}
                  series={[{
                    name: 'OHLC',
                    data: [
                      { x: 'Open', y: rawData.map(item => item.open) },
                      { x: 'High', y: rawData.map(item => item.high) },
                      { x: 'Low', y: rawData.map(item => item.low) },
                      { x: 'Close', y: rawData.map(item => item.close) }
                    ]
                  }]}
                  type="boxPlot"
                  height={300}
                />
              </div>
            </div>
          )}
          
          {activeChart === 'correlation' && (
            <div className="text-center p-8">
              <div className="text-4xl mb-4">🔗</div>
              <h4 className="text-lg font-semibold text-gradient mb-2">Correlation Analysis</h4>
              <p className="text-gray-600 mb-4">
                Feature correlation analysis will be displayed here once comprehensive EDA is run.
              </p>
              <button
                onClick={loadEDAData}
                className="px-6 py-3 gradient-teal text-white rounded-lg hover:shadow-lg transition-all duration-300"
              >
                Run Full EDA Analysis
              </button>
            </div>
          )}
        </div>
      )}
      
      {/* Statistics Summary */}
      {edaData?.report && (
        <div className="glass-card p-6">
          <h3 className="text-xl font-bold mb-4 text-gradient">Statistical Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(edaData.report).slice(0, 8).map(([key, value]) => (
              <div key={key} className="p-4 glass rounded-xl">
                <p className="text-sm text-gray-600 capitalize mb-1">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-2xl font-bold text-gradient">
                  {typeof value === 'number' ? value.toFixed(2) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* No Data State */}
      {!loading && !rawData && (
        <div className="glass-card p-12 text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Data Available</h3>
          <p className="text-gray-600 mb-4">
            Please run the pipeline or select a different ticker to view EDA analysis.
          </p>
          <button
            onClick={loadEDAData}
            className="px-6 py-3 gradient-blue text-white rounded-lg hover:shadow-lg transition-all duration-300"
          >
            Reload Data
          </button>
        </div>
      )}
    </div>
  );
};

export default EDAVisualization;