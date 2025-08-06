import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import axios from 'axios';
import { toast } from 'react-toastify';
import useCryptoStore from '../stores/cryptoStore';
import { API_ENDPOINTS } from '../config/api';

const ModelComparison = () => {
  const { selectedTicker } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [forecasts, setForecasts] = useState(null);
  const [selectedModel, setSelectedModel] = useState('arima');
  const [forecastPeriods, setForecastPeriods] = useState(7);
  
  const models = [
    { id: 'arima', name: 'ARIMA', color: '#3b82f6', description: 'Auto-Regressive Integrated Moving Average' },
    { id: 'sarima', name: 'SARIMA', color: '#10b981', description: 'Seasonal ARIMA' },
    { id: 'rf', name: 'Random Forest', color: '#f59e0b', description: 'Ensemble Learning' },
    { id: 'xgb', name: 'XGBoost', color: '#ef4444', description: 'Gradient Boosting' },
  ];
  
  useEffect(() => {
    if (selectedTicker) {
      loadModelData();
    }
  }, [selectedTicker]);
  
  const loadModelData = async () => {
    setLoading(true);
    try {
      // Try to get existing model metrics
      const response = await axios.get(`${API_ENDPOINTS.TRAIN(selectedTicker)}`, {
        params: { feature: 'Close', test_size: 0.2 }
      });
      
      if (response.data && response.data.metrics) {
        setModelMetrics(response.data.metrics);
        toast.success(`📊 Loaded model metrics for ${selectedTicker}`);
      }
    } catch (error) {
      console.error('No existing models found:', error);
      setModelMetrics(null);
    } finally {
      setLoading(false);
    }
  };
  
  const trainModels = async () => {
    setTraining(true);
    try {
      toast.info('🚀 Training models... This may take a few minutes.');
      
      const response = await axios.get(`${API_ENDPOINTS.TRAIN(selectedTicker)}`, {
        params: { 
          feature: 'Close', 
          test_size: 0.2,
          source: 'yfinance'
        },
        timeout: 60000 // 60 second timeout for training
      });
      
      if (response.data && response.data.metrics) {
        setModelMetrics(response.data.metrics);
        toast.success('✅ Models trained successfully!');
      }
    } catch (error) {
      console.error('Training failed:', error);
      toast.error('Failed to train models. Please try again.');
    } finally {
      setTraining(false);
    }
  };
  
  const generateForecast = async () => {
    if (!selectedModel) {
      toast.warning('Please select a model first');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.get(`${API_ENDPOINTS.FORECAST(selectedTicker)}`, {
        params: {
          model_type: selectedModel,
          periods: forecastPeriods,
          source: 'yfinance'
        }
      });
      
      if (response.data && response.data.forecast) {
        setForecasts(response.data.forecast);
        toast.success(`📈 Generated ${forecastPeriods}-day forecast using ${selectedModel.toUpperCase()}`);
      }
    } catch (error) {
      console.error('Forecast failed:', error);
      toast.error('Failed to generate forecast. Please train the models first.');
    } finally {
      setLoading(false);
    }
  };
  
  // Prepare metrics chart data
  const getMetricsChartOptions = () => ({
    chart: {
      type: 'bar',
      height: 350,
      toolbar: { show: true },
      background: 'transparent',
    },
    plotOptions: {
      bar: {
        horizontal: true,
        distributed: true,
        barHeight: '70%',
      }
    },
    colors: models.map(m => m.color),
    xaxis: {
      categories: ['MAE', 'RMSE', 'MAPE', 'R²'],
      labels: {
        formatter: (val) => typeof val === 'number' ? val.toFixed(4) : val
      }
    },
    yaxis: {
      labels: {
        style: {
          fontSize: '12px',
          fontWeight: 600,
        }
      }
    },
    title: {
      text: 'Model Performance Comparison',
      align: 'center',
      style: {
        fontSize: '18px',
        fontWeight: 'bold',
      }
    },
    dataLabels: {
      enabled: true,
      formatter: (val) => typeof val === 'number' ? val.toFixed(4) : val,
      style: {
        fontSize: '12px',
      }
    },
    legend: {
      show: true,
      position: 'top',
    },
    theme: {
      mode: 'light',
    }
  });
  
  const getMetricsChartSeries = () => {
    if (!modelMetrics) return [];
    
    return [
      {
        name: 'MAE',
        data: models.map(m => modelMetrics[m.id]?.mae || 0)
      },
      {
        name: 'RMSE', 
        data: models.map(m => modelMetrics[m.id]?.rmse || 0)
      },
      {
        name: 'MAPE',
        data: models.map(m => modelMetrics[m.id]?.mape || 0)
      },
      {
        name: 'R²',
        data: models.map(m => modelMetrics[m.id]?.r2 || 0)
      }
    ];
  };
  
  // Prepare forecast chart
  const getForecastChartOptions = () => ({
    chart: {
      type: 'line',
      height: 350,
      toolbar: { show: true },
      background: 'transparent',
    },
    stroke: {
      curve: 'smooth',
      width: 3,
    },
    xaxis: {
      type: 'datetime',
      labels: {
        datetimeFormatter: {
          day: 'dd MMM'
        }
      }
    },
    yaxis: {
      labels: {
        formatter: (val) => `$${val?.toFixed(2)}`
      }
    },
    title: {
      text: `${selectedModel.toUpperCase()} Forecast - Next ${forecastPeriods} Days`,
      align: 'center',
      style: {
        fontSize: '18px',
        fontWeight: 'bold',
      }
    },
    colors: [models.find(m => m.id === selectedModel)?.color || '#3b82f6'],
    markers: {
      size: 5,
      strokeWidth: 2,
      hover: {
        size: 7
      }
    },
    theme: {
      mode: 'light',
    }
  });
  
  const getForecastChartSeries = () => {
    if (!forecasts) return [];
    
    const data = Object.entries(forecasts).map(([date, values]) => ({
      x: new Date(date),
      y: values.forecast || values
    }));
    
    return [{
      name: 'Forecast',
      data: data
    }];
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gradient">
          🤖 Model Comparison & Training
        </h2>
        <div className="flex items-center gap-3">
          <span className="px-4 py-2 glass-card text-sm font-semibold">
            {selectedTicker}
          </span>
          <button
            onClick={trainModels}
            disabled={training}
            className={`
              px-6 py-3 rounded-lg font-semibold transition-all duration-300
              ${training 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'gradient-blue text-white hover:shadow-lg transform hover:scale-105'
              }
            `}
          >
            {training ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Training...
              </span>
            ) : (
              '🚀 Train All Models'
            )}
          </button>
        </div>
      </div>
      
      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {models.map(model => (
          <div 
            key={model.id}
            className={`
              glass-card p-4 cursor-pointer transition-all duration-300 transform hover:scale-105
              ${selectedModel === model.id ? 'ring-2 ring-purple-500 shadow-xl' : ''}
            `}
            onClick={() => setSelectedModel(model.id)}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold text-lg">{model.name}</h3>
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: model.color }}
              />
            </div>
            <p className="text-sm text-gray-600">{model.description}</p>
            {modelMetrics && modelMetrics[model.id] && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">MAE:</span>
                    <span className="ml-1 font-semibold">
                      {modelMetrics[model.id].mae?.toFixed(4)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">R²:</span>
                    <span className="ml-1 font-semibold">
                      {modelMetrics[model.id].r2?.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Metrics Comparison Chart */}
      {modelMetrics && (
        <div className="glass-card p-6">
          <ReactApexChart
            options={getMetricsChartOptions()}
            series={getMetricsChartSeries()}
            type="bar"
            height={350}
          />
        </div>
      )}
      
      {/* Forecast Section */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold">Generate Forecast</h3>
          <div className="flex items-center gap-3">
            <select
              value={forecastPeriods}
              onChange={(e) => setForecastPeriods(Number(e.target.value))}
              className="px-4 py-2 glass-dropdown"
            >
              <option value={7}>7 Days</option>
              <option value={14}>14 Days</option>
              <option value={30}>30 Days</option>
              <option value={60}>60 Days</option>
              <option value={90}>90 Days</option>
            </select>
            <button
              onClick={generateForecast}
              disabled={loading || !modelMetrics}
              className={`
                px-6 py-2 rounded-lg font-semibold transition-all duration-300
                ${loading || !modelMetrics
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'gradient-green text-white hover:shadow-lg'
                }
              `}
            >
              {loading ? 'Generating...' : '📈 Generate Forecast'}
            </button>
          </div>
        </div>
        
        {forecasts && (
          <ReactApexChart
            options={getForecastChartOptions()}
            series={getForecastChartSeries()}
            type="line"
            height={350}
          />
        )}
      </div>
      
      {/* No Models State */}
      {!loading && !modelMetrics && !training && (
        <div className="glass-card p-12 text-center">
          <div className="text-6xl mb-4">🤖</div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Models Trained</h3>
          <p className="text-gray-600 mb-6">
            Train models to compare their performance and generate forecasts.
          </p>
          <button
            onClick={trainModels}
            className="px-8 py-3 gradient-blue text-white rounded-lg hover:shadow-lg transition-all duration-300 font-semibold"
          >
            🚀 Start Training
          </button>
        </div>
      )}
      
      {/* Loading State */}
      {loading && !training && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-20 h-20 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-lg text-gray-600">Loading model data...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;