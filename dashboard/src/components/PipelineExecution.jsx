import React, { useState, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import axios from 'axios';
import useCryptoStore from '../stores/cryptoStore';
import { API_ENDPOINTS } from '../config/api';

const PipelineExecution = () => {
  const { selectedTicker, wsMessages, pipelineStatus, actions } = useCryptoStore();
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState([]);
  const [pipelineResults, setPipelineResults] = useState(null);
  const [activeResultsTab, setActiveResultsTab] = useState('summary');
  const logsEndRef = useRef(null);
  
  // Pipeline configuration options
  const [config, setConfig] = useState({
    tickers: 'TOP30',
    sources: 'yfinance,binance',
    period: '90d',
    interval: '1d',
    max_days: 90,
    feature: 'Close',
    test_size: 0.2,
    include_eda: true,
    include_clustering: true
  });
  
  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [logs]);
  
  useEffect(() => {
    // Handle WebSocket messages for pipeline status updates
    if (wsMessages.length > 0) {
      const latestMessage = wsMessages[wsMessages.length - 1];
      
      switch (latestMessage.type) {
        case 'pipeline_start':
          setIsRunning(true);
          setProgress(0);
          setCurrentStep(latestMessage.status.current_step);
          handlePipelineLog({
            level: 'info',
            message: '🚀 Pipeline started',
            step: 'initialization'
          });
          break;
          
        case 'pipeline_update':
        case 'step_update':
          if (latestMessage.status) {
            setProgress(latestMessage.status.progress || 0);
            setCurrentStep(latestMessage.status.current_step);
            if (latestMessage.status.messages?.length > 0) {
              const lastMessage = latestMessage.status.messages[latestMessage.status.messages.length - 1];
              handlePipelineLog({
                level: 'info',
                message: lastMessage,
                step: latestMessage.status.current_step
              });
            }
          }
          break;
          
        case 'step_complete':
          handlePipelineLog({
            level: 'success',
            message: `✅ Completed: ${latestMessage.step}`,
            step: latestMessage.step
          });
          break;
          
        case 'pipeline_end':
          setIsRunning(false);
          setProgress(latestMessage.success ? 100 : progress);
          setCurrentStep(latestMessage.success ? 'completed' : 'failed');
          handlePipelineLog({
            level: latestMessage.success ? 'success' : 'error',
            message: latestMessage.status.messages?.[latestMessage.status.messages.length - 1] || 
                    (latestMessage.success ? '🎉 Pipeline completed successfully!' : '❌ Pipeline failed'),
            step: latestMessage.success ? 'completed' : 'failed'
          });
          if (latestMessage.success) {
            fetchPipelineResults();
          }
          break;
          
        default:
          // Handle any other message types
          if (latestMessage.message || latestMessage.content) {
            handlePipelineLog({
              level: latestMessage.level || 'info',
              message: latestMessage.message || latestMessage.content,
              step: latestMessage.step || currentStep
            });
          }
      }
    }
  }, [wsMessages, progress, currentStep]);
  
  useEffect(() => {
    // Update running state based on pipeline status
    setIsRunning(pipelineStatus === 'running' || pipelineStatus === 'starting');
  }, [pipelineStatus]);
  
  const handlePipelineStatusUpdate = (message) => {
    const { status } = message;
    
    if (status) {
      setProgress(status.progress || 0);
      setCurrentStep(status.current_step || '');
      
      if (status.messages?.length > 0) {
        const lastMessage = status.messages[status.messages.length - 1];
        handlePipelineLog({
          level: 'info',
          message: lastMessage,
          step: status.current_step
        });
      }
    }
  };
  
  const handlePipelineLog = (message) => {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level: message.level || 'info',
      message: message.message || message.content || JSON.stringify(message),
      step: message.step || currentStep
    };
    setLogs(prev => [...prev, logEntry]);
  };
  
  const fetchPipelineResults = async () => {
    try {
      // Fetch database summary
      const summaryResponse = await axios.get(API_ENDPOINTS.DATABASE.SUMMARY);
      
      // Fetch available tickers
      const tickersResponse = await axios.get(API_ENDPOINTS.DATABASE.TICKERS);
      
      // Fetch recent EDA results
      const edaResults = [];
      if (tickersResponse.data.tickers) {
        for (const ticker of tickersResponse.data.tickers.slice(0, 5)) {
          try {
            const edaResponse = await axios.get(`${API_ENDPOINTS.EDA}/${ticker}`, {
              params: { source: 'yfinance' }
            });
            if (edaResponse.data) {
              edaResults.push({ ticker, data: edaResponse.data });
            }
          } catch (error) {
            // Skip if no EDA data for this ticker
          }
        }
      }
      
      setPipelineResults({
        summary: summaryResponse.data,
        tickers: tickersResponse.data,
        edaResults,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to fetch pipeline results:', error);
    }
  };
  
  const startPipeline = async () => {
    try {
      setIsRunning(true);
      setProgress(0);
      setLogs([]);
      setPipelineResults(null);
      
      // Initialize WebSocket connection before starting pipeline
      await actions.initWebSocket();
      
      // Add initial log
      setLogs([{
        timestamp: new Date().toISOString(),
        level: 'info',
        message: '🚀 Starting pipeline execution...',
        step: 'initialization'
      }]);
      
      const response = await actions.runPipeline(config);
      
      if (response && response.error) {
        toast.error(`Failed to start pipeline: ${response.error}`);
        setIsRunning(false);
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Pipeline failed to start: ${response.error}`,
          step: 'error'
        }]);
      }
    } catch (error) {
      console.error('Pipeline execution failed:', error);
      toast.error('Failed to start pipeline execution');
      setIsRunning(false);
      setLogs(prev => [...prev, {
        timestamp: new Date().toISOString(),
        level: 'error',
        message: `Pipeline execution failed: ${error.message}`,
        step: 'error'
      }]);
    }
  };
  
  const pipelineSteps = [
    { id: 'Data Download', name: '📥 Data Download', description: 'Fetching data from sources' },
    { id: 'Data Preprocessing (yfinance)', name: '🔄 YFinance Preprocessing', description: 'Processing YFinance data' },
    { id: 'Data Preprocessing (binance)', name: '🔄 Binance Preprocessing', description: 'Processing Binance data' },
    { id: 'Dimensionality Reduction', name: '📊 Dimensionality Reduction', description: 'Reducing data dimensions' },
    { id: 'Clustering Analysis', name: '🎯 Clustering', description: 'Performing clustering analysis' },
    { id: 'Model Training & Comparison', name: '🤖 Model Training', description: 'Training and comparing models' }
  ];
  
  const getStepStatus = (stepId) => {
    // Check if step is currently active
    if (currentStep === stepId) return 'active';
    
    // Check if step is completed by looking at logs or completed steps
    const stepIndex = pipelineSteps.findIndex(step => step.id === stepId);
    const currentStepIndex = pipelineSteps.findIndex(step => step.id === currentStep);
    
    // If current step is further along, this step is completed
    if (currentStepIndex > stepIndex) return 'completed';
    
    // If pipeline is completed and this is the last step
    if (currentStep === 'completed' && stepIndex === pipelineSteps.length - 1) return 'completed';
    
    // Check completed steps from logs
    const isCompleted = logs.some(log => 
      log.message.includes(`✅ Completed: ${stepId}`) || 
      log.message.includes(`Completed: ${stepId}`)
    );
    
    return isCompleted ? 'completed' : 'pending';
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gradient">
          🚀 Pipeline Execution
        </h2>
        <div className="flex items-center gap-3">
          <button
            onClick={startPipeline}
            disabled={isRunning}
            className={`
              px-6 py-3 rounded-xl font-semibold transition-all duration-300
              ${isRunning
                ? 'bg-gray-400 cursor-not-allowed'
                : 'gradient-blue text-white hover:shadow-lg transform hover:scale-105'
              }
            `}
          >
            {isRunning ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running Pipeline...
              </span>
            ) : (
              '🚀 Start Pipeline'
            )}
          </button>
        </div>
      </div>
      
      {/* Configuration Panel */}
      <div className="glass-card p-6">
        <h3 className="text-xl font-semibold mb-4">Pipeline Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Tickers
            </label>
            <select
              value={config.tickers}
              onChange={(e) => setConfig({ ...config, tickers: e.target.value })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value="TOP30">Top 30 Cryptocurrencies</option>
              <option value={selectedTicker}>Current Ticker Only</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Data Sources
            </label>
            <select
              value={config.sources}
              onChange={(e) => setConfig({ ...config, sources: e.target.value })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value="yfinance,binance">Both (YFinance + Binance)</option>
              <option value="yfinance">YFinance Only</option>
              <option value="binance">Binance Only</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Time Period
            </label>
            <select
              value={config.period}
              onChange={(e) => setConfig({ ...config, period: e.target.value })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value="30d">30 Days</option>
              <option value="90d">90 Days</option>
              <option value="180d">180 Days</option>
              <option value="365d">1 Year</option>
              <option value="max">Maximum Available</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Interval
            </label>
            <select
              value={config.interval}
              onChange={(e) => setConfig({ ...config, interval: e.target.value })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value="1d">Daily</option>
              <option value="1h">Hourly</option>
              <option value="15m">15 Minutes</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Target Feature
            </label>
            <select
              value={config.feature}
              onChange={(e) => setConfig({ ...config, feature: e.target.value })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value="Close">Close Price</option>
              <option value="High">High Price</option>
              <option value="Low">Low Price</option>
              <option value="Volume">Volume</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Test Split Size
            </label>
            <select
              value={config.test_size}
              onChange={(e) => setConfig({ ...config, test_size: parseFloat(e.target.value) })}
              className="w-full px-4 py-2 glass-dropdown"
              disabled={isRunning}
            >
              <option value={0.1}>10%</option>
              <option value={0.2}>20%</option>
              <option value={0.3}>30%</option>
            </select>
          </div>
        </div>
        
        <div className="mt-4 flex items-center gap-4">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={config.include_eda}
              onChange={(e) => setConfig({ ...config, include_eda: e.target.checked })}
              className="rounded text-purple-500 focus:ring-purple-500"
              disabled={isRunning}
            />
            <span className="text-sm text-gray-700">Include EDA Analysis</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={config.include_clustering}
              onChange={(e) => setConfig({ ...config, include_clustering: e.target.checked })}
              className="rounded text-purple-500 focus:ring-purple-500"
              disabled={isRunning}
            />
            <span className="text-sm text-gray-700">Include Clustering</span>
          </label>
        </div>
      </div>
      
      {/* Progress Tracking - Only show when pipeline is active or has been started */}
      {(isRunning || logs.length > 0 || currentStep) ? (
        <div className="glass-card p-6">
          <h3 className="text-xl font-semibold mb-4">Pipeline Progress</h3>
          
          {/* Progress Steps */}
          <div className="relative mb-8">
            {/* Progress Steps */}
            <div className="grid grid-cols-6 gap-4">
              {pipelineSteps.map((step, index) => {
                const status = getStepStatus(step.id);
                
                return (
                  <div
                    key={step.id}
                    className={`
                      flex flex-col items-center text-center transition-all duration-300
                      ${status === 'active' ? 'transform scale-110' : ''}
                    `}
                  >
                    <div
                      className={`
                        w-12 h-12 rounded-full flex items-center justify-center text-lg mb-2 transition-all duration-300
                        ${status === 'completed'
                          ? 'gradient-green text-white shadow-lg'
                          : status === 'active'
                            ? 'gradient-blue text-white animate-pulse shadow-lg'
                            : 'bg-gray-200 text-gray-500'
                        }
                      `}
                    >
                      {status === 'completed' ? '✓' : index + 1}
                    </div>
                    <div className="text-sm font-medium text-gray-800">{step.name}</div>
                    <div className="text-xs text-gray-500">{step.description}</div>
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* Overall Progress */}
          <div className="mt-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="font-medium text-gray-700">Overall Progress</span>
              <span className="font-bold text-purple-600">{progress.toFixed(0)}%</span>
            </div>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500 rounded-full"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            {currentStep && (
              <div className="mt-2 text-sm text-gray-600">
                Current Step: <span className="font-medium text-purple-600">{currentStep}</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="glass-card p-6 text-center">
          <div className="text-gray-500 mb-4">
            <div className="text-4xl mb-2">🚀</div>
            <h3 className="text-lg font-semibold mb-2">Ready to Start Pipeline</h3>
            <p className="text-sm">Configure your settings above and click "Start Pipeline" to begin the cryptocurrency analysis process.</p>
          </div>
        </div>
      )}
      
      {/* Execution Logs - Only show when there are logs or pipeline is running */}
      {(logs.length > 0 || isRunning) && (
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold">Execution Logs</h3>
            <button
              onClick={() => setLogs([])}
              className="px-3 py-1 text-sm glass-button rounded-lg"
            >
              Clear Logs
            </button>
          </div>
        <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div
                key={index}
                className={`
                  py-1 flex items-start gap-2
                  ${log.level === 'error' ? 'text-red-400' :
                    log.level === 'warning' ? 'text-yellow-400' :
                    log.level === 'success' ? 'text-green-400' :
                    'text-gray-300'}
                `}
              >
                <span className="opacity-50 text-xs min-w-[80px]">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="opacity-70 text-xs min-w-[60px]">
                  [{log.step || 'info'}]
                </span>
                <span className="flex-1">{log.message}</span>
              </div>
            ))
          ) : (
            <div className="text-gray-500 italic">No logs available...</div>
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
      )}
      
      {/* Pipeline Results */}
      {pipelineResults && (
        <div className="glass-card p-6">
          <h3 className="text-xl font-semibold mb-4">Pipeline Results</h3>
          
          {/* Results Tabs */}
          <div className="flex space-x-1 mb-6">
            {[
              { id: 'summary', name: '📊 Summary', icon: '📊' },
              { id: 'tickers', name: '📈 Tickers', icon: '📈' },
              { id: 'eda', name: '📊 EDA Results', icon: '📊' },
              { id: 'models', name: '🤖 Models', icon: '🤖' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveResultsTab(tab.id)}
                className={`
                  px-4 py-2 rounded-lg font-medium transition-all duration-300
                  ${activeResultsTab === tab.id
                    ? 'gradient-blue text-white shadow-lg'
                    : 'glass-button hover:shadow-md'
                  }
                `}
              >
                {tab.name}
              </button>
            ))}
          </div>
          
          {/* Results Content */}
          <div className="min-h-[400px]">
            {activeResultsTab === 'summary' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="glass p-4 rounded-lg">
                    <h4 className="font-semibold text-gray-700 mb-2">Database Records</h4>
                    <p className="text-2xl font-bold text-gradient">
                      {pipelineResults.summary?.data?.reduce((total, item) => total + (item.record_count || 0), 0)?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div className="glass p-4 rounded-lg">
                    <h4 className="font-semibold text-gray-700 mb-2">Available Tickers</h4>
                    <p className="text-2xl font-bold text-gradient">
                      {pipelineResults.tickers?.count || 'N/A'}
                    </p>
                  </div>
                  <div className="glass p-4 rounded-lg">
                    <h4 className="font-semibold text-gray-700 mb-2">EDA Analyses</h4>
                    <p className="text-2xl font-bold text-gradient">
                      {pipelineResults.edaResults?.length || 0}
                    </p>
                  </div>
                </div>
                
                <div className="glass p-4 rounded-lg">
                  <h4 className="font-semibold text-gray-700 mb-2">Pipeline Summary</h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p><strong>Execution Time:</strong> {new Date(pipelineResults.timestamp).toLocaleString()}</p>
                    <p><strong>Configuration:</strong> {config.tickers} | {config.sources} | {config.period}</p>
                    <p><strong>Features:</strong> EDA: {config.include_eda ? '✓' : '✗'} | Clustering: {config.include_clustering ? '✓' : '✗'}</p>
                  </div>
                </div>
              </div>
            )}
            
            {activeResultsTab === 'tickers' && (
              <div className="space-y-4">
                <div className="glass p-4 rounded-lg">
                  <h4 className="font-semibold text-gray-700 mb-4">Available Tickers</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                    {pipelineResults.tickers?.tickers?.slice(0, 24).map(ticker => (
                      <div key={ticker} className="px-3 py-2 glass rounded-lg text-center text-sm font-medium">
                        {ticker}
                      </div>
                    ))}
                  </div>
                  {pipelineResults.tickers?.tickers?.length > 24 && (
                    <p className="text-sm text-gray-500 mt-2">
                      +{pipelineResults.tickers.tickers.length - 24} more tickers
                    </p>
                  )}
                </div>
              </div>
            )}
            
            {activeResultsTab === 'eda' && (
              <div className="space-y-4">
                {pipelineResults.edaResults?.length > 0 ? (
                  pipelineResults.edaResults.map((result, index) => (
                    <div key={index} className="glass p-4 rounded-lg">
                      <h4 className="font-semibold text-gray-700 mb-2">
                        EDA Results for {result.ticker}
                      </h4>
                      <div className="text-sm text-gray-600">
                        <p><strong>Analysis Type:</strong> Exploratory Data Analysis</p>
                        <p><strong>Data Points:</strong> {result.data?.statistics?.total_records || result.data?.report?.num_records || 'N/A'}</p>
                        <p><strong>Date Range:</strong> {result.data?.statistics?.date_range?.start || result.data?.report?.date_range?.start || 'N/A'} to {result.data?.statistics?.date_range?.end || result.data?.report?.date_range?.end || 'N/A'}</p>
                        <p><strong>Success:</strong> {result.data?.success ? '✅' : '❌'}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-4xl mb-2">📊</div>
                    <p>No EDA results available</p>
                  </div>
                )}
              </div>
            )}
            
            {activeResultsTab === 'models' && (
              <div className="space-y-4">
                <div className="glass p-4 rounded-lg">
                  <h4 className="font-semibold text-gray-700 mb-4">Model Information</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h5 className="font-medium text-gray-600">Supported Models</h5>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                          <span className="text-sm">ARIMA</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                          <span className="text-sm">SARIMA</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                          <span className="text-sm">Random Forest</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                          <span className="text-sm">XGBoost</span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h5 className="font-medium text-gray-600">Training Status</h5>
                      <p className="text-sm text-gray-600">
                        Models will be trained on-demand when you access the Models tab or make predictions.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PipelineExecution;