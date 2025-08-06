import { create } from 'zustand';
import axios from 'axios';
import { toast } from 'react-toastify';
import { API_ENDPOINTS } from '../config/api';

const useCryptoStore = create((set, get) => ({
  // State
  selectedTicker: 'BTC',
  tickers: ['BTC', 'ETH', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK'],
  databaseSummary: null,
  edaResults: {},
  correlationData: null,
  clusteringData: null,
  modelMetrics: {},
  signals: {},
  news: [],
  whatIfResults: {},
  pipelineStatus: null,
  wsConnection: null,
  wsMessages: [],
  loading: false,
  error: null,
  
  // Actions
  actions: {
    // Set selected ticker
    setSelectedTicker: (ticker) => set({ selectedTicker: ticker }),
    
    // Fetch available tickers from database
    fetchTickers: async () => {
      try {
        const response = await axios.get(API_ENDPOINTS.DATABASE.TICKERS);
        if (response.data && response.data.tickers) {
          set({ tickers: response.data.tickers });
        }
      } catch (error) {
        console.error('Failed to fetch tickers:', error);
        // Use default tickers if fetch fails
      }
    },
    
    // Fetch database summary
    fetchDatabaseSummary: async () => {
      try {
        const response = await axios.get(API_ENDPOINTS.DATABASE.SUMMARY);
        if (response.data && response.data.data) {
          // Process the database summary to extract meaningful info
          const data = response.data.data;
          const uniqueTickers = [...new Set(data.map(item => item.ticker))];
          const sources = [...new Set(data.map(item => item.source))];
          const totalRecords = data.reduce((sum, item) => sum + (item.record_count || 0), 0);
          
          // Find latest date
          const allDates = data.map(item => new Date(item.last_date)).filter(date => !isNaN(date));
          const latestDate = allDates.length > 0 ? new Date(Math.max(...allDates)) : null;
          const earliestDate = allDates.length > 0 ? new Date(Math.min(...allDates)) : null;
          
          set({ 
            databaseSummary: {
              unique_tickers: uniqueTickers.length,
              sources: sources,
              total_records: totalRecords,
              date_range: {
                start: earliestDate?.toISOString(),
                end: latestDate?.toISOString()
              },
              raw_data: data
            }
          });
        }
      } catch (error) {
        console.error('Failed to fetch database summary:', error);
      }
    },
    
    // Fetch ticker data
    fetchTickerData: async (ticker, source = 'yfinance', limit = 90) => {
      try {
        set({ loading: true });
        const url = API_ENDPOINTS.DATABASE['ticker-data'](ticker);
        const response = await axios.get(url, {
          params: { source, limit }
        });
        return response.data;
      } catch (error) {
        console.error('Failed to fetch ticker data:', error);
        toast.error(`Failed to fetch data for ${ticker}`);
        return null;
      } finally {
        set({ loading: false });
      }
    },
    
    // Fetch EDA results
    fetchEDAResults: async (ticker) => {
      try {
        const response = await axios.get(`${API_ENDPOINTS.EDA}/${ticker}`, {
          params: { source: 'yfinance' }
        });
        
        if (response.data) {
          set(state => ({
            edaResults: {
              ...state.edaResults,
              [ticker]: response.data
            }
          }));
        }
      } catch (error) {
        console.error('Failed to fetch EDA results:', error);
      }
    },
    
    // Fetch correlation data
    fetchCorrelation: async (tickers, feature = 'Close') => {
      try {
        set({ loading: true });
        const response = await axios.get(API_ENDPOINTS.CORRELATION, {
          params: {
            tickers: tickers.join(','),
            feature,
            source: 'yfinance'
          }
        });
        
        if (response.data) {
          set({ correlationData: response.data });
        }
      } catch (error) {
        console.error('Failed to fetch correlation:', error);
        toast.error('Failed to fetch correlation analysis');
      } finally {
        set({ loading: false });
      }
    },
    
    // Fetch clustering data
    fetchClustering: async (source = 'yfinance') => {
      try {
        set({ loading: true });
        const response = await axios.get(API_ENDPOINTS.CLUSTERING, {
          params: { source }
        });
        
        if (response.data) {
          set({ clusteringData: response.data });
        }
      } catch (error) {
        console.error('Failed to fetch clustering:', error);
        toast.error('Failed to fetch clustering analysis');
      } finally {
        set({ loading: false });
      }
    },
    
    // Train models
    trainModels: async (ticker, feature = 'Close', testSize = 0.2) => {
      try {
        set({ loading: true });
        const response = await axios.get(API_ENDPOINTS.TRAIN(ticker), {
          params: { feature, test_size: testSize, source: 'yfinance' },
          timeout: 60000
        });
        
        if (response.data && response.data.metrics) {
          set(state => ({
            modelMetrics: {
              ...state.modelMetrics,
              [ticker]: response.data.metrics
            }
          }));
          toast.success(`Models trained successfully for ${ticker}`);
        }
      } catch (error) {
        console.error('Failed to train models:', error);
        toast.error('Failed to train models');
      } finally {
        set({ loading: false });
      }
    },
    
    // Fetch signals
    fetchSignals: async (ticker, modelType = 'arima', periods = 7) => {
      try {
        const response = await axios.get(API_ENDPOINTS.SIGNALS(ticker), {
          params: { 
            model_type: modelType, 
            periods, 
            source: 'yfinance' 
          }
        });
        
        if (response.data) {
          set(state => ({
            signals: {
              ...state.signals,
              [ticker]: response.data
            }
          }));
        }
      } catch (error) {
        console.error('Failed to fetch signals:', error);
        toast.error('Failed to fetch trading signals');
      }
    },
    
    // Fetch news
    fetchNews: async (limit = 20) => {
      try {
        const response = await axios.get(API_ENDPOINTS.NEWS.FEED, {
          params: { limit }
        });
        
        if (response.data && response.data.articles) {
          set({ news: response.data.articles });
        }
      } catch (error) {
        console.error('Failed to fetch news:', error);
        toast.error('Failed to fetch news feed');
      }
    },
    
    // Run what-if scenario
    runWhatIfScenario: async (type, params) => {
      try {
        set({ loading: true });
        const endpoint = API_ENDPOINTS.WHATIF[type.toUpperCase().replace('-', '_')];
        const response = await axios.get(endpoint, { params });
        
        if (response.data) {
          set(state => ({
            whatIfResults: {
              ...state.whatIfResults,
              [type]: response.data
            }
          }));
        }
      } catch (error) {
        console.error('Failed to run what-if scenario:', error);
        toast.error('Failed to run scenario analysis');
      } finally {
        set({ loading: false });
      }
    },
    
    // Initialize WebSocket connection
    initWebSocket: async () => {
      try {
        const ws = new WebSocket(API_ENDPOINTS.WEBSOCKET);
        let hasShownError = false;
        
        ws.onopen = () => {
          console.log('WebSocket connected');
          set({ wsConnection: ws });
          // Reset error flag when successfully connected
          hasShownError = false;
        };
        
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            set(state => ({
              wsMessages: [...state.wsMessages, message],
              pipelineStatus: message.pipeline_status || message.status || state.pipelineStatus
            }));
            
            // Show toast for important messages
            if (message.level === 'error') {
              toast.error(message.message || message.content);
            } else if (message.level === 'success') {
              toast.success(message.message || message.content);
            } else if (message.level === 'info') {
              toast.info(message.message || message.content);
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          // Only show error toast once
          if (!hasShownError) {
            toast.error('WebSocket connection error. Retrying in background...');
            hasShownError = true;
          }
        };
        
        ws.onclose = () => {
          console.log('WebSocket disconnected');
          set({ wsConnection: null });
          // Only show disconnection message once
          if (!hasShownError) {
            toast.warn('WebSocket disconnected. Reconnecting in background...');
            hasShownError = true;
          }
          // Attempt to reconnect after 5 seconds
          setTimeout(() => {
            get().actions.initWebSocket();
          }, 5000);
        };
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
      }
    },
    
    // Run full pipeline
    runPipeline: async (params) => {
      try {
        set({ loading: true, pipelineStatus: 'starting' });
        const response = await axios.post(API_ENDPOINTS.PIPELINE.FULL, null, {
          params,
          timeout: 300000 // 5 minute timeout
        });
        
        if (response.data) {
          set({ pipelineStatus: 'completed' });
          toast.success('Pipeline execution completed!');
          return response.data;
        }
      } catch (error) {
        console.error('Pipeline execution failed:', error);
        set({ pipelineStatus: 'failed' });
        toast.error('Pipeline execution failed');
        throw error;
      } finally {
        set({ loading: false });
      }
    },
    
    // Clear WebSocket messages
    clearWSMessages: () => set({ wsMessages: [] }),
    
    // Clear error
    clearError: () => set({ error: null }),
  }
}));

export default useCryptoStore;