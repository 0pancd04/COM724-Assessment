// API Configuration
const API_BASE_URL = 'http://localhost:8000';

export const API_ENDPOINTS = {
  // Database endpoints
  DATABASE: `${API_BASE_URL}/database`,
  DATABASE_SUMMARY: `${API_BASE_URL}/database/summary`,
  DATABASE_TICKERS: `${API_BASE_URL}/database/tickers`,
  
  // Data download endpoints
  DOWNLOAD: {
    SINGLE: (ticker) => `${API_BASE_URL}/download/${ticker}`,
    TOP30: `${API_BASE_URL}/download/top30`,
    UNIFIED: `${API_BASE_URL}/download/unified`,
  },
  
  // Analysis endpoints
  EDA: `${API_BASE_URL}/eda`,
  EDA_CHARTS: `${API_BASE_URL}/eda/charts`,
  CORRELATION: `${API_BASE_URL}/correlation`,
  CLUSTERING: `${API_BASE_URL}/clustering`,
  ANALYSIS: `${API_BASE_URL}/analysis`,
  
  // Model endpoints
  TRAIN: (ticker) => `${API_BASE_URL}/train/${ticker}`,
  FORECAST: (ticker) => `${API_BASE_URL}/forecast/${ticker}`,
  SIGNALS: (ticker) => `${API_BASE_URL}/signals/${ticker}`,
  INDICATORS: (ticker) => `${API_BASE_URL}/indicators/${ticker}`,
  BACKTEST: (ticker) => `${API_BASE_URL}/backtest/${ticker}`,
  
  // Pipeline endpoints
  PIPELINE: {
    FULL: `${API_BASE_URL}/pipeline/full`,
    STATUS: `${API_BASE_URL}/pipeline/status`,
    PREPROCESSING: `${API_BASE_URL}/pipeline/preprocessing`,
    DOWNLOAD_TRAIN: `${API_BASE_URL}/pipeline/download-and-train`,
  },
  
  // News endpoints
  NEWS: {
    FEED: `${API_BASE_URL}/news/feed`,
    SENTIMENT: `${API_BASE_URL}/news/sentiment`,
  },
  
  // What-if scenarios
  WHATIF: {
    PRICE_CHANGE: `${API_BASE_URL}/whatif/price-change`,
    TRADING_STRATEGY: `${API_BASE_URL}/whatif/trading-strategy`,
    PORTFOLIO: `${API_BASE_URL}/whatif/portfolio-allocation`,
    DCA: `${API_BASE_URL}/whatif/dca-investment`,
  },
  
  // WebSocket
  WEBSOCKET: `ws://localhost:8000/ws`,
};

// Helper function to build URL with query parameters
export const buildUrl = (endpoint, params = {}) => {
  const url = new URL(endpoint);
  Object.keys(params).forEach(key => {
    if (params[key] !== undefined && params[key] !== null) {
      url.searchParams.append(key, params[key]);
    }
  });
  return url.toString();
};

// API request helper with error handling
export const apiRequest = async (url, options = {}) => {
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API Request Failed:', error);
    throw error;
  }
};

export default API_ENDPOINTS;