import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { ToastConfig } from './components/ToastConfig';

import useCryptoStore from './stores/cryptoStore';
import ErrorBoundary from './components/ErrorBoundary';
import PipelineExecution from './components/PipelineExecution';
import EDAVisualization from './components/EDAVisualization';
import CorrelationAnalysis from './components/CorrelationAnalysis';
import ClusteringAnalysis from './components/ClusteringAnalysis';
import ModelComparison from './components/ModelComparison';
import TradingSignals from './components/TradingSignals';
import WhatIfScenarios from './components/WhatIfScenarios';
import NewsFeed from './components/NewsFeed';
import './App.css';

// Loading Component
const Loading = () => (
  <div className="flex items-center justify-center min-h-screen bg-pattern">
    <div className="text-center">
      <div className="w-20 h-20 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
      <p className="text-lg text-gray-600">Loading ICT Platform...</p>
    </div>
  </div>
);

// Dashboard Layout Component
const DashboardLayout = ({ children }) => {
  const { selectedTicker, tickers, databaseSummary, actions } = useCryptoStore();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  const navItems = [
    { path: '/pipeline', name: '🚀 Pipeline', gradient: 'gradient-blue' },
    { path: '/eda', name: '📊 EDA Analysis', gradient: 'gradient-green' },
    { path: '/correlation', name: '🔗 Correlation', gradient: 'gradient-purple' },
    { path: '/clustering', name: '🎯 Clustering', gradient: 'gradient-orange' },
    { path: '/models', name: '🤖 Models', gradient: 'gradient-blue' },
    { path: '/signals', name: '📈 Signals', gradient: 'gradient-green' },
    { path: '/whatif', name: '🔮 What-If', gradient: 'gradient-purple' },
    { path: '/news', name: '📰 News', gradient: 'gradient-orange' }
  ];
  
  return (
    <div className="min-h-screen bg-pattern">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-float"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-float" style={{ animationDelay: '4s' }}></div>
      </div>
      
      {/* Header */}
      <header className="glass-card mx-4 mt-4 mb-2 relative z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 glass-button rounded-lg mr-4 hover:shadow-lg transition-all duration-300"
              >
                <svg className="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d={sidebarOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
                </svg>
              </button>
              <h1 className="text-3xl font-bold gradient-text-animate">
                🪙 Intelligent Coin Trading Platform
              </h1>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Ticker Selector */}
              <div className="relative">
                <select
                  value={selectedTicker}
                  onChange={(e) => actions.setSelectedTicker(e.target.value)}
                  className="px-6 py-3 pr-10 glass-dropdown text-gray-700 font-semibold transition-all duration-300 appearance-none cursor-pointer"
                >
                  {tickers.map(ticker => (
                    <option key={ticker} value={ticker}>{ticker}</option>
                  ))}
                </select>
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
                  <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
              
              {/* Database Status */}
              {databaseSummary && (
                <div className="flex items-center gap-2 px-4 py-2 glass-card animate-glow">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50" />
                  <span className="font-semibold text-gray-700">
                    {databaseSummary.total_records?.toLocaleString()} records
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>
      
      <div className="flex gap-4 px-4 relative z-10">
        {/* Sidebar */}
        <aside className={`
          ${sidebarOpen ? 'w-72' : 'w-0'} 
          transition-all duration-500 overflow-hidden
        `}>
          <div className="glass-card h-[calc(100vh-120px)] overflow-y-auto">
            <nav className="p-6">
              <h2 className="text-sm font-bold text-gray-600 uppercase tracking-wider mb-4">
                Analysis Tools
              </h2>
              <ul className="space-y-2">
                {navItems.map(item => (
                  <li key={item.path}>
                    <NavLink
                      to={item.path}
                      className={({ isActive }) => `
                        block px-4 py-3 rounded-xl transition-all duration-300 font-medium
                        ${isActive
                          ? `${item.gradient} text-white shadow-lg transform scale-105`
                          : 'glass-button text-gray-700 hover:shadow-md hover:transform hover:scale-102'
                        }
                      `}
                    >
                      {item.name}
                    </NavLink>
                  </li>
                ))}
              </ul>
              
              {/* Database Summary */}
              {databaseSummary && (
                <div className="mt-8 p-4 glass rounded-xl">
                  <h3 className="text-sm font-bold text-gray-700 mb-3">Database Summary</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Tickers:</span>
                      <span className="font-semibold text-gray-800">{databaseSummary.unique_tickers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Sources:</span>
                      <span className="font-semibold text-gray-800">
                        {databaseSummary.sources?.join(', ')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Latest:</span>
                      <span className="font-semibold text-gray-800">
                        {databaseSummary.date_range?.end && 
                          new Date(databaseSummary.date_range.end).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </nav>
          </div>
        </aside>
        
        {/* Main Content */}
        <main className="flex-1">
          <div className="glass-card min-h-[calc(100vh-120px)] p-6">
            {children}
          </div>
        </main>
      </div>
      
      {/* Footer */}
      <footer className="glass-card mx-4 mt-4 mb-4 relative z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 font-medium">
              © 2024 SOLiGence - COM724 Assessment
            </span>
            <div className="flex items-center gap-4">
              <span className="px-3 py-1 glass-button rounded-full text-xs font-semibold">
                Powered by YFinance & Binance APIs
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

function App() {
  const { actions } = useCryptoStore();
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    initializeApp();
  }, []);
  
  const initializeApp = async () => {
    try {
      await actions.fetchDatabaseSummary();
      await actions.fetchTickers();
      toast.success('✅ Platform initialized successfully!');
    } catch (error) {
      console.error('Failed to initialize app:', error);
      toast.error('Failed to initialize platform. Please refresh.');
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) {
    return <Loading />;
  }
  
  return (
    <Router>
      <ToastConfig />
      
      <Routes>
        <Route path="/" element={<Navigate to="/pipeline" replace />} />
        
        <Route path="/pipeline" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="Pipeline component encountered an error">
              <PipelineExecution />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/eda" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="EDA component encountered an error">
              <EDAVisualization />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/correlation" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="Correlation component encountered an error">
              <CorrelationAnalysis />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/clustering" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="Clustering component encountered an error">
              <ClusteringAnalysis />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/models" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="Models component encountered an error">
              <ModelComparison />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/signals" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="Signals component encountered an error">
              <TradingSignals />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/whatif" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="What-If component encountered an error">
              <WhatIfScenarios />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="/news" element={
          <DashboardLayout>
            <ErrorBoundary fallbackMessage="News component encountered an error">
              <NewsFeed />
            </ErrorBoundary>
          </DashboardLayout>
        } />
        
        <Route path="*" element={<Navigate to="/pipeline" replace />} />
      </Routes>
    </Router>
  );
}

export default App;