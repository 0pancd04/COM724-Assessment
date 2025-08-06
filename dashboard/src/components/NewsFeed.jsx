import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const NewsFeed = () => {
  const { selectedTicker, news, marketSentiment, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all');
  const [refreshing, setRefreshing] = useState(false);
  
  useEffect(() => {
    loadNews();
  }, [selectedTicker, filter]);
  
  const loadNews = async () => {
    setLoading(true);
    try {
      await actions.fetchNewsFeed(filter === 'ticker' ? selectedTicker : null, 20, false);
      await actions.fetchMarketSentiment(filter === 'ticker' ? selectedTicker : null);
    } catch (error) {
      console.error('Failed to load news:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await actions.fetchNewsFeed(filter === 'ticker' ? selectedTicker : null, 20, true);
      await actions.fetchMarketSentiment(filter === 'ticker' ? selectedTicker : null);
    } catch (error) {
      console.error('Failed to refresh news:', error);
    } finally {
      setRefreshing(false);
    }
  };
  
  const getSentimentColor = (sentiment) => {
    if (sentiment > 0.3) return 'text-green-600';
    if (sentiment < -0.3) return 'text-red-600';
    return 'text-gray-600';
  };
  
  const getSentimentLabel = (sentiment) => {
    if (sentiment > 0.3) return 'Positive';
    if (sentiment < -0.3) return 'Negative';
    return 'Neutral';
  };
  
  const getSentimentIcon = (sentiment) => {
    if (sentiment > 0.3) {
      return (
        <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z" clipRule="evenodd" />
        </svg>
      );
    } else if (sentiment < -0.3) {
      return (
        <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-7.536 4.879a1 1 0 001.415 1.414 3 3 0 014.242 0 1 1 0 001.415-1.414 5 5 0 00-7.072 0z" clipRule="evenodd" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-7 4a1 1 0 011-1h6a1 1 0 110 2H8a1 1 0 01-1-1z" clipRule="evenodd" />
      </svg>
    );
  };
  
  const prepareSentimentChart = () => {
    if (!marketSentiment?.history) return null;
    
    return {
      series: [{
        name: 'Sentiment Score',
        data: marketSentiment.history.map(h => ({
          x: new Date(h.date),
          y: h.score
        }))
      }],
      options: {
        chart: {
          type: 'area',
          height: 200,
          toolbar: { show: false }
        },
        stroke: {
          curve: 'smooth',
          width: 2
        },
        fill: {
          type: 'gradient',
          gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.7,
            opacityTo: 0.3,
            stops: [0, 90, 100]
          }
        },
        xaxis: {
          type: 'datetime'
        },
        yaxis: {
          min: -1,
          max: 1,
          labels: {
            formatter: (val) => val.toFixed(1)
          }
        },
        annotations: {
          yaxis: [
            {
              y: 0,
              borderColor: '#999',
              label: {
                text: 'Neutral',
                style: {
                  color: '#999',
                  background: '#fff'
                }
              }
            }
          ]
        },
        colors: ['#00E396']
      }
    };
  };
  
  const sentimentChart = prepareSentimentChart();
  
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    
    if (diffHours < 1) {
      const diffMins = Math.floor(diffMs / (1000 * 60));
      return `${diffMins} minutes ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hours ago`;
    } else {
      const diffDays = Math.floor(diffHours / 24);
      return `${diffDays} days ago`;
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          📰 Crypto News & Market Sentiment
        </h2>
        <div className="flex gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-3 py-2 glass-dropdown"
          >
            <option value="all">All News</option>
            <option value="ticker">{selectedTicker} News</option>
          </select>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className={`
              px-4 py-2 rounded-md font-medium transition-colors
              ${refreshing
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
              }
            `}
          >
            {refreshing ? (
              <span className="flex items-center">
                <svg className="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Refreshing...
              </span>
            ) : (
              'Refresh'
            )}
          </button>
        </div>
      </div>
      
      {/* Market Sentiment Overview */}
      {marketSentiment && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">Market Sentiment</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-600">Overall Sentiment</p>
                {getSentimentIcon(marketSentiment.overall_sentiment)}
              </div>
              <p className={`text-2xl font-bold ${getSentimentColor(marketSentiment.overall_sentiment)}`}>
                {getSentimentLabel(marketSentiment.overall_sentiment)}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Score: {marketSentiment.overall_sentiment?.toFixed(2)}
              </p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-2">Positive Articles</p>
              <p className="text-2xl font-bold text-green-600">
                {marketSentiment.positive_count || 0}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                {((marketSentiment.positive_count / marketSentiment.total_articles) * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-2">Neutral Articles</p>
              <p className="text-2xl font-bold text-gray-600">
                {marketSentiment.neutral_count || 0}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                {((marketSentiment.neutral_count / marketSentiment.total_articles) * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="p-4 bg-red-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-2">Negative Articles</p>
              <p className="text-2xl font-bold text-red-600">
                {marketSentiment.negative_count || 0}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                {((marketSentiment.negative_count / marketSentiment.total_articles) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          
          {/* Sentiment Trend Chart */}
          {sentimentChart && (
            <div>
              <h4 className="text-md font-medium mb-2 text-gray-700">Sentiment Trend</h4>
              <ReactApexChart
                options={sentimentChart.options}
                series={sentimentChart.series}
                type="area"
                height={200}
              />
            </div>
          )}
        </div>
      )}
      
      {/* News Feed */}
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Loading news feed...</p>
          </div>
        </div>
      )}
      
      {!loading && news && news.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3 text-gray-700">Latest News</h3>
          <div className="space-y-4">
            {news.map((article, idx) => (
              <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <a
                      href={article.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-lg font-semibold text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      {article.title}
                    </a>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-sm text-gray-500">
                        {article.source}
                      </span>
                      <span className="text-sm text-gray-500">
                        {formatDate(article.published)}
                      </span>
                      {article.sentiment_score !== undefined && (
                        <div className="flex items-center gap-1">
                          {getSentimentIcon(article.sentiment_score)}
                          <span className={`text-sm font-medium ${getSentimentColor(article.sentiment_score)}`}>
                            {getSentimentLabel(article.sentiment_score)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <p className="text-gray-700 line-clamp-3 mb-3">
                  {article.summary}
                </p>
                
                {article.ticker_mentions && article.ticker_mentions.length > 0 && (
                  <div className="flex gap-2 flex-wrap">
                    {article.ticker_mentions.map((ticker, tidx) => (
                      <span
                        key={tidx}
                        className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded"
                      >
                        {ticker}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {!loading && (!news || news.length === 0) && (
        <div className="text-center py-12">
          <p className="text-gray-600">No news articles available</p>
          <button
            onClick={handleRefresh}
            className="mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Load News
          </button>
        </div>
      )}
    </div>
  );
};

export default NewsFeed;
