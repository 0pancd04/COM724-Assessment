import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import useCryptoStore from '../stores/cryptoStore';

const ClusteringAnalysis = () => {
  const { clusteringResults, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    source: 'yfinance',
    max_days: 90,
    n_clusters: 4,
    algorithm: 'kmeans'
  });
  
  const loadClustering = async () => {
    setLoading(true);
    try {
      await actions.fetchClusteringAnalysis(params);
    } catch (error) {
      console.error('Failed to load clustering analysis:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const prepareScatterPlot = () => {
    if (!clusteringResults?.pca_data || !clusteringResults?.clusters) return null;
    
    const clusters = [...new Set(clusteringResults.clusters)];
    const series = clusters.map(cluster => {
      const clusterData = clusteringResults.pca_data
        .filter((_, idx) => clusteringResults.clusters[idx] === cluster)
        .map(point => ({
          x: point[0],
          y: point[1]
        }));
      
      return {
        name: `Cluster ${cluster + 1}`,
        data: clusterData
      };
    });
    
    return {
      series,
      options: {
        chart: {
          type: 'scatter',
          height: 400,
          toolbar: { show: true }
        },
        xaxis: {
          title: { text: 'First Principal Component' },
          tickAmount: 7
        },
        yaxis: {
          title: { text: 'Second Principal Component' },
          tickAmount: 7
        },
        legend: {
          position: 'top'
        },
        colors: ['#008FFB', '#00E396', '#FEB019', '#FF4560', '#775DD0']
      }
    };
  };
  
  const prepareSilhouetteChart = () => {
    if (!clusteringResults?.silhouette_scores) return null;
    
    return {
      series: [{
        name: 'Silhouette Score',
        data: clusteringResults.silhouette_scores.map((score, idx) => ({
          x: `Sample ${idx + 1}`,
          y: score
        }))
      }],
      options: {
        chart: {
          type: 'bar',
          height: 300,
          toolbar: { show: false }
        },
        plotOptions: {
          bar: {
            colors: {
              ranges: [{
                from: -1,
                to: 0,
                color: '#FF4560'
              }, {
                from: 0,
                to: 0.5,
                color: '#FEB019'
              }, {
                from: 0.5,
                to: 1,
                color: '#00E396'
              }]
            }
          }
        },
        xaxis: {
          labels: { show: false }
        },
        yaxis: {
          title: { text: 'Silhouette Score' },
          min: -1,
          max: 1
        }
      }
    };
  };
  
  const scatterPlot = prepareScatterPlot();
  const silhouetteChart = prepareSilhouetteChart();
  
  const getClusterColor = (cluster) => {
    const colors = ['bg-blue-100', 'bg-green-100', 'bg-yellow-100', 'bg-red-100', 'bg-purple-100'];
    return colors[cluster % colors.length];
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        🎯 Clustering & Grouping Analysis
      </h2>
      
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Data Source
          </label>
          <select
            className="w-full px-3 py-2 glass-dropdown"
            value={params.source}
            onChange={(e) => setParams({...params, source: e.target.value})}
          >
            <option value="yfinance">YFinance</option>
            <option value="binance">Binance</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Algorithm
          </label>
          <select
            className="w-full px-3 py-2 glass-dropdown"
            value={params.algorithm}
            onChange={(e) => setParams({...params, algorithm: e.target.value})}
          >
            <option value="kmeans">K-Means</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Number of Clusters
          </label>
          <input
            type="number"
            min="2"
            max="10"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={params.n_clusters}
            onChange={(e) => setParams({...params, n_clusters: parseInt(e.target.value)})}
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={loadClustering}
            disabled={loading}
            className={`
              w-full px-4 py-2 rounded-md font-medium transition-colors
              ${loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
              }
            `}
          >
            {loading ? 'Analyzing...' : 'Run Clustering'}
          </button>
        </div>
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Performing clustering analysis...</p>
          </div>
        </div>
      )}
      
      {!loading && clusteringResults && (
        <>
          {/* Clustering Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Silhouette Score</p>
              <p className="text-2xl font-bold text-blue-600">
                {clusteringResults.silhouette_score?.toFixed(3) || 'N/A'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {clusteringResults.silhouette_score > 0.5 ? 'Good' :
                 clusteringResults.silhouette_score > 0.25 ? 'Fair' : 'Poor'} clustering
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Number of Clusters</p>
              <p className="text-2xl font-bold text-green-600">
                {clusteringResults.n_clusters || params.n_clusters}
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Inertia</p>
              <p className="text-2xl font-bold text-purple-600">
                {clusteringResults.inertia?.toFixed(2) || 'N/A'}
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Davies-Bouldin</p>
              <p className="text-2xl font-bold text-orange-600">
                {clusteringResults.davies_bouldin?.toFixed(3) || 'N/A'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Lower is better
              </p>
            </div>
          </div>
          
          {/* Scatter Plot */}
          {scatterPlot && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                PCA Visualization
              </h3>
              <ReactApexChart
                options={scatterPlot.options}
                series={scatterPlot.series}
                type="scatter"
                height={400}
              />
            </div>
          )}
          
          {/* Cluster Assignments */}
          {clusteringResults.cluster_assignments && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Cryptocurrency Clusters
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(clusteringResults.cluster_assignments).map(([cluster, tickers]) => (
                  <div key={cluster} className={`p-4 rounded-lg border ${getClusterColor(parseInt(cluster))}`}>
                    <h4 className="font-semibold mb-2">Cluster {parseInt(cluster) + 1}</h4>
                    <div className="space-y-1">
                      {tickers.slice(0, 5).map(ticker => (
                        <div key={ticker} className="text-sm text-gray-700">
                          {ticker}
                        </div>
                      ))}
                      {tickers.length > 5 && (
                        <div className="text-sm text-gray-500 italic">
                          +{tickers.length - 5} more
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Silhouette Analysis */}
          {silhouetteChart && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Silhouette Analysis
              </h3>
              <ReactApexChart
                options={silhouetteChart.options}
                series={silhouetteChart.series}
                type="bar"
                height={300}
              />
            </div>
          )}
          
          {/* Cluster Characteristics */}
          {clusteringResults.cluster_characteristics && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Cluster Characteristics
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Cluster
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Size
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Avg Return
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Avg Volatility
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Risk Level
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {Object.entries(clusteringResults.cluster_characteristics).map(([cluster, chars]) => (
                      <tr key={cluster}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          Cluster {parseInt(cluster) + 1}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {chars.size}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                          chars.avg_return > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {chars.avg_return > 0 ? '+' : ''}{(chars.avg_return * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(chars.avg_volatility * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            chars.risk_level === 'High' ? 'bg-red-100 text-red-800' :
                            chars.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {chars.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ClusteringAnalysis;
