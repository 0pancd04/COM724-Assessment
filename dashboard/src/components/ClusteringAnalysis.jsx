import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import { toast } from 'react-toastify';
import useCryptoStore from '../stores/cryptoStore';

const ClusteringAnalysis = () => {
  const { clusteringData, actions } = useCryptoStore();
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    source: 'yfinance',
    max_days: 90,
    n_clusters: 4,
    algorithm: 'kmeans',
    feature: 'Close'
  });
  
  const loadClustering = async () => {
    if (!params.n_clusters || params.n_clusters < 2 || params.n_clusters > 10) {
      toast.warn('Please select between 2 and 10 clusters');
      return;
    }
    try {
      setLoading(true);
      await actions.fetchClusteringAnalysis(params);
      toast.success('Clustering analysis completed successfully!');
    } catch (error) {
      console.error('Failed to load clustering analysis:', error);
      toast.error('Failed to perform clustering analysis');
    } finally {
      setLoading(false);
    }
  };
  
  const prepareScatterPlot = () => {
    if (!clusteringData?.report?.chart_file_json) return null;
    
    try {
      const chartData = JSON.parse(clusteringData.report.chart_file_json);
      const data = chartData.data;
      
      // Group data by cluster
      const clusterGroups = {};
      data.forEach(point => {
        const cluster = point.customdata[0];
        if (!clusterGroups[cluster]) {
          clusterGroups[cluster] = [];
        }
        clusterGroups[cluster].push({
          x: point.x,
          y: point.y,
          ticker: point.hovertext
        });
      });
      
      // Create series for each cluster
      const series = Object.entries(clusterGroups).map(([cluster, points]) => ({
        name: `Cluster ${parseInt(cluster) + 1}`,
        data: points
      }));
      
      return {
        series,
        options: {
          chart: {
            type: 'scatter',
            height: 400,
            toolbar: { show: true },
            fontFamily: 'Inter, sans-serif',
            background: 'transparent',
            foreColor: '#374151',
            animations: {
              enabled: true,
              easing: 'easeinout',
              speed: 800,
              animateGradually: {
                enabled: true,
                delay: 150
              },
              dynamicAnimation: {
                enabled: true,
                speed: 350
              }
            }
          },
          title: {
            text: `Cryptocurrency Clusters (${clusteringData.report.chosen_algorithm || 'PCA'})`,
            align: 'left',
            style: {
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#1f2937'
            }
          },
          subtitle: {
            text: `Silhouette Score: ${(clusteringData.report.chosen_silhouette || 0).toFixed(3)}`,
            align: 'left',
            style: {
              fontSize: '14px',
              color: '#4b5563'
            }
          },
          xaxis: {
            title: { 
              text: 'First Principal Component',
              style: {
                fontSize: '12px',
                color: '#4b5563'
              }
            },
            tickAmount: 7,
            labels: {
              style: {
                colors: '#4b5563'
              }
            }
          },
          yaxis: {
            title: { 
              text: 'Second Principal Component',
              style: {
                fontSize: '12px',
                color: '#4b5563'
              }
            },
            tickAmount: 7,
            labels: {
              style: {
                colors: '#4b5563'
              }
            }
          },
          legend: {
            position: 'top',
            horizontalAlign: 'left',
            labels: {
              colors: '#4b5563'
            }
          },
          markers: {
            size: 6,
            strokeWidth: 0,
            hover: {
              size: 8,
              sizeOffset: 3
            }
          },
          grid: {
            borderColor: '#e5e7eb',
            strokeDashArray: 4,
            xaxis: {
              lines: {
                show: true
              }
            },
            yaxis: {
              lines: {
                show: true
              }
            }
          },
          tooltip: {
            theme: 'light',
            custom: ({ seriesIndex, dataPointIndex, w }) => {
              const point = w.config.series[seriesIndex].data[dataPointIndex];
              return `
                <div class="p-2 bg-white shadow-lg rounded-lg border border-gray-200">
                  <div class="font-semibold text-gray-800">${point.ticker}</div>
                  <div class="text-sm text-gray-600">
                    <div>Component 1: ${point.x.toFixed(3)}</div>
                    <div>Component 2: ${point.y.toFixed(3)}</div>
                  </div>
                </div>
              `;
            }
          },
          colors: ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6']
        }
      };
    } catch (error) {
      console.error('Error preparing scatter plot:', error);
      return null;
    }
  };
  
  const prepareSilhouetteChart = () => {
    if (!clusteringData?.report) return null;
    
    // Get silhouette scores for each algorithm
    const scores = {
      'KMeans': clusteringData.report.KMeans_silhouette || 0,
      'Agglomerative': clusteringData.report.Agglomerative_silhouette || 0,
      'DBSCAN': clusteringData.report.DBSCAN_silhouette || 0
    };
    
    return {
      series: [{
        name: 'Silhouette Score',
        data: Object.entries(scores).map(([algo, score]) => ({
          x: algo,
          y: score
        }))
      }],
      options: {
        chart: {
          type: 'bar',
          height: 300,
          toolbar: { show: false },
          fontFamily: 'Inter, sans-serif',
          background: 'transparent',
          foreColor: '#374151',
          animations: {
            enabled: true,
            easing: 'easeinout',
            speed: 800,
            animateGradually: {
              enabled: true,
              delay: 150
            }
          }
        },
        title: {
          text: 'Algorithm Comparison',
          align: 'left',
          style: {
            fontSize: '16px',
            fontWeight: 'bold',
            color: '#1f2937'
          }
        },
        subtitle: {
          text: 'Silhouette Scores by Algorithm',
          align: 'left',
          style: {
            fontSize: '14px',
            color: '#4b5563'
          }
        },
        plotOptions: {
          bar: {
            borderRadius: 4,
            columnWidth: '60%',
            colors: {
              ranges: [{
                from: -1,
                to: 0,
                color: '#ef4444'
              }, {
                from: 0,
                to: 0.5,
                color: '#f59e0b'
              }, {
                from: 0.5,
                to: 1,
                color: '#10b981'
              }]
            },
            dataLabels: {
              position: 'top'
            }
          }
        },
        dataLabels: {
          enabled: true,
          formatter: (val) => val.toFixed(3),
          offsetY: -20,
          style: {
            fontSize: '12px',
            colors: ['#4b5563']
          }
        },
        xaxis: {
          categories: Object.keys(scores),
          labels: {
            style: {
              colors: '#4b5563',
              fontSize: '12px'
            }
          },
          axisBorder: {
            show: true,
            color: '#e5e7eb'
          },
          axisTicks: {
            show: true,
            color: '#e5e7eb'
          }
        },
        yaxis: {
          title: { 
            text: 'Silhouette Score',
            style: {
              fontSize: '12px',
              color: '#4b5563'
            }
          },
          min: -1,
          max: 1,
          tickAmount: 5,
          labels: {
            style: {
              colors: '#4b5563'
            },
            formatter: (val) => val.toFixed(2)
          }
        },
        grid: {
          borderColor: '#e5e7eb',
          strokeDashArray: 4,
          xaxis: {
            lines: {
              show: true
            }
          }
        },
        tooltip: {
          theme: 'light',
          y: {
            title: {
              formatter: () => 'Score: '
            },
            formatter: (val) => val.toFixed(3)
          },
          marker: {
            show: false
          },
          style: {
            fontSize: '12px'
          }
        },
        states: {
          hover: {
            filter: {
              type: 'lighten',
              value: 0.1
            }
          },
          active: {
            filter: {
              type: 'darken',
              value: 0.1
            }
          }
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
              w-full px-4 py-2 rounded-lg font-medium transition-all duration-300
              ${loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white hover:shadow-lg transform hover:scale-105'
              }
            `}
          >
            {loading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Analyzing...</span>
              </div>
            ) : (
              <div className="flex items-center justify-center gap-2">
                <span>Run Clustering Analysis</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            )}
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
      
      {!loading && clusteringData && (
        <>
          {/* Clustering Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Silhouette Score</p>
              <p className="text-2xl font-bold text-blue-600">
                {clusteringData.report?.chosen_silhouette?.toFixed(3) || 'N/A'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {clusteringData.report?.chosen_silhouette > 0.5 ? 'Good' :
                 clusteringData.report?.chosen_silhouette > 0.25 ? 'Fair' : 'Poor'} clustering
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Number of Clusters</p>
              <p className="text-2xl font-bold text-green-600">
                {clusteringData.report?.n_clusters || params.n_clusters}
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Algorithm</p>
              <p className="text-2xl font-bold text-purple-600">
                {clusteringData.report?.chosen_clustering_method || 'N/A'}
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Data Source</p>
              <p className="text-2xl font-bold text-orange-600">
                {clusteringData.report?.source || params.source}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Feature: {clusteringData.report?.feature || params.feature}
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
          {clusteringData?.report?.cluster_assignments && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Cryptocurrency Clusters
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(clusteringData.report.cluster_assignments).map(([cluster, tickers]) => {
                  const clusterStats = clusteringData.report.cluster_characteristics?.[cluster] || {};
                  return (
                    <div key={cluster} className={`p-4 rounded-lg border ${getClusterColor(parseInt(cluster))}`}>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold">Cluster {parseInt(cluster) + 1}</h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          clusterStats.risk_level === 'High' ? 'bg-red-100 text-red-800' :
                          clusterStats.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {clusterStats.risk_level || 'Unknown'} Risk
                        </span>
                      </div>
                      <div className="space-y-2 mb-3">
                        <div className="text-sm text-gray-600">
                          <span className="font-medium">Return:</span>{' '}
                          <span className={clusterStats.avg_return > 0 ? 'text-green-600' : 'text-red-600'}>
                            {clusterStats.avg_return !== undefined ? `${(clusterStats.avg_return * 100).toFixed(2)}%` : 'N/A'}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600">
                          <span className="font-medium">Volatility:</span>{' '}
                          {clusterStats.avg_volatility !== undefined ? `${(clusterStats.avg_volatility * 100).toFixed(2)}%` : 'N/A'}
                        </div>
                        {clusterStats.tickers_with_data !== undefined && (
                          <div className="text-xs text-gray-500">
                            Data available: {clusterStats.tickers_with_data}/{clusterStats.size} tickers
                          </div>
                        )}
                      </div>
                      <div className="space-y-1">
                        {tickers.slice(0, 5).map(ticker => (
                          <div key={ticker} className="text-sm text-gray-700 flex items-center space-x-1">
                            <span className="w-2 h-2 rounded-full bg-gray-400"></span>
                            <span>{ticker.replace('_yf', '').replace('_bn', '')}</span>
                          </div>
                        ))}
                        {tickers.length > 5 && (
                          <div className="relative group">
                            <div className="text-sm text-blue-600 italic pl-3 cursor-pointer hover:text-blue-800 transition-colors">
                              +{tickers.length - 5} more (hover to see all)
                            </div>
                            <div className="absolute left-0 top-6 w-64 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                              <div className="font-semibold mb-2">All tickers in this cluster:</div>
                              <div className="grid grid-cols-2 gap-1">
                                {tickers.map(ticker => (
                                  <div key={ticker} className="truncate">
                                    {ticker.replace('_yf', '').replace('_bn', '')}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
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
          
          {/* Cluster Characteristics Table */}
          {clusteringData?.report?.cluster_characteristics && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-700">
                Cluster Characteristics Summary
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
                        Data Coverage
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
                    {Object.entries(clusteringData.report.cluster_characteristics).map(([cluster, chars]) => (
                      <tr key={cluster}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          Cluster {parseInt(cluster) + 1}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {chars.size}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {chars.tickers_with_data !== undefined ? 
                            `${chars.tickers_with_data}/${chars.size}` : 
                            'N/A'
                          }
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                          chars.avg_return > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {chars.avg_return !== undefined ? 
                            `${chars.avg_return > 0 ? '+' : ''}${(chars.avg_return * 100).toFixed(2)}%` : 
                            'N/A'
                          }
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {chars.avg_volatility !== undefined ? 
                            `${(chars.avg_volatility * 100).toFixed(2)}%` : 
                            'N/A'
                          }
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
