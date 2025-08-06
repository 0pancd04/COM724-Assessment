import React, { useEffect, useState } from "react";
import axios from "axios";
import { API_ENDPOINTS } from "../config/api";
import EDAVisualization from "../components/EDAVisualization";

export default function EDAView() {
    const [charts, setCharts] = useState({});
    const [loading, setLoading] = useState(false);
    const [selectedTicker, setSelectedTicker] = useState("BTC");
    const [selectedDataSource, setSelectedDataSource] = useState("yfinance");
    
    const availableTickers = ["BTC", "ETH", "ADA", "DOT", "SOL", "MATIC"];
    const availableDataSources = [
        { value: "yfinance", label: "Yahoo Finance", icon: "📈" },
        { value: "binance", label: "Binance", icon: "🟡" }
    ];

    useEffect(() => {
        loadLegacyCharts();
    }, [selectedTicker, selectedDataSource]);

    const loadLegacyCharts = async () => {
        setLoading(true);
        const chartTypes = [
            "temporal_line",
            "histograms", 
            "box_plots",
            "rolling_average",
            "candlestick",
            "rolling_volatility"
        ];

        const chartPromises = chartTypes.map(async (type) => {
            try {
                const response = await axios.get(`/eda_chart/${type}`, { 
                    params: { 
                        ticker: selectedTicker,
                        source: selectedDataSource 
                    } 
                });
                return { type, html: response.data };
            } catch (error) {
                console.warn(`Failed to load ${type} chart:`, error);
                return { type, html: null };
            }
        });

        try {
            const results = await Promise.all(chartPromises);
            const chartsData = {};
            results.forEach(({ type, html }) => {
                if (html) chartsData[type] = html;
            });
            setCharts(chartsData);
        } catch (error) {
            console.error("Failed to load charts:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <h1 className="text-4xl font-bold text-gradient">📊 EDA Analysis</h1>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <label className="text-sm font-medium text-gray-700">Ticker:</label>
                            <select
                                value={selectedTicker}
                                onChange={(e) => setSelectedTicker(e.target.value)}
                                className="px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            >
                                {availableTickers.map(ticker => (
                                    <option key={ticker} value={ticker}>{ticker}</option>
                                ))}
                            </select>
                        </div>
                        <div className="flex items-center gap-2">
                            <label className="text-sm font-medium text-gray-700">Data Source:</label>
                            <select
                                value={selectedDataSource}
                                onChange={(e) => setSelectedDataSource(e.target.value)}
                                className="px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            >
                                {availableDataSources.map(source => (
                                    <option key={source.value} value={source.value}>
                                        {source.icon} {source.label}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <button
                            onClick={loadLegacyCharts}
                            className="px-4 py-2 gradient-blue text-white rounded-lg hover:shadow-lg transition-all duration-300"
                            disabled={loading}
                        >
                            {loading ? "Loading..." : "Refresh"}
                        </button>
                    </div>
                </div>

                {/* Modern EDA Component */}
                <div className="mb-8">
                    <h2 className="text-2xl font-semibold mb-4 text-gradient">Interactive Analysis</h2>
                    <EDAVisualization dataSource={selectedDataSource} />
                </div>

                {/* Legacy Charts Section */}
                {Object.keys(charts).length > 0 && (
                    <div className="space-y-6">
                        <h2 className="text-2xl font-semibold text-gradient">Stored EDA Charts</h2>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {Object.entries(charts).map(([key, html]) => (
                                <div key={key} className="glass-card p-4">
                                    <h3 className="text-lg font-semibold mb-2 capitalize text-gradient">
                                        {key.replace(/_/g, ' ')}
                                    </h3>
                                    <div 
                                        className="chart-container"
                                        dangerouslySetInnerHTML={{ __html: html }} 
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* No Data State */}
                {!loading && Object.keys(charts).length === 0 && (
                    <div className="glass-card p-12 text-center">
                        <div className="text-6xl mb-4">📊</div>
                        <h3 className="text-xl font-semibold text-gray-700 mb-2">No Legacy Charts Available</h3>
                        <p className="text-gray-600 mb-4">
                            Use the interactive analysis above or run the pipeline to generate stored charts.
                        </p>
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center justify-center h-64">
                        <div className="text-center">
                            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                            <p className="text-lg text-gray-600">Loading EDA charts...</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
