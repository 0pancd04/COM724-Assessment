import React, { useEffect, useState, useRef } from "react";
import { useStore } from "../store";
import instance from "../api";
import ReactApexChart from "react-apexcharts";

export default function Dashboard() {
    const {
        ticker,
        interval,
        setTicker,
        setInterval,
        realTimeData,
        setRealTimeData,
    } = useStore();
    const ws = useRef(null);
    const [forecasts, setForecasts] = useState(null);
    const [signals, setSignals] = useState(null);
    const [indicators, setIndicators] = useState(null);

    // WebSocket for real-time kline
    useEffect(() => {
        if (!ticker || !interval) return; // don’t open until both exist
        if (ws.current) ws.current.close();

        const symbol = `${ticker.toLowerCase()}usdt`;
        const stream = `wss://stream.binance.com:9443/ws/${symbol}@kline_${interval}`;
        const socket = new WebSocket(stream);

        socket.onmessage = (evt) => {
            const msg = JSON.parse(evt.data);
            if (msg.k && msg.k.x) {
                const { t, o, h, l, c } = msg.k;
                setRealTimeData((prev) =>
                    [...prev, { x: new Date(t), y: [o, h, l, c] }].slice(-200)
                );
            }
        };
        socket.onerror = (err) => console.error("WS error", err);
        ws.current = socket;

        return () => socket.close();
    }, [ticker, interval]);

    // Fetch forecasts & signals
    useEffect(() => {
        instance
            .get(`/forecast/${ticker}`, {
                params: { model_type: "arima", periods: 7 },
            })
            .then((res) => setForecasts(res.data.forecast));
        instance
            .get(`/signals/${ticker}`, {
                params: { model_type: "arima", periods: 7 },
            })
            .then((res) => setSignals(res.data.signals));
        instance
            .get(`/indicators/${ticker}`)
            .then((res) => setIndicators(res.data));
    }, [ticker]);

    // Fetch ticker list
    const [tickers, setTickers] = useState([]);
    useEffect(() => {
        instance
            .get("/available_tickers_based_on_clusters_grouping")
            .then((res) => setTickers(Object.keys(res.data.available_tickers)));
    }, []);

    return (
        <div className="space-y-6">
            <div className="flex space-x-4">
                <select
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    className="p-2 border rounded"
                >
                    {tickers.map((t) => (
                        <option key={t} value={t}>
                            {t}
                        </option>
                    ))}
                </select>
                <select
                    value={interval}
                    onChange={(e) => setInterval(e.target.value)}
                    className="p-2 border rounded"
                >
                    {["1m", "5m", "15m", "1h", "4h", "1d"].map((iv) => (
                        <option key={iv} value={iv}>
                            {iv}
                        </option>
                    ))}
                </select>
            </div>
            {realTimeData.length > 0 && (
                <ReactApexChart
                    type="candlestick"
                    series={[{ data: realTimeData }]}
                    options={{
                        chart: { id: "chart" },
                        xaxis: { type: "datetime" },
                    }}
                    height={350}
                />
            )}
            {indicators && (
                <div className="grid grid-cols-2 gap-4">
                    {/* RSI is a simple date→number map */}
                    <div>
                        RSI:{" "}
                        {(() => {
                            const vals = Object.values(indicators.rsi);
                            return vals.length
                                ? vals[vals.length - 1].toFixed(2)
                                : "--";
                        })()}
                    </div>
                    {/* MACD is date→{MACD,Signal,Hist} */}
                    <div>
                        MACD:{" "}
                        {(() => {
                            const entries = Object.entries(indicators.macd);
                            if (!entries.length) return "--";
                            const last = entries[entries.length - 1][1]; // [date, {…}]
                            return last.MACD.toFixed(2);
                        })()}
                    </div>
                </div>
            )}

            {forecasts && (
                <div className="grid grid-cols-3 gap-4">
                    {Object.entries(forecasts).map(
                        ([date, { forecast, lower, upper }]) => (
                            <div
                                key={date}
                                className="p-2 bg-white rounded shadow"
                            >
                                <div className="font-semibold">
                                    {new Date(date).toLocaleDateString()}
                                </div>
                                <div>Forecast: {forecast.toFixed(2)}</div>
                                <div>Lower CI: {lower.toFixed(2)}</div>
                                <div>Upper CI: {upper.toFixed(2)}</div>
                            </div>
                        )
                    )}
                </div>
            )}

            {signals && (
                <table className="min-w-full bg-white">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Signal</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(signals).map(([d, s]) => (
                            <tr key={d}>
                                <td>{new Date(d).toLocaleString()}</td>
                                <td>{s}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
