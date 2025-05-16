import React, { useEffect, useState, useRef } from "react";
import { useStore } from "../store";
import instance from "../api";
import ReactApexChart from "react-apexcharts";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

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

    const [clusters, setClusters] = useState({});
    const [forecasts, setForecasts] = useState(null);
    const [signals, setSignals] = useState(null);

    const [isTraining, setIsTraining] = useState(false);
    const [loadingForecasts, setLoadingForecasts] = useState(false);
    const [loadingSignals, setLoadingSignals] = useState(false);

    const [rsiData, setRsiData] = useState([]);
    const [macdData, setMacdData] = useState({ macd: [], signal: [] });

    // helper to compute RSI
    function computeRSI(bars, windowLen = 14) {
        const closes = bars.map((b) => b.y[3]);
        const deltas = closes.slice(1).map((v, i) => v - closes[i]);
        const rsi = [];
        for (let i = windowLen; i < closes.length; i++) {
            const slice = deltas.slice(i - windowLen, i);
            const gains = slice.filter((d) => d > 0).reduce((a, b) => a + b, 0);
            const losses = slice
                .filter((d) => d < 0)
                .reduce((a, b) => a - b, 0);
            const avgGain = gains / windowLen;
            const avgLoss = losses / windowLen;
            const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
            const score = 100 - 100 / (1 + rs);
            rsi.push({ x: bars[i].x, y: score });
        }
        return rsi;
    }

    // helper to compute EMA
    function ema(values, span) {
        const k = 2 / (span + 1);
        const emaArr = [];
        values.forEach((v, i) => {
            if (i === 0) {
                emaArr.push(v);
            } else {
                emaArr.push(v * k + emaArr[i - 1] * (1 - k));
            }
        });
        return emaArr;
    }

    // helper to compute MACD
    function computeMACD(bars, fastLen = 12, slowLen = 26, signalLen = 9) {
        const closes = bars.map((b) => b.y[3]);
        const emaFast = ema(closes, fastLen);
        const emaSlow = ema(closes, slowLen);
        const macdLine = emaFast.map((v, i) => v - emaSlow[i]);
        const signalLine = ema(macdLine, signalLen);
        const macd = bars.slice(slowLen - 1).map((b, i) => ({
            x: b.x,
            y: macdLine[i + slowLen - 1],
        }));
        const signal = bars.slice(slowLen + signalLen - 2).map((b, i) => ({
            x: b.x,
            y: signalLine[i + slowLen + signalLen - 2],
        }));
        return { macd, signal };
    }

    // load clusters once
    useEffect(() => {
        instance
            .get("/available_tickers_based_on_clusters_grouping")
            .then((r) => setClusters(r.data.available_tickers))
            .catch(console.error);
    }, []);

    // decide train vs fetch
    useEffect(() => {
        if (!ticker) return;
        setIsTraining(true);
        instance
            .get(`/model_status/${ticker}`)
            .then((r) => {
                const allDone = Object.values(r.data).every((v) => v);
                if (allDone) {
                    setIsTraining(false);
                    fetchData();
                } else {
                    toast.info(`Training models for ${ticker}…`, {
                        autoClose: 3000,
                    });
                    instance
                        .post(`/train/${ticker}`, null, {
                            params: { feature: "Close", test_size: 0.2 },
                        })
                        .then(() => {
                            setIsTraining(false);
                            fetchData();
                        })
                        .catch((e) => {
                            setIsTraining(false);
                            toast.error("Training failed");
                            console.error(e);
                        });
                }
            })
            .catch((e) => {
                setIsTraining(false);
                console.error(e);
            });
    }, [ticker]);

    // fetch forecasts & signals
    const fetchData = () => {
        setLoadingForecasts(true);
        setLoadingSignals(true);

        instance
            .get(`/forecast/${ticker}`, {
                params: { model_type: "arima", periods: 7 },
            })
            .then((r) => setForecasts(r.data.forecast))
            .finally(() => setLoadingForecasts(false));

        instance
            .get(`/signals/${ticker}`, {
                params: { model_type: "arima", periods: 7 },
            })
            .then((r) => setSignals(r.data.signals))
            .finally(() => setLoadingSignals(false));
    };

    // historical OHLC + MAs
    useEffect(() => {
        if (!ticker || !interval) return;
        const sym = `${ticker.toUpperCase()}USDT`,
            url = `https://api.binance.com/api/v3/klines?symbol=${sym}&interval=${interval}&limit=200`;
        fetch(url)
            .then((r) => r.json())
            .then((data) => {
                const ohlc = data.map((k) => ({
                    x: k[0],
                    y: [+k[1], +k[2], +k[3], +k[4]],
                }));
                // 50 & 200 MA
                const closes = ohlc.map((b) => b.y[3]);
                const ma = (n) =>
                    closes.map((_, i, arr) =>
                        i >= n - 1
                            ? arr
                                  .slice(i - n + 1, i + 1)
                                  .reduce((a, b) => a + b, 0) / n
                            : null
                    );
                const ma50 = ma(50),
                    ma200 = ma(200);

                setRealTimeData(
                    ohlc.map((bar, i) => ({
                        ...bar,
                        ma50: ma50[i],
                        ma200: ma200[i],
                    }))
                );
            })
            .catch(console.error);
    }, [ticker, interval]);

    // compute RSI & MACD any time the series updates
    useEffect(() => {
        if (Array.isArray(realTimeData) && realTimeData.length > 0) {
            setRsiData(computeRSI(realTimeData, 14));
            setMacdData(computeMACD(realTimeData, 12, 26, 9));
        }
    }, [realTimeData]);

    // websocket live updates
    useEffect(() => {
        if (!ticker || !interval) return;
        if (ws.current) ws.current.close();
        const s = ticker.toLowerCase() + "usdt",
            wsUrl = `wss://stream.binance.com:9443/ws/${s}@kline_${interval}`;
        const sock = new WebSocket(wsUrl);
        sock.onmessage = (e) => {
            const { k } = JSON.parse(e.data);
            if (k.x) {
                const newBar = {
                    x: k.t,
                    y: [+k.o, +k.h, +k.l, +k.c],
                };
                setRealTimeData((prev) => {
                    const last = prev[prev.length - 1];
                    if (last && last.x === newBar.x) {
                        return [...prev.slice(0, -1), newBar];
                    }
                    return [...prev, newBar].slice(-200);
                });
            }
        };
        sock.onerror = console.error;
        ws.current = sock;
        return () => sock.close();
    }, [ticker, interval]);

    return (
        <>
            <div className="space-y-6">
                {/* Selection */}
                <div className="flex space-x-4">
                    <select
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value)}
                        className="p-2 border rounded"
                    >
                        <option value="" disabled>
                            Select ticker
                        </option>
                        {Object.entries(clusters).map(([grp, arr]) => (
                            <optgroup key={grp} label={`Cluster ${grp}`}>
                                {arr.map((t) => (
                                    <option key={t} value={t}>
                                        {t}
                                    </option>
                                ))}
                            </optgroup>
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

                {/* Real-Time Chart + MAs */}
                <h2 className="text-lg font-bold">Real-Time Chart</h2>
                {Array.isArray(realTimeData) && realTimeData.length > 0 && (
                    <ReactApexChart
                        height={350}
                        width={800}
                        options={{
                            chart: { id: "realtime", type: "candlestick" },
                            xaxis: { type: "datetime" },
                            stroke: { width: [1, 2, 2] },
                        }}
                        series={[
                            {
                                name: "OHLC",
                                type: "candlestick",
                                data: realTimeData.map((b) => ({
                                    x: b.x,
                                    y: b.y,
                                })),
                            },
                            {
                                name: "MA50",
                                type: "line",
                                data: realTimeData
                                    .map((b) => [b.x, b.ma50])
                                    .filter((p) => p[1] != null),
                            },
                            {
                                name: "MA200",
                                type: "line",
                                data: realTimeData
                                    .map((b) => [b.x, b.ma200])
                                    .filter((p) => p[1] != null),
                            },
                        ]}
                    />
                )}

                {/* RSI & MACD Mini-Charts */}
                <h2 className="text-lg font-bold">Indicators</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h3 className="font-semibold">RSI (14)</h3>
                        <ReactApexChart
                            type="line"
                            height={200}
                            series={[
                                {
                                    name: "RSI",
                                    data: rsiData.map((p) => [p.x, p.y]),
                                },
                            ]}
                            options={{
                                chart: { id: "rsi", toolbar: { show: false } },
                                xaxis: { type: "datetime" },
                                yaxis: { min: 0, max: 100 },
                            }}
                        />
                    </div>
                    <div>
                        <h3 className="font-semibold">MACD</h3>
                        <ReactApexChart
                            type="line"
                            height={200}
                            series={[
                                {
                                    name: "MACD",
                                    data: macdData.macd.map((p) => [p.x, p.y]),
                                },
                                {
                                    name: "Signal",
                                    data: macdData.signal.map((p) => [
                                        p.x,
                                        p.y,
                                    ]),
                                },
                            ]}
                            options={{
                                chart: { id: "macd", toolbar: { show: false } },
                                xaxis: { type: "datetime" },
                            }}
                        />
                    </div>
                </div>

                {/* Forecasts */}
                <h2 className="text-lg font-bold">7-Day Forecast</h2>
                {loadingForecasts ? (
                    <div>Loading forecasts…</div>
                ) : (
                    forecasts && (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                            {Object.entries(forecasts).map(
                                ([date, { forecast, lower, upper }]) => (
                                    <div
                                        key={date}
                                        className="p-4 bg-white rounded shadow"
                                    >
                                        <h4 className="font-semibold">
                                            {new Date(
                                                date
                                            ).toLocaleDateString()}
                                        </h4>
                                        <p>
                                            <strong>Forecast:</strong>{" "}
                                            {forecast.toFixed(2)}
                                        </p>
                                        <p>
                                            <strong>Lower CI:</strong>{" "}
                                            {lower.toFixed(2)}
                                        </p>
                                        <p>
                                            <strong>Upper CI:</strong>{" "}
                                            {upper.toFixed(2)}
                                        </p>
                                    </div>
                                )
                            )}
                        </div>
                    )
                )}

                {/* Signals */}
                <h2 className="text-lg font-bold">Buy/Sell Signals</h2>
                {loadingSignals ? (
                    <div>Loading signals…</div>
                ) : (
                    signals && (
                        <table className="min-w-full bg-white">
                            <thead className="bg-gray-100">
                                <tr>
                                    <th className="px-4 py-2 text-left">
                                        Date
                                    </th>
                                    <th className="px-4 py-2 text-left">
                                        Signal
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(signals).map(([d, s]) => (
                                    <tr key={d}>
                                        <td className="px-4 py-2">
                                            {new Date(d).toLocaleString()}
                                        </td>
                                        <td className="px-4 py-2">{s}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )
                )}
            </div>
            <ToastContainer />
        </>
    );
}
