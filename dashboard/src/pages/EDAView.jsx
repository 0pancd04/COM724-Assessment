import React, { useEffect, useState } from "react";
import instance from "../api";

export default function EDAView() {
    const [charts, setCharts] = useState({});
    const ticker = "BTC"; // or selection like Dashboard

    useEffect(() => {
        [
            "temporal_line",
            "histograms",
            "box_plots",
            "rolling_average",
            "candlestick",
            "rolling_volatility",
        ].forEach((type) => {
            instance
                .get(`/eda_chart/${type}`, { params: { ticker } })
                .then((res) =>
                    setCharts((charts) => ({ ...charts, [type]: res.data }))
                )
                .catch(() => {});
        });
    }, []);

    return (
        <div className="space-y-6">
            {Object.entries(charts).map(([key, html]) => (
                <div key={key} dangerouslySetInnerHTML={{ __html: html }} />
            ))}
        </div>
    );
}
