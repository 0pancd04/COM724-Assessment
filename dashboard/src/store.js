import { create } from "zustand";

export const useStore = create((set) => ({
    ticker: "BTC",
    interval: "1m",
    setTicker: (ticker) => set({ ticker }),
    setInterval: (interval) => set({ interval }),
    realTimeData: [],
    setRealTimeData: (data) => set({ realTimeData: data }),
}));
