import React from "react";
import {
    BrowserRouter as Router,
    Routes,
    Route,
    NavLink,
} from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import EDAView from "./pages/EDAView";

export default function App() {
    return (
        <Router>
            <div className="flex h-screen">
                <nav className="w-64 bg-gray-800 text-white p-4">
                    <ul className="space-y-4">
                        <li>
                            <NavLink
                                to="/dashboard"
                                className={({ isActive }) =>
                                    isActive ? "text-blue-400" : ""
                                }
                            >
                                Dashboard
                            </NavLink>
                        </li>
                        <li>
                            <NavLink
                                to="/eda"
                                className={({ isActive }) =>
                                    isActive ? "text-blue-400" : ""
                                }
                            >
                                EDA Analysis
                            </NavLink>
                        </li>
                    </ul>
                </nav>
                <main className="flex-1 bg-gray-100 p-6 overflow-auto">
                    <Routes>
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/eda" element={<EDAView />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}
