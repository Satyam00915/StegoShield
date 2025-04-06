import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import { useNavigate } from "react-router-dom";
import toast, { Toaster } from "react-hot-toast";
import Header from "../components/Header";
import { useAuth } from "../context/AuthContext";

const COLORS = ["#00C49F", "#FF8042"];

const Dashboard = () => {
    const [userName, setUserName] = useState("CyberNova");
    const [history, setHistory] = useState([]);
    const [filteredHistory, setFilteredHistory] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const { isLoggedIn } = useAuth();

    const navigate = useNavigate();

    useEffect(() => {
        if (!localStorage.getItem("user")) {
            toast.error("You need to be logged in to access this page.");
            navigate("/login");
        }
    }, [isLoggedIn])

    useEffect(() => {
        fetch("http://localhost:5000/api/user")
            .then((res) => res.json())
            .then((data) => {
                setUserName(data.name || "CyberNova");
                toast.success(`Welcome back, ${data.name || "CyberNova"}!`);
            })
            .catch(() => {
                setUserName("CyberNova");
                toast.error("Failed to fetch user data");
            });

        const stored = JSON.parse(localStorage.getItem("uploadHistory")) || [];
        const sorted = [...stored].sort((a, b) => new Date(b.date) - new Date(a.date));
        setHistory(sorted);
        setFilteredHistory(sorted);
    }, []);

    useEffect(() => {
        const filtered = history.filter((item) =>
            item.name.toLowerCase().includes(searchQuery.toLowerCase())
        );
        setFilteredHistory(filtered);
    }, [searchQuery, history]);

    const summary = filteredHistory.reduce(
        (acc, curr) => {
            acc[curr.result] += 1;
            return acc;
        },
        { Malicious: 0, Safe: 0 }
    );

    const chartData = [
        { name: "Malicious", value: summary.Malicious },
        { name: "Safe", value: summary.Safe },
    ];

    const barData = filteredHistory.slice(0, 5).map((item) => ({
        name: item.name.length > 10 ? item.name.slice(0, 10) + "..." : item.name,
        confidence: (item.confidence * 100).toFixed(2),
    }));

    return (
        <div className="min-h-screen bg-gray-100 dark:bg-gray-900 px-4 py-6">
            <Toaster position="top-right" />
            <Header />

            <div className="max-w-6xl mx-auto space-y-6 mt-4">
                <motion.h2
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-xl font-semibold text-gray-800 dark:text-white"
                >
                    Welcome back, {userName} 👋
                </motion.h2>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow text-center"
                    >
                        <h4 className="text-gray-500 dark:text-gray-400">Total Files</h4>
                        <p className="text-2xl font-bold text-gray-800 dark:text-white">
                            {filteredHistory.length}
                        </p>
                    </motion.div>
                    <div className="bg-green-100 dark:bg-green-900 p-4 rounded-xl shadow text-center">
                        <h4 className="text-gray-500 dark:text-gray-200">Safe</h4>
                        <p className="text-2xl font-bold text-green-700 dark:text-green-300">
                            {summary.Safe}
                        </p>
                    </div>
                    <div className="bg-red-100 dark:bg-red-900 p-4 rounded-xl shadow text-center">
                        <h4 className="text-gray-500 dark:text-gray-200">Malicious</h4>
                        <p className="text-2xl font-bold text-red-700 dark:text-red-300">
                            {summary.Malicious}
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow">
                        <h4 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">
                            Prediction Summary
                        </h4>
                        <ResponsiveContainer width="100%" height={250}>
                            <PieChart>
                                <Pie
                                    data={chartData}
                                    dataKey="value"
                                    nameKey="name"
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={80}
                                >
                                    {chartData.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={COLORS[index % COLORS.length]}
                                        />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow">
                        <h4 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">
                            Top 5 Recent Confidences
                        </h4>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={barData}>
                                <XAxis dataKey="name" stroke="#ccc" />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="confidence" fill="#8884d8" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow mt-6">
                    <h4 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
                        Recent Upload History
                    </h4>

                    <input
                        type="text"
                        placeholder="Search by filename..."
                        className="mb-4 px-3 py-2 rounded-lg border w-full dark:bg-gray-700 dark:text-white"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />

                    {filteredHistory.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400">No uploads found.</p>
                    ) : (
                        <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                            {filteredHistory.slice(0, 10).map((item, i) => (
                                <li
                                    key={i}
                                    className="border-b border-gray-200 dark:border-gray-700 pb-2"
                                >
                                    <strong>{item.name}</strong> - {item.result} (
                                    {(item.confidence * 100).toFixed(2)}%) on{" "}
                                    {new Date(item.date).toLocaleString()}
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
