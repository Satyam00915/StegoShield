import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import toast, { Toaster } from "react-hot-toast";
import Header from "../components/Header";
import { useAuth } from "../context/AuthContext";
import { FileText, ShieldCheck, ShieldX } from "lucide-react";

import { Line, Pie, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Tooltip,
  Legend,
  Filler
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Tooltip,
  Legend,
  Filler
);

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [history, setHistory] = useState([]);
  const [filteredHistory, setFilteredHistory] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilter, setActiveFilter] = useState("All");

  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    const storedHistory = JSON.parse(localStorage.getItem("uploadHistory")) || [];

    if (!storedUser) {
      toast.error("You need to be logged in to access this page.");
      return navigate("/login");
    }

    setUser(storedUser);

    const sortedHistory = storedHistory.sort((a, b) => new Date(b.date) - new Date(a.date));
    setHistory(sortedHistory);
    setFilteredHistory(sortedHistory);

    toast.success(`Welcome back, ${storedUser.name}!`);
  }, [isLoggedIn]);

  useEffect(() => {
    let filtered = [...history];

    if (activeFilter !== "All") {
      filtered = filtered.filter((item) => item.result === activeFilter);
    }

    if (searchQuery.trim() !== "") {
      filtered = filtered.filter((item) =>
        item.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredHistory(filtered);
  }, [searchQuery, activeFilter, history]);

  const summary = filteredHistory.reduce(
    (acc, curr) => {
      acc[curr.result] = (acc[curr.result] || 0) + 1;
      return acc;
    },
    {}
  );

  const chartData = Object.keys(summary).map((key) => ({
    name: key,
    value: summary[key]
  }));

  const barData = filteredHistory.slice(0, 5).map((item) => ({
    name: item.name.length > 10 ? item.name.slice(0, 10) + "..." : item.name,
    confidence: (item.confidence * 100).toFixed(2)
  }));

  const trendData = filteredHistory.reduce((acc, item) => {
    const date = new Date(item.date).toLocaleDateString();
    acc[date] = (acc[date] || 0) + 1;
    return acc;
  }, {});

  const lineData = Object.keys(trendData).map((date) => ({
    date,
    uploads: trendData[date],
  })).sort((a, b) => new Date(a.date) - new Date(b.date));

  const pieChartData = {
    labels: chartData.map((d) => d.name),
    datasets: [
      {
        label: "Prediction Summary",
        data: chartData.map((d) => d.value),
        backgroundColor: ["#0e4f63", "#3891ab"],
        borderWidth: 3,
      },
    ],
  };

  const barChartData = {
    labels: barData.map((d) => d.name),
    datasets: [
      {
        label: "Confidence (%)",
        data: barData.map((d) => d.confidence),
        backgroundColor: "#0e4f63",
        borderRadius: 4,
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1000,
      easing: "easeOutQuart",
    },
    plugins: {
      legend: { display: false },
    },
    layout: {
      padding: 30,
    },
    scales: {
      x: {
        ticks: {
          font: { size: 10 },
        },
      },
      y: {
        beginAtZero: true,
        ticks: {
          font: { size: 12 },
        },
      },
    },
  };

  const lineChartData = {
    labels: lineData.map((d) => d.date),
    datasets: [
      {
        label: "Uploads Over Time",
        data: lineData.map((d) => d.uploads),
        fill: true,
        borderColor: "#219b97",
        backgroundColor: "rgba(33,155,151,0.2)",
        tension: 0.4,
      },
    ],
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1500,
      easing: "easeInOutCubic",
    },
    plugins: {
      legend: { display: false },
    },
    layout: {
      padding: 30,
    },
    scales: {
      x: {
        ticks: {
          font: { size: 10 },
        },
      },
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
          font: { size: 12 },
        },
      },
    },
  };

  return (
    <div className="min-h-screen bg-blue-50 dark:bg-gray-800 px-4 ">
      <Toaster position="top-right" />
      <Header />

      <div className="max-w-6xl mx-auto space-y-6 mt-4">
        <motion.h2
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-5xl font-extrabold bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent dark:from-gray-500 dark:to-gray-400e"
        >
          Welcome back, {user?.name || "User"}!
        </motion.h2>

        {/* 🧮 Stats Section */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-4">
          {[
            {
              label: "Total Files",
              value: filteredHistory.length,
              icon: <FileText className="text-blue-500" size={28} />,
              color: "bg-blue-100 dark:bg-blue-600/20",
              textColor: "text-gray-800 dark:text-white",
            },
            {
              label: "Safe Files",
              value: summary["Safe"] || 0,
              icon: <ShieldCheck className="text-green-500" size={28} />,
              color: "bg-green-100 dark:bg-green-600/20",
              textColor: "text-green-600 dark:text-green-300",
            },
            {
              label: "Malicious Files",
              value: summary["Malicious"] || 0,
              icon: <ShieldX className="text-red-500" size={28} />,
              color: "bg-red-100 dark:bg-red-600/20",
              textColor: "text-red-600 dark:text-red-300",
            },
          ].map((card, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              whileHover={{ scale: 1.03 }}
              className="p-5 rounded-xl bg-blue-100 dark:bg-gray-800 shadow flex items-center gap-4 border-2 border-blue-200"
            >
              <div className={`p-3 rounded-full ${card.color}`}>{card.icon}</div>
              <div>
                <h4 className="text-sm text-gray-600 dark:text-gray-400">{card.label}</h4>
                <p className={`text-2xl font-bold ${card.textColor}`}>{card.value}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* 📁 Filter Tabs */}
        <div className="flex gap-4 mt-6 items-center">
          {["All", "Safe", "Malicious"].map((status) => (
            <button
              key={status}
              className={`px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300 ${
                activeFilter === status
                  ? "bg-blue-400 text-white shadow"
                  : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-blue-300 dark:hover:bg-blue-600"
              }`}
              onClick={() => setActiveFilter(status)}
            >
              {status}
            </button>
          ))}
        </div>

        {/* 📊 Charts */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-100 dark:bg-gray-800 p-5 rounded-xl shadow h-96 flex flex-col items-center border-2 border-blue-200">
            <h4 className="text-xl font-bold mb-4 text-gray-800 dark:text-white text-center">
              Prediction Summary
            </h4>
            <div className="w-full h-full">
              <Pie data={pieChartData} options={{ maintainAspectRatio: false, animation: { duration: 1200, easing: "easeOutBounce" } }} />
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-gray-800 p-4 rounded-xl shadow h-96 border-2 border-blue-200">
            <h4 className="text-xl font-bold mb-4 text-gray-800 dark:text-white text-center">
              Top 5 Recent Confidences
            </h4>
            <Bar data={barChartData} options={barChartOptions} />
          </div>

          <div className="bg-blue-100 dark:bg-gray-800 p-5 rounded-xl shadow h-96 border-2 border-blue-200">
            <h4 className="text-xl font-bold mb-4 text-gray-800 dark:text-white text-center">
              Upload Trend Over Time
            </h4>
            <Line data={lineChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* 🗂️ Upload History */}
        <div className="bg-blue-100 dark:bg-gray-800 p-5 rounded-xl shadow mt-6 border-2 border-blue-200">
          <h4 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
            Recent Upload History
          </h4>

          <input
            type="text"
            placeholder="🔍 Search by filename..."
            className="mb-4 px-4 py-2 rounded-lg border w-full bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-300"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          {filteredHistory.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400">No uploads found.</p>
          ) : (
            <ul className="divide-y divide-gray-300 dark:divide-gray-700 max-h-72 overflow-y-auto pr-1">
              {filteredHistory.map((item, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className={`flex items-start justify-between p-3 rounded-lg transition hover:bg-gray-50 dark:hover:bg-gray-700 ${
                    i % 2 === 0 ? "bg-blue-50 dark:bg-gray-700/40" : "bg-white dark:bg-gray-800"
                  }`}
                >
                  <div className="flex flex-col">
                    <span className="font-semibold text-gray-800 dark:text-white">
                      {item.name}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {new Date(item.date).toLocaleString()}
                    </span>
                    <span className="text-xs mt-1 inline-flex items-center gap-1 text-gray-600 dark:text-gray-300 ">
                      📁 <span className="capitalize">{item.filetype || "unknown"}</span>
                    </span>
                  </div>

                  <div className="text-right space-y-1">
                    <span
                      className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        item.result === "Safe"
                          ? "bg-green-100 text-green-700 dark:bg-green-700/20 dark:text-green-300"
                          : "bg-red-100 text-red-700 dark:bg-red-700/20 dark:text-red-300"
                      }`}
                    >
                      {item.result === "Safe" ? "✅ Safe" : "❌ Malicious"}
                    </span>
                    <p className="text-sm font-semibold text-indigo-500 ">
                      {(item.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </motion.li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
