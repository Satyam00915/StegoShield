import { useEffect, useState } from "react";
import { Typewriter } from 'react-simple-typewriter';
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import toast, { Toaster } from "react-hot-toast";
import Header from "../components/Header";
import { useAuth } from "../context/AuthContext";
import { FileText, ShieldCheck, ShieldX, X, Download, Trash2, Filter, Calendar, BarChart2, Activity } from "lucide-react";
import { Line, Pie, Bar, Doughnut } from "react-chartjs-2";
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
  const [selectedDate, setSelectedDate] = useState("");
  const [isExporting, setIsExporting] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [fileToDelete, setFileToDelete] = useState(null);
  const fullText = `Welcome back, ${user?.name || "User"}!`;
  const [typedText, setTypedText] = useState("");
  const [activeChart, setActiveChart] = useState("pie"); // 'pie' or 'doughnut'

  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));


    if (!storedUser) {
      toast.error("You need to be logged in to access this page.");
      return navigate("/login");
    }

    console.log(user);

    // ✅ FETCH HISTORY FROM BACKEND
    fetch(`http://localhost:5000/api/history?user_id=${storedUser.id}`, {
      credentials: "include",
    })
      .then(res => res.json())
      .then(data => {
        localStorage.setItem("uploadHistory", JSON.stringify(data));
        const sortedHistory = data.sort((a, b) => new Date(b.date) - new Date(a.date));
        setHistory(sortedHistory);
        setFilteredHistory(sortedHistory);
      })
      .catch(err => {
        console.error("Failed to fetch history:", err);
        toast.error("Could not load history.");
      });

    setUser(storedUser);

  }, [isLoggedIn]);

  useEffect(() => {
    let filtered = [...history];
  
    // Filter by result (Safe/Malicious)
    if (activeFilter !== "All") {
      filtered = filtered.filter((item) => item.result === activeFilter);
    }
  
    // Filter by search query
    if (searchQuery.trim() !== "") {
      filtered = filtered.filter((item) =>
        item.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
  
    // Filter by selected date (exact match)
    if (selectedDate) {
      filtered = filtered.filter((item) => {
        const itemDate = new Date(item.date).toDateString();
        const selected = new Date(selectedDate).toDateString();
        return itemDate === selected;
      });
    }
  
    setFilteredHistory(filtered);
  }, [searchQuery, activeFilter, history, selectedDate]);
  
  const summary = filteredHistory.reduce(
    (acc, curr) => {
      acc[curr.result] = (acc[curr.result] || 0) + 1;
      return acc;
    },
    {}
  );

  const exportHistory = () => {
    setIsExporting(true);
    try {
      const dataStr = JSON.stringify(filteredHistory, null, 2);
      const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;

      const exportFileName = `stegoshield_history_${new Date().toISOString().slice(0, 10)}.json`;

      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileName);
      linkElement.click();

      toast.success("Export completed successfully");
    } catch (error) {
      toast.error("Export failed");
    } finally {
      setIsExporting(false);
    }
  };

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setTypedText(fullText.slice(0, index + 1));
      index++;
      if (index === fullText.length) {
        clearInterval(interval);
      }
    }, 80); // typing speed (ms per letter)

    return () => clearInterval(interval); // cleanup if component unmounts
  }, [fullText]);


  const deleteFile = (index) => {
    const updatedHistory = [...history];
    updatedHistory.splice(index, 1);
    localStorage.setItem("uploadHistory", JSON.stringify(updatedHistory));
    setHistory(updatedHistory);
    setIsDeleteModalOpen(false);
    toast.success("File deleted successfully");
  };

  const clearFilters = () => {
    setSearchQuery("");
    setActiveFilter("All");
    setDateRange({ start: null, end: null });
  };

  // Chart data configurations
  const pieChartData = {
    labels: Object.keys(summary),
    datasets: [
      {
        data: Object.values(summary),
        backgroundColor: ["#0e4f63", "#3891ab", "#5bc0de"],
        borderWidth: 2,
      },
    ],
  };

  const barChartData = {
    labels: filteredHistory.slice(0, 5).map((item) =>
      item.name.length > 10 ? `${item.name.substring(0, 10)}...` : item.name
    ),
    datasets: [
      {
        label: "Confidence (%)",
        data: filteredHistory.slice(0, 5).map((item) => (item.confidence * 100).toFixed(2)),
        backgroundColor: ["#0e4f63", "#1a6f7a", "#258f8f", "#30afa4", "#3bcfb9"],
        borderRadius: 6,
      },
    ],
  };

  const lineChartData = {
    labels: Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      return date.toLocaleDateString();
    }),
    datasets: [
      {
        label: "Daily Uploads",
        data: Array.from({ length: 7 }, (_, i) => {
          const date = new Date();
          date.setDate(date.getDate() - (6 - i));
          return filteredHistory.filter(item =>
            new Date(item.date).toLocaleDateString() === date.toLocaleDateString()
          ).length;
        }),
        borderColor: "#219b97",
        backgroundColor: "rgba(33,155,151,0.2)",
        tension: 0.3,
        fill: true,
      },
    ],
  };

  return (
    <div className="min-h-screen bg-blue-50 dark:bg-gray-900 px-4 pb-10">
      <Toaster position="top-right" />
      <Header />

      <div className="max-w-7xl mx-auto space-y-6 mt-4">
        {/* Welcome Header */}
        <div className="flex justify-between items-center">
        <motion.h2
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-3xl md:text-4xl font-extrabold bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent dark:from-blue-600 dark:to-blue-500"
        >
          <span className="inline-flex items-center">
            <Typewriter
              words={[`Welcome back, ${user?.name || "User"}!`]}
              typeSpeed={70}
              cursor={false}
              loop={false}
            />
            <span className="ml-1 w-2 h-6 bg-[#8fbcc4] dark:bg-blue-500 animate-blink rounded-sm" />
          </span>
        </motion.h2>

          <div className="flex gap-2">
            <button
              onClick={exportHistory}
              disabled={isExporting || filteredHistory.length === 0}
              className="flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-gray-700 hover:bg-blue-200 dark:hover:bg-gray-600 rounded-lg transition"
            >
              {isExporting ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="animate-spin" size={18} /> Exporting...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Download size={18} /> Export
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              label: "Total Files",
              value: filteredHistory.length,
              icon: <FileText size={20} />,
              change: "+12%",
              color: "bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-300",
            },
            {
              label: "Safe Files",
              value: summary["Safe"] || 0,
              icon: <ShieldCheck size={20} />,
              change: "+8%",
              color: "bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-300",
            },
            {
              label: "Malicious Files",
              value: summary["Malicious"] || 0,
              icon: <ShieldX size={20} />,
              change: "-4%",
              color: "bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-300",
            },
            {
              label: "Avg Confidence",
              value: filteredHistory.length > 0
                ? ((filteredHistory.reduce((sum, item) => sum + item.confidence, 0) / filteredHistory.length) * 100).toFixed(2) + "%"
                : "0%",
              icon: <Activity size={20} />,
              change: "+2%",
              color: "bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-300",
            },
          ].map((card, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`p-4 rounded-xl shadow-sm border ${card.color} dark:border-gray-700`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{card.label}</p>
                  <p className="text-2xl font-bold mt-1">{card.value}</p>
                </div>
                <div className="p-2 rounded-lg bg-white/30 dark:bg-black/20">
                  {card.icon}
                </div>
              </div>
              <p className="text-xs mt-2 opacity-80">{card.change} from last week</p>
            </motion.div>
          ))}
        </div>

        {/* Filters Section */}
        <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow border border-gray-200 dark:border-gray-700">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div className="flex-1">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search files..."
                  className="w-full pl-10 pr-4 py-2 rounded-lg border bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <span className="absolute left-3 top-2.5 text-gray-400">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </span>
              </div>
            </div>

            <div className="flex gap-2">
              <div className="relative">
                <select
                  className="appearance-none pl-3 pr-8 py-2 rounded-lg border bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                  value={activeFilter}
                  onChange={(e) => setActiveFilter(e.target.value)}
                >
                  <option value="All">All Results</option>
                  <option value="Safe">Safe Only</option>
                  <option value="Malicious">Malicious Only</option>
                </select>
                <div className="absolute right-2 top-2.5 pointer-events-none">
                  <Filter size={18} className="text-gray-400" />
                </div>
              </div>

              <div className="relative">
                <input
                  type="date"
                  className="pl-3 pr-8 py-2 rounded-lg border bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                  value={selectedDate || ""}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  max={new Date().toISOString().split('T')[0]}
                />
                <div className="absolute right-2 top-2.5 pointer-events-none">
                  <Calendar size={18} className="text-gray-400" />
                </div>
              </div>


              <button
                onClick={clearFilters}
                className="px-3 py-2 rounded-lg border bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Summary Chart */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow border border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-semibold text-gray-800 dark:text-white">Result Distribution</h3>
              <div className="flex gap-1">
                <button
                  onClick={() => setActiveChart("pie")}
                  className={`p-1 rounded ${activeChart === "pie" ? "bg-blue-100 text-blue-600 dark:bg-blue-900/30" : "text-gray-400"}`}
                >
                  <PieChart size={18} />
                </button>
                <button
                  onClick={() => setActiveChart("doughnut")}
                  className={`p-1 rounded ${activeChart === "doughnut" ? "bg-blue-100 text-blue-600 dark:bg-blue-800/30" : "text-gray-400"}`}
                >
                  <DonutChart size={18} />
                </button>
              </div>
            </div>
            <div className="h-64">
              {activeChart === "pie" ? (
                <Pie
                  data={pieChartData}
                  options={{
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { position: 'right' },
                      tooltip: {
                        callbacks: {
                          label: (context) => {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const value = context.raw;
                            const percentage = Math.round((value / total) * 100);
                            return `${context.label}: ${value} (${percentage}%)`;
                          }
                        }
                      }
                    }
                  }}
                />
              ) : (
                <Doughnut
                  data={pieChartData}
                  options={{
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                      legend: { position: 'right' }
                    }
                  }}
                />
              )}
            </div>
          </div>

          {/* Confidence Chart */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-800 dark:text-white mb-4">Top Files by Confidence</h3>
            <div className="h-64">
              <Bar
                data={barChartData}
                options={{
                  maintainAspectRatio: false,
                  responsive: true,
                  plugins: {
                    legend: { display: false },
                    tooltip: {
                      callbacks: {
                        label: (context) => `${context.parsed.y}% confidence`
                      }
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      ticks: {
                        callback: (value) => `${value}%`
                      }
                    }
                  }
                }}
              />
            </div>
          </div>

          {/* Activity Chart */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-800 dark:text-white mb-4">Weekly Activity</h3>
            <div className="h-64">
              <Line
                data={lineChartData}
                options={{
                  maintainAspectRatio: false,
                  responsive: true,
                  plugins: {
                    legend: { display: false },
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        precision: 0
                      }
                    }
                  }
                }}
              />
            </div>
          </div>
        </div>

        {/* File History Section */}
        <div className="bg-white dark:bg-gray-900 rounded-xl shadow border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-800 dark:text-white">Recent File Analysis</h3>
          </div>

          {filteredHistory.length === 0 ? (
            <div className="p-8 text-center">
              <div className="mx-auto w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
                <FileText className="text-gray-400" size={24} />
              </div>
              <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-1">No files found</h4>
              <p className="text-gray-500 dark:text-gray-400">Try adjusting your filters or upload new files</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 dark:bg-gray-800 text-left text-xs text-gray-500 dark:text-gray-400 uppercase">
                  <tr>
                    <th className="px-6 py-3">File Name</th>
                    <th className="px-6 py-3">Date</th>
                    <th className="px-6 py-3">Result</th>
                    <th className="px-6 py-3">Confidence</th>
                    <th className="px-6 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {filteredHistory.map((item, i) => (
                    <motion.tr
                      key={i}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer"
                      onClick={() => {
                        setSelectedFile(item);
                        setIsModalOpen(true);
                      }}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                            <FileText className="text-blue-500" size={16} />
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900 dark:text-white">
                              {item.name.length > 30 ? `${item.name.substring(0, 30)}...` : item.name}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {(item.size / 1024).toFixed(2)} KB
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 dark:text-white">
                          {new Date(item.date).toLocaleDateString()}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {new Date(item.date).toLocaleTimeString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${item.result === "Safe"
                          ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                          : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                          }`}>
                          {item.result}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${item.result === "Safe" ? "bg-green-500" : "bg-red-500"
                                }`}
                              style={{ width: `${item.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                            {(item.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setFileToDelete(i);
                            setIsDeleteModalOpen(true);
                          }}
                          className="text-red-500 hover:text-red-700 dark:hover:text-red-400 p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20"
                        >
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* File Details Modal */}
      {isModalOpen && selectedFile && (
        <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm flex justify-center items-center z-50 p-4">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white dark:bg-gray-900 rounded-xl shadow-lg w-full max-w-md"
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-800 dark:text-white">File Analysis Details</h3>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                    <FileText className="text-blue-500" size={24} />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 dark:text-white">{selectedFile.name}</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Uploaded on {new Date(selectedFile.date).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                    <p className="text-sm text-gray-500 dark:text-gray-400">Result</p>
                    <p className={`mt-1 font-medium ${selectedFile.result === "Safe"
                      ? "text-green-600 dark:text-green-400"
                      : "text-red-600 dark:text-red-400"
                      }`}>
                      {selectedFile.result}
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                    <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
                    <p className="mt-1 font-medium text-indigo-600 dark:text-indigo-400">
                      {(selectedFile.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Details</p>
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-300">File Type:</span>
                      <span className="font-medium">{selectedFile.type || "Unknown"}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-300">File Size:</span>
                      <span className="font-medium">{(selectedFile.size / 1024).toFixed(2)} KB</span>
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Analysis Metadata</p>
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-300">Analysis Date:</span>
                      <span className="font-medium">{new Date(selectedFile.date).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-300">Analysis Version:</span>
                      <span className="font-medium">v2.1.4</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-3">
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                >
                  Close
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && fileToDelete !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm flex justify-center items-center z-50 p-4">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white dark:bg-gray-900 rounded-xl shadow-lg w-full max-w-md"
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-800 dark:text-white">Confirm Deletion</h3>
                <button
                  onClick={() => setIsDeleteModalOpen(false)}
                  className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="space-y-4">
                <p className="text-gray-600 dark:text-gray-300">
                  Are you sure you want to delete the analysis for <span className="font-medium">"{history[fileToDelete]?.name}"</span>? This action cannot be undone.
                </p>

                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-red-100 dark:bg-red-900/30">
                      <Trash2 className="text-red-500 dark:text-red-400" size={18} />
                    </div>
                    <p className="text-sm text-red-600 dark:text-red-400">
                      This will permanently remove this record from your history.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-3">
                <button
                  onClick={() => setIsDeleteModalOpen(false)}
                  className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={() => deleteFile(fileToDelete)}
                  className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition"
                >
                  Delete Permanently
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

// Helper components for icons
const PieChart = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2V12V22Z" fill="currentColor" />
    <path d="M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22V12V2Z" fill="currentColor" opacity="0.5" />
  </svg>
);

const DonutChart = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" />
    <path d="M12 16C14.2091 16 16 14.2091 16 12C16 9.79086 14.2091 8 12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16Z" stroke="currentColor" strokeWidth="2" />
    <path d="M12 2V12V22" stroke="currentColor" strokeWidth="2" />
  </svg>
);

export default Dashboard;