import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";
import { 
  Loader2, Upload, ShieldCheck, ShieldAlert, File, X, History, 
  Info, Download, BarChart2, Settings, HelpCircle, Trash2, 
  ChevronDown, ChevronUp, Star, Clock, Filter, HardDrive, 
  Cloud, Database, Activity, Shield, AlertTriangle, Zap
} from "lucide-react";
import Header from "../components/Header";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useDropzone } from "react-dropzone";
import Footer from "../components/Footer";

const FileUpload = () => {
  // File state management
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);

  const [imagePreview, setImagePreview] = useState("");
  const [audioPreview, setAudioPreview] = useState("");
  const [videoPreview, setVideoPreview] = useState("");

  const [imageResult, setImageResult] = useState(null);
  const [audioResult, setAudioResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false);
  const [isAnalyzingAudio, setIsAnalyzingAudio] = useState(false);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);

  const [imageProgress, setImageProgress] = useState(0);
  const [audioProgress, setAudioProgress] = useState(0);
  const [videoProgress, setVideoProgress] = useState(0);

  // History and UI states
  const [history, setHistory] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [showTutorial, setShowTutorial] = useState(false);
  const [activeTab, setActiveTab] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [showClearHistoryModal, setShowClearHistoryModal] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [scanSpeed, setScanSpeed] = useState("balanced");
  const [isFavoriteView, setIsFavoriteView] = useState(false);
  const [sortBy, setSortBy] = useState("date");
  const [sortOrder, setSortOrder] = useState("desc");

  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();

  // All existing logic functions remain exactly the same
  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("uploadHistory")) || [];
    setHistory(stored);
  }, []);

  useEffect(() => {
    if (!localStorage.getItem("user")) {
      toast.error("You need to be logged in to access this page.");
      navigate("/login");
    }
  }, [isLoggedIn]);

  const saveToLocalHistory = (name, result, confidence, fileSize) => {
    const newItem = {
      id: Date.now(),
      name,
      result,
      confidence,
      file_size: fileSize, // Add file size
      date: new Date().toISOString(),
      isFavorite: false,
      scanType: name.split('.').pop().toLowerCase() === 'mp3' ? 'audio' : 
               name.split('.').pop().toLowerCase() === 'mp4' ? 'video' : 'image'
    };
    const updated = [newItem, ...history];
    localStorage.setItem("uploadHistory", JSON.stringify(updated));
    setHistory(updated);
  };

  const handleDrop = useCallback(
    (acceptedFiles, type, setFile, setPreview, setResult, setProgress) => {
      const file = acceptedFiles[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onloadend = () => {
        setFile(file);
        setPreview(reader.result);
        setResult(null);
        setProgress(0);
      };
      reader.readAsDataURL(file);
    },
    []
  );

  const handleAnalyze = (file, setResult, setProgress, setAnalyzing) => {
    if (!file) return toast.error("Please upload a file first!");

    setAnalyzing(true);
    setProgress(0);
    setResult(null);
    toast.loading("Uploading & Analyzing...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("scanSpeed", scanSpeed);
    formData.append("file_size", file.size);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "https://stegoshield-3ius.onrender.com/upload");
    xhr.withCredentials = true;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        setProgress(percent);
      }
    };

    xhr.onload = () => {
      toast.dismiss();
      setAnalyzing(false);
      try {
        const response = JSON.parse(xhr.responseText);
        setResult(response);
        toast.success("Analysis Complete ✅");
        saveToLocalHistory(file.name, response.result, response.confidence);
      } catch (err) {
        toast.error("Invalid response from server");
      }
    };

    xhr.onerror = () => {
      toast.dismiss();
      setAnalyzing(false);
      toast.error("Upload failed ❌");
    };

    xhr.send(formData);
  };

  // Enhanced DropZone component with better UI
  const DropZone = ({ type, accept, setFileState, setPreview, setResult, setProgress }) => {
    const onDrop = (acceptedFiles) =>
      handleDrop(acceptedFiles, type, setFileState, setPreview, setResult, setProgress);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop,
      accept,
      multiple: false,
    });

    return (
      <div
        {...getRootProps()}
        className={`cursor-pointer border-2 border-dashed rounded-lg p-6 text-center transition
          ${isDragActive ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20" : "border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800"}`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center space-y-2">
          {isDragActive ? (
            <p className="text-sm text-indigo-600 dark:text-indigo-300">Drop the {type} file here...</p>
          ) : (
            <>
              <Upload size={24} className="text-gray-500 dark:text-gray-400" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Drag & drop or click to upload {type}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500">
                Supported: {accept.split(",").join(", ")}
              </p>
            </>
          )}
        </div>
      </div>
    );
  };

  // Enhanced Skeleton component
  const Skeleton = () => (
    <div className="animate-pulse space-y-3">
      <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-full" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
    </div>
  );

  // Toggle favorite status
  const toggleFavorite = (id) => {
    const updated = history.map(item => 
      item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
    );
    localStorage.setItem("uploadHistory", JSON.stringify(updated));
    setHistory(updated);
  };

  // Filter and sort history
  const filteredHistory = history
    .filter(item => {
      const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.result.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesTab = activeTab === "all" || 
        (activeTab === "malicious" && item.result === "Malicious") || 
        (activeTab === "safe" && item.result === "Safe");
      const matchesFavorite = !isFavoriteView || item.isFavorite;
      return matchesSearch && matchesTab && matchesFavorite;
    })
    .sort((a, b) => {
      if (sortBy === "date") {
        return sortOrder === "asc" 
          ? new Date(a.date) - new Date(b.date) 
          : new Date(b.date) - new Date(a.date);
      } else if (sortBy === "name") {
        return sortOrder === "asc" 
          ? a.name.localeCompare(b.name) 
          : b.name.localeCompare(a.name);
      } else if (sortBy === "confidence") {
        return sortOrder === "asc" 
          ? a.confidence - b.confidence 
          : b.confidence - a.confidence;
      }
      return 0;
    });

  // Clear history function
  const clearAllHistory = async () => {
    try {
      const user = JSON.parse(localStorage.getItem("user"));
      if (!user?.id) {
        toast.error("User not authenticated");
        return;
      }

      const response = await fetch('https://stegoshield-3ius.onrender.com/api/history/all', {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ user_id: user.id }),
      });

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        throw new Error(`Expected JSON but got: ${text.substring(0, 100)}`);
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to clear database history');
      }

      localStorage.removeItem("uploadHistory");
      setHistory([]);
      setSearchTerm("");
      setShowClearHistoryModal(false);
      toast.success("All history permanently deleted");
    } catch (error) {
      console.error("Clear history error:", error);
      toast.error(error.message || "Failed to clear history");
    }
  };

  // Enhanced renderSection function with better UI
  const renderSection = (
    label,
    file,
    preview,
    result,
    setFile,
    setPreview,
    setResult,
    setProgress,
    setAnalyzing,
    progress,
    analyzing,
    accept,
    type
  ) => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full md:w-1/3 space-y-4 bg-white dark:bg-gray-800 p-5 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-700 dark:text-white flex items-center gap-2">
            {label}
            <button
              onClick={() => toast(<div className="p-2">
                <h4 className="font-bold mb-1">About {label} Analysis</h4>
                <p className="text-sm">Detects hidden data, watermarks, and anomalies in {type} files.</p>
              </div>, { duration: 5000 })}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <Info size={16} />
            </button>
          </h3>
          {file && (
            <button
              onClick={() => {
                setFile(null);
                setPreview("");
                setResult(null);
              }}
              className="text-gray-400 hover:text-red-500"
            >
              <X size={18} />
            </button>
          )}
        </div>

        <DropZone
          type={type}
          accept={accept}
          setFileState={setFile}
          setPreview={setPreview}
          setResult={setResult}
          setProgress={setProgress}
        />

        {file && (
          <div className="text-xs bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-gray-700 dark:text-gray-300 space-y-1">
            <p className="flex justify-between"><span className="font-medium">Name:</span> <span className="truncate max-w-[180px]">{file.name}</span></p>
            <p className="flex justify-between"><span className="font-medium">Size:</span> {(file.size / 1024).toFixed(2)} KB</p>
            <p className="flex justify-between"><span className="font-medium">Type:</span> {file.type}</p>
          </div>
        )}

        {preview && (
          <motion.div
            animate={
              analyzing
                ? {
                  scale: [1, 1.02, 1],
                  boxShadow: [
                    "0 0 0px rgba(99,102,241,0)",
                    "0 0 15px rgba(99,102,241,0.6)",
                    "0 0 0px rgba(99,102,241,0)",
                  ],
                }
                : {}
            }
            transition={analyzing ? { repeat: Infinity, duration: 1.5 } : {}}
            className="w-full relative group"
          >
            {type === "image" && (
              <>
                <img src={preview} alt="Preview" className="w-full rounded-md" />
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      window.open(preview, "_blank");
                    }}
                    className="bg-white p-2 rounded-full shadow-lg hover:bg-gray-100"
                  >
                    <Download size={16} />
                  </button>
                </div>
              </>
            )}
            {type === "audio" && <audio controls className="w-full"><source src={preview} /></audio>}
            {type === "video" && <video controls className="w-full"><source src={preview} /></video>}
          </motion.div>
        )}

        {showAdvancedOptions && (
          <div className="space-y-2">
            <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Scan Speed</label>
            <select
              value={scanSpeed}
              onChange={(e) => setScanSpeed(e.target.value)}
              className="w-full text-sm p-2 border rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="fast">Fast (Basic Checks)</option>
              <option value="balanced">Balanced (Recommended)</option>
              <option value="thorough">Thorough (Deep Analysis)</option>
            </select>
          </div>
        )}

        <button
          onClick={() => handleAnalyze(file, setResult, setProgress, setAnalyzing)}
          className={`w-full px-4 py-2 bg-[#113742] text-white rounded-md hover:bg-gray-900 transition flex items-center justify-center gap-2 ${analyzing ? "cursor-not-allowed opacity-70" : ""
            }`}
          disabled={analyzing}
        >
          {analyzing ? (
            <>
              <Loader2 className="animate-spin" size={18} />
              Analyzing... {Math.round(progress)}%
            </>
          ) : (
            <>
              <BarChart2 size={18} />
              Analyze {label}
            </>
          )}
        </button>

        {analyzing && (
          <div className="space-y-3">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
              <div
                className="bg-[#113742] h-2.5 transition-all duration-200 ease-in-out rounded-full"
                style={{ width: `${progress}%` }}
              />
            </div>
            <Skeleton />
          </div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-4 p-3 border rounded-md shadow-md ${result.result === "Malicious"
              ? "bg-red-50 border-red-200 dark:bg-red-900/30 dark:border-red-700"
              : "bg-green-50 border-green-200 dark:bg-green-900/30 dark:border-green-700"
              }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold">
                {result.result === "Malicious" ? (
                  <ShieldAlert className="text-red-600 dark:text-red-400" />
                ) : (
                  <ShieldCheck className="text-green-600 dark:text-green-400" />
                )}
                <span className={result.result === "Malicious" ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"}>
                  {result.result}
                </span>
              </div>
              <span className="text-xs bg-white dark:bg-gray-700 px-2 py-1 rounded-full">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>

            <button
              onClick={() => {
                setSelectedHistoryItem({
                  name: file.name,
                  result: result.result,
                  confidence: result.confidence,
                  date: new Date().toLocaleString()
                });
                setShowModal(true);
              }}
              className="text-xs mt-2 text-[#113742] dark:text-indigo-400 hover:underline"
            >
              View detailed report
            </button>
          </motion.div>
        )}
      </motion.div>
    );
  };

  return (
    <>
      <Header />
      <div className="min-h-screen bg-blue-50 dark:bg-gray-900">
        <div className="max-w-7xl 2xl:max-w-[1800px] mx-auto px-4 pb-20">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="pt-8 pb-12 text-center"
          >
            <h1 className="text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent dark:from-[#113742] dark:to-[#8fbcc4] mb-2 p-2">
              StegoShield Analyzer
            </h1>
            <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Detect hidden data, watermarks, and anomalies in your files
            </p>
          </motion.div>

          {/* Quick Stats Banner */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-8 grid grid-cols-3 gap-4">
            <div className="text-center p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
              <p className="text-2xl font-bold text-gray-600 dark:text-blue-400">{history.length}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Scans</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-green-100 dark:bg-green-900/20">
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {history.filter(h => h.result === "Safe").length}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Clean Files</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-red-50 dark:bg-red-900/20">
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                {history.filter(h => h.result === "Malicious").length}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Threats Found</p>
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <div className="mb-6">
            <button
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-300"
            >
              {showAdvancedOptions ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              Advanced Options
            </button>
          </div>

          {/* Main Analysis Sections */}
          <div className="flex flex-col md:flex-row items-start gap-6 mb-12">
            {renderSection(
              "Image",
              imageFile,
              imagePreview,
              imageResult,
              setImageFile,
              setImagePreview,
              setImageResult,
              setImageProgress,
              setIsAnalyzingImage,
              imageProgress,
              isAnalyzingImage,
              "image/*,.png,.jpg,.jpeg,.webp,.gif",
              "image"
            )}
            {renderSection(
              "Audio",
              audioFile,
              audioPreview,
              audioResult,
              setAudioFile,
              setAudioPreview,
              setAudioResult,
              setAudioProgress,
              setIsAnalyzingAudio,
              audioProgress,
              isAnalyzingAudio,
              "audio/*,.mp3,.wav,.ogg,.flac,.aac",
              "audio"
            )}
            {renderSection(
              "Video",
              videoFile,
              videoPreview,
              videoResult,
              setVideoFile,
              setVideoPreview,
              setVideoResult,
              setVideoProgress,
              setIsAnalyzingVideo,
              videoProgress,
              isAnalyzingVideo,
              "video/*,.mp4,.mov,.avi,.mkv,.webm",
              "video"
            )}
          </div>

          {/* Enhanced History Section */}
          <div className="mt-12">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
              <div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
                  <History size={24} /> Scan History
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {filteredHistory.length} items • Last scan: {history[0]?.date ? new Date(history[0].date).toLocaleString() : "N/A"}
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Search history..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 border rounded-lg w-full dark:bg-gray-800 dark:border-gray-700 "
                  />
                  <svg
                    className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>

                <div className="flex gap-2">
                  <div className="relative">
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                      className="appearance-none pl-3 pr-8 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700 text-sm dark:text-gray-400"
                    >
                      <option value="date">Sort by Date</option>
                      <option value="name">Sort by Name</option>
                      <option value="confidence">Sort by Confidence</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-2.5 h-4 w-4 text-gray-400" />
                  </div>

                  <button
                    onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
                    className="px-3 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400"
                  >
                    {sortOrder === "asc" ? "↑" : "↓"}
                  </button>

                  <button
                    onClick={() => setIsFavoriteView(!isFavoriteView)}
                    className={`px-3 py-2 border rounded-lg flex items-center gap-1 ${isFavoriteView ? "bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700" : "dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400"}`}
                  >
                    <Star size={16} className={isFavoriteView ? "text-yellow-500 fill-yellow-500" : ""} />
                  </button>

                  <button
                    onClick={() => setShowClearHistoryModal(true)}
                    className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-gray-400 rounded-lg transition"
                    disabled={history.length === 0}
                  >
                    Clear
                  </button>
                </div>
              </div>
            </div>

            {/* History Filter Tabs */}
            <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
              <button
                onClick={() => setActiveTab("all")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "all" ? "bg-[#113742] text-white" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                All Scans
              </button>
              <button
                onClick={() => setActiveTab("safe")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "safe" ? "bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                Safe Files
              </button>
              <button
                onClick={() => setActiveTab("malicious")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "malicious" ? "bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                Threats
              </button>
              <button
                onClick={() => setActiveTab("image")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "image" ? "bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                Images
              </button>
              <button
                onClick={() => setActiveTab("audio")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "audio" ? "bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                Audio
              </button>
              <button
                onClick={() => setActiveTab("video")}
                className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap ${activeTab === "video" ? "bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300" : "bg-gray-200 dark:bg-gray-700 dark:text-gray-400"}`}
              >
                Videos
              </button>
            </div>

            {filteredHistory.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredHistory.map((item, i) => (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    key={item.id}
                    onClick={() => {
                      setSelectedHistoryItem({
                        ...item,
                        date: new Date(item.date).toLocaleString()
                      });
                      setShowModal(true);
                    }}
                    className={`cursor-pointer rounded-lg p-4 shadow-md border transition transform hover:scale-[1.01] relative ${item.result === "Malicious"
                      ? "bg-red-50 dark:bg-red-900/30 border-red-300 dark:border-red-700 hover:shadow-red-300 dark:hover:shadow-red-900/50"
                      : "bg-green-50 dark:bg-green-900/30 border-green-300 dark:border-green-700 hover:shadow-green-300 dark:hover:shadow-green-900/50"
                      }`}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleFavorite(item.id);
                      }}
                      className="absolute top-16 right-5 z-10"
                    >
                      <Star
                        size={18}
                        className={item.isFavorite ? "text-yellow-500 fill-yellow-500" : "text-gray-400"}
                      />
                    </button>

                    <div className="flex justify-between items-start gap-2">
                      <div className="min-w-0">
                        <h4 className="text-md font-semibold text-gray-800 dark:text-white break-words">
                          {item.name}
                        </h4>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {new Date(item.date).toLocaleString()}
                        </p>
                      </div>
                      <span
                        className={`text-xs px-2 py-1 rounded-full flex-shrink-0 ${item.result === "Malicious"
                          ? "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300"
                          : "bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300"
                          }`}
                      >
                        {item.result}
                      </span>
                    </div>

                    <div className="mt-3 flex items-center justify-between">
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Confidence: {(item.confidence * 100).toFixed(2)}%
                      </div>
                      
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 border-2 border-dashed rounded-lg">
                <p className="text-gray-500 dark:text-gray-400">
                  {searchTerm ? "No matching results found" : "Your scan history is empty"}
                </p>
                <button
                  onClick={() => {
                    setSearchTerm("");
                    setActiveTab("all");
                  }}
                  className="mt-2 text-indigo-600 dark:text-indigo-400 text-sm hover:underline"
                >
                  {searchTerm ? "Clear search" : "Upload a file to get started"}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Help Button */}
        <div className="fixed bottom-6 right-6 flex flex-col gap-3">
          <button
            onClick={() => setShowTutorial(true)}
            className="p-3 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-white rounded-full shadow-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition"
          >
            <HelpCircle size={24} />
          </button>
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="p-3 bg-[#113742] text-white rounded-full shadow-lg hover:bg-gray-900 transition"
          >
            <ChevronUp size={24} />
          </button>
        </div>

        {/* Enhanced Modal */}
        {showModal && selectedHistoryItem && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 px-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md relative shadow-2xl"
            >
              <button
                className="absolute top-4 right-4 text-gray-500 hover:text-red-500"
                onClick={() => setShowModal(false)}
              >
                <X size={24} />
              </button>

              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                Scan Details
              </h3>

              <div className="space-y-4 text-sm">
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-500 dark:text-gray-400">File Name:</span>
                  <span className="font-medium dark:text-gray-500">{selectedHistoryItem.name}</span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-500 dark:text-gray-400">Result:</span>
                  <span className={`font-medium ${selectedHistoryItem.result === "Malicious"
                    ? "text-red-600 dark:text-red-400"
                    : "text-green-600 dark:text-green-400"
                    }`}>
                    {selectedHistoryItem.result}
                  </span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
                  <span className="font-medium dark:text-gray-500">
                    {(selectedHistoryItem.confidence * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-500 dark:text-gray-400">File Type:</span>
                  <span className="font-medium dark:text-gray-500 capitalize">
                    {selectedHistoryItem.scanType || selectedHistoryItem.name.split('.').pop()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Scanned On:</span>
                  <span className="font-medium dark:text-gray-500">
                    {selectedHistoryItem.date}
                  </span>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t flex justify-center">
                <button
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 bg-[#113742] text-white rounded-lg hover:bg-gray-900 transition"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </div>
        )}

        {/* Tutorial Modal */}
        {showTutorial && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 px-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md relative shadow-2xl"
            >
              <button
                className="absolute top-4 right-4 text-gray-500 hover:text-red-500"
                onClick={() => setShowTutorial(false)}
              >
                <X size={24} />
              </button>

              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                <span className="bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent">StegoShield Analyzer</span>
              </h3>

              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <div className="bg-indigo-100 dark:bg-indigo-900/50 p-2 rounded-full flex-shrink-0">
                    <Upload size={20} className="text-indigo-600 dark:text-indigo-300" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 dark:text-white">Upload Files</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Drag and drop files into the designated zones or click to browse. Supports images, audio, and video files.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="bg-green-100 dark:bg-green-900/50 p-2 rounded-full flex-shrink-0">
                    <BarChart2 size={20} className="text-green-600 dark:text-green-300" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 dark:text-white">Analyze Content</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Click the "Analyze" button to scan for hidden data, watermarks, or anomalies using advanced steganography detection.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="bg-purple-100 dark:bg-purple-900/50 p-2 rounded-full flex-shrink-0">
                    <ShieldCheck size={20} className="text-purple-600 dark:text-purple-300" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 dark:text-white">Review Results</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Results will indicate if the file is clean or potentially malicious with a confidence percentage.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="bg-yellow-100 dark:bg-yellow-900/50 p-2 rounded-full flex-shrink-0">
                    <Star size={20} className="text-yellow-600 dark:text-yellow-300" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 dark:text-white">History & Favorites</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      All scans are saved in your history. Mark important scans as favorites for quick access later.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t flex justify-center">
                <button
                  onClick={() => setShowTutorial(false)}
                  className="px-4 py-2 bg-[#113742] text-white rounded-lg hover:bg-gray-900 transition flex items-center gap-2"
                >
                  <Zap size={18} />
                  Get Started
                </button>
              </div>
            </motion.div>
          </div>
        )}

        {/* Clear History Confirmation Modal */}
        {showClearHistoryModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 px-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md relative shadow-2xl"
            >
              <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">
                Clear All History
              </h3>

              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Are you sure you want to permanently delete all your scan history? This action cannot be undone.
              </p>

              <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-full bg-red-100 dark:bg-red-900/30">
                    <Trash2 className="text-red-500 dark:text-red-400" size={18} />
                  </div>
                  <p className="text-sm text-red-600 dark:text-red-400">
                    This will permanently remove all records from both your local history and our database.
                  </p>
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => setShowClearHistoryModal(false)}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={clearAllHistory}
                  className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition flex items-center gap-2"
                >
                  <Trash2 size={18} />
                  Delete Permanently
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>
      <Footer />
    </>
  );
};

export default FileUpload;
