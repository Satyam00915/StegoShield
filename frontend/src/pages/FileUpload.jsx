// FileUploader.jsx

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { Loader2, Upload, ShieldCheck, ShieldAlert, File } from "lucide-react";
import Header from "../components/Header";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useDropzone } from "react-dropzone";

const FileUploader = () => {
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

  const [history, setHistory] = useState([]);

  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();

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

  const saveToLocalHistory = (name, result, confidence) => {
    const newItem = {
      name,
      result,
      confidence,
      date: new Date().toLocaleString(),
    };
    const updated = [newItem, ...history].slice(0, 10);
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

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/upload");
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
        className="cursor-pointer border-2 border-dashed rounded-lg p-4 text-center transition hover:bg-gray-100 dark:hover:bg-gray-800"
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-sm text-gray-700 dark:text-gray-300">Drop the file here...</p>
        ) : (
          <p className="text-sm text-gray-600 dark:text-gray-400 flex justify-center items-center gap-1">
            <File size={18} /> Drag & Drop or Click to Upload {type}
          </p>
        )}
      </div>
    );
  };

  const Skeleton = () => (
    <div className="animate-pulse flex flex-col space-y-4 mt-3">
      <div className="h-40 bg-gray-200 dark:bg-gray-700 rounded-md" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mx-auto" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mx-auto" />
    </div>
  );

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
        className="w-full md:w-1/3 space-y-4 bg-white dark:bg-[#111827] p-5 rounded-lg shadow-lg"
      >
        <h3 className="text-lg font-bold text-center text-gray-700 dark:text-white">{label}</h3>

        <DropZone
          type={type}
          accept={accept}
          setFileState={setFile}
          setPreview={setPreview}
          setResult={setResult}
          setProgress={setProgress}
        />

        {file && (
          <div className="text-xs bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-gray-700 dark:text-gray-300">
            <p><strong>Name:</strong> {file.name}</p>
            <p><strong>Size:</strong> {(file.size / 1024).toFixed(2)} KB</p>
            <p><strong>Type:</strong> {file.type}</p>
          </div>
        )}

        {preview && (
          <motion.div
            animate={
              analyzing
                ? {
                    scale: [1, 1.05, 1],
                    boxShadow: [
                      "0 0 0px rgba(99,102,241,0)",
                      "0 0 15px rgba(99,102,241,0.6)",
                      "0 0 0px rgba(99,102,241,0)",
                    ],
                  }
                : {}
            }
            transition={analyzing ? { repeat: Infinity, duration: 1.5 } : {}}
            className="w-full"
          >
            {type === "image" && <img src={preview} alt="Preview" className="w-full rounded-md" />}
            {type === "audio" && <audio controls className="w-full"><source src={preview} /></audio>}
            {type === "video" && <video controls className="w-full"><source src={preview} /></video>}
          </motion.div>
        )}

        <button
          onClick={() => handleAnalyze(file, setResult, setProgress, setAnalyzing)}
          className={`w-full px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 ${analyzing ? "cursor-not-allowed opacity-70" : ""}`}
          disabled={analyzing}
        >
          {analyzing ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="animate-spin" size={18} /> Analyzing...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <Upload size={18} /> Analyze {label}
            </span>
          )}
        </button>

        {analyzing && (
          <>
            <div className="w-full bg-gray-300 dark:bg-gray-700 rounded-full h-2.5 mt-1">
              <div
                className="bg-indigo-600 h-2.5 transition-all duration-200 ease-in-out rounded-full"
                style={{ width: `${progress}%` }}
              />
            </div>
            <Skeleton />
          </>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-4 p-3 border rounded-md shadow-md text-center ${result.result === "Malicious"
              ? "bg-red-100 dark:bg-red-900"
              : "bg-green-100 dark:bg-green-900"
              }`}
          >
            <div className="flex items-center justify-center gap-2 font-semibold text-lg">
              {result.result === "Malicious" ? (
                <ShieldAlert className="text-red-600" />
              ) : (
                <ShieldCheck className="text-green-600" />
              )}
              {result.result}
            </div>
            <p className="text-sm mt-1 text-gray-600 dark:text-gray-300">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>
          </motion.div>
        )}
      </motion.div>
    );
  };

  return (
    <>
      <Header />
      <div className="max-w-7xl mx-auto my-10 px-4">
        <motion.h2
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl font-extrabold bg-gradient-to-r from-gray-900 to-indigo-400 bg-clip-text text-transparent text-center dark:text-white mb-10"
        >
          🛡️ StegoShield - File Analyzer
        </motion.h2>

        <div className="flex flex-col md:flex-row gap-6">
          {renderSection("Image", imageFile, imagePreview, imageResult, setImageFile, setImagePreview, setImageResult, setImageProgress, setIsAnalyzingImage, imageProgress, isAnalyzingImage, "image/*", "image")}
          {renderSection("Audio", audioFile, audioPreview, audioResult, setAudioFile, setAudioPreview, setAudioResult, setAudioProgress, setIsAnalyzingAudio, audioProgress, isAnalyzingAudio, "audio/*", "audio")}
          {renderSection("Video", videoFile, videoPreview, videoResult, setVideoFile, setVideoPreview, setVideoResult, setVideoProgress, setIsAnalyzingVideo, videoProgress, isAnalyzingVideo, "video/*", "video")}
        </div>

        {history.length > 0 && (
          <div className="mt-12">
            <h3 className="text-2xl font-bold text-center text-gray-800 dark:text-white mb-6">🕒 Recent StegoScan Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {history.map((item, i) => (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  key={i}
                  className={`rounded-lg p-4 shadow-md border transition transform hover:scale-[1.01] 
                    ${item.result === "Malicious"
                      ? "bg-red-50 dark:bg-red-900 border-red-400 hover:shadow-xl hover:border-red-500"
                      : "bg-green-50 dark:bg-green-900 border-green-400 hover:border-green-500 hover:shadow-xl"
                    }`}
                >
                  <h4 className="text-md font-semibold text-gray-800 dark:text-white truncate">{item.name}</h4>
                  <p className="text-sm mt-1 text-gray-700 dark:text-gray-300">
                    <span className="font-medium">Result:</span>{" "}
                    <span className={`font-bold ${item.result === "Malicious" ? "text-red-600" : "text-green-600"}`}>
                      {item.result}
                    </span>
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    <span className="font-medium">Confidence:</span> {(item.confidence * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs mt-2 text-gray-500 dark:text-gray-400 italic">{item.date}</p>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default FileUploader;
