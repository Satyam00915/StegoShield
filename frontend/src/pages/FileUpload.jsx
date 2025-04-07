import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { Loader2, Upload, ShieldCheck, ShieldAlert } from "lucide-react";
import Header from "../components/Header";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const FileUploader = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [fileType, setFileType] = useState("");
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
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
  }, [isLoggedIn])

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

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    setResult(null);
    setUploadProgress(0);

    if (uploadedFile) {
      const type = uploadedFile.type;
      setFileType(type.split("/")[0]);

      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result);
      if (["image", "audio", "video"].some((t) => type.startsWith(t))) {
        reader.readAsDataURL(uploadedFile);
      } else {
        setPreviewUrl("");
      }
    }
  };

  const handleAnalyze = () => {
    if (!file) return toast.error("Upload a file first!");

    setIsAnalyzing(true);
    setResult(null);
    toast.loading("Uploading & Analyzing...");

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/api/predict");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        setUploadProgress(percent);
      }
    };

    xhr.onload = () => {
      toast.dismiss();
      setIsAnalyzing(false);
      try {
        const response = JSON.parse(xhr.responseText);
        setResult(response);
        toast.success("Analysis Complete ✅");
        saveToLocalHistory(file.name, response.result, response.confidence);
      } catch (err) {
        toast.error("Invalid response");
      }
    };

    xhr.onerror = () => {
      setIsAnalyzing(false);
      toast.dismiss();
      toast.error("Upload failed ❌");
    };

    xhr.send(formData);
  };

  const renderPreview = () => {
    if (!previewUrl) return null;

    if (fileType === "image") {
      return <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} src={previewUrl} alt="Preview" className="mt-4 w-full rounded-lg border" />;
    } else if (fileType === "audio") {
      return <motion.audio initial={{ opacity: 0 }} animate={{ opacity: 1 }} controls className="mt-4 w-full"><source src={previewUrl} /></motion.audio>;
    } else if (fileType === "video") {
      return <motion.video initial={{ opacity: 0 }} animate={{ opacity: 1 }} controls className="mt-4 w-full rounded-lg border"><source src={previewUrl} /></motion.video>;
    }
    return null;
  };

  return (
    <>
      <Header />

      <div className="max-w-xl mx-auto my-10 p-6 bg-white dark:bg-gray-900 shadow-xl rounded-xl space-y-6">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white">🔍 StegoShield - File Analyzer</h2>

        <input
          type="file"
          accept="image/*,audio/*,video/*"
          onChange={handleFileChange}
          className="w-full bg-gray-100 dark:bg-gray-800 rounded-md px-4 py-2 text-sm"
        />

        {file && (
          <div className="text-sm mt-2 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-gray-700 dark:text-gray-300">
            <p><strong>Name:</strong> {file.name}</p>
            <p><strong>Size:</strong> {(file.size / 1024).toFixed(2)} KB</p>
            <p><strong>Type:</strong> {file.type}</p>
          </div>
        )}

        {renderPreview()}

        <button
          onClick={handleAnalyze}
          className={`w-full flex items-center justify-center gap-2 mt-2 px-6 py-2 ${isAnalyzing ? "bg-gray-400 cursor-not-allowed" : " text-white bg-gray-800 hover:bg-gray-700"
            } text-white rounded-full transition`}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="animate-spin" size={20} /> Analyzing...
            </>
          ) : (
            <>
              <Upload size={20} /> Analyze File
            </>
          )}
        </button>

        {isAnalyzing && (
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mt-2 overflow-hidden">
            <div
              className="bg-indigo-600 h-2.5 transition-all duration-200 ease-in-out"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className={`mt-6 p-4 border rounded-lg text-center shadow-md ${result.result === "Malicious" ? "bg-red-50 dark:bg-red-900" : "bg-green-50 dark:bg-green-900"
              }`}
          >
            <div className="flex items-center justify-center gap-2 text-xl font-semibold">
              {result.result === "Malicious" ? <ShieldAlert className="text-red-600" /> : <ShieldCheck className="text-green-600" />}
              Prediction:{" "}
              <span className={result.result === "Malicious" ? "text-red-600" : "text-green-600"}>
                {result.result}
              </span>
            </div>
            <div className="mt-2">
              <p className="text-sm text-gray-700 dark:text-gray-200">Confidence Level</p>
              <div className="w-full h-3 bg-gray-300 dark:bg-gray-700 rounded-full mt-1">
                <div
                  className={`h-3 rounded-full ${result.result === "Malicious" ? "bg-red-500" : "bg-green-500"
                    }`}
                  style={{ width: `${(result.confidence * 100).toFixed(2)}%` }}
                />
              </div>
              <p className="text-xs text-gray-600 mt-1 dark:text-gray-400">{(result.confidence * 100).toFixed(2)}%</p>
            </div>
          </motion.div>
        )}

        {history.length > 0 && (
          <div className="mt-8">
            <h3 className="font-semibold text-gray-700 dark:text-gray-200 mb-2">📁 Upload History</h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              {history.map((item, i) => (
                <li key={i} className="border-b py-1">
                  <strong>{item.name}</strong> - {item.result} ({(item.confidence * 100).toFixed(2)}%) on {item.date}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </>
  );
};

export default FileUploader;
