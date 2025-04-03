import React, { useState } from "react";
import { motion } from "framer-motion";
import { Input } from "../tools/Input";
import { Button } from "../tools/Neon";

function FileUpload() {
  const [imageFile, setImageFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState("");

  const uploadFile = async (file, type) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(
      `${type} Prediction: ${data.result} (Confidence: ${data.confidence}%)`
    );
  };

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#030303]">
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.05] via-transparent to-rose-500/[0.05] blur-3xl" />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeInOut" }}
        className="relative z-10 container mx-auto px-4 md:px-6 text-center"
      >
        <h1 className="text-4xl sm:text-6xl md:text-8xl font-bold mb-6 text-white">
          Upload Files
        </h1>
        <p className="text-white/60 text-lg mb-8">
          Analyze Images, Videos, and Audio for Hidden Steganographic Content
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Image Upload */}
          <div className="bg-white/10 p-6 rounded-lg shadow-lg">
            <h2 className="text-white mb-4">Upload Image</h2>
            <Input
              type="file"
              accept="image/*"
              className="mb-4"
              onChange={(e) => setImageFile(e.target.files[0])}
            />
            <Button
              onClick={() => uploadFile(imageFile, "Image")}
              className="px-4 py-2 text-white rounded"
            >
              Upload & Analyze
            </Button>
          </div>

          {/* Video Upload */}
          <div className="bg-white/10 p-6 rounded-lg shadow-lg">
            <h2 className="text-white mb-4">Upload Video</h2>
            <Input
              type="file"
              accept="video/*"
              className="mb-4"
              onChange={(e) => setVideoFile(e.target.files[0])}
            />
            <Button
              onClick={() => uploadFile(videoFile, "Video")}
              className="px-4 py-2 text-white rounded"
            >
              Upload & Analyze
            </Button>
          </div>

          {/* Audio Upload */}
          <div className="bg-white/10 p-6 rounded-lg shadow-lg">
            <h2 className="text-white mb-4">Upload Audio</h2>
            <Input
              type="file"
              accept="audio/*"
              className="mb-4"
              onChange={(e) => setAudioFile(e.target.files[0])}
            />
            <Button
              onClick={() => uploadFile(audioFile, "Audio")}
              className="px-4 py-2 text-white rounded"
            >
              Upload & Analyze
            </Button>
          </div>
        </div>

        <p className="mt-6 text-white/80">{result}</p>
      </motion.div>
    </div>
  );
}

export default FileUpload;
