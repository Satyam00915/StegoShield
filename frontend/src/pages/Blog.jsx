import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import { useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import toast from "react-hot-toast";
import {
  BookText,
  Brain,
  ShieldAlert,
  Eye,
  Code2,
  FileSearch,
} from "lucide-react";

const blogs = [
  {
    title: "What is Steganography?",
    summary:
      "A quick introduction to digital steganography and how attackers hide malicious payloads.",
    date: "April 2, 2025",
    icon: <BookText className="w-6 h-6 text-indigo-500" />,
  },
  {
    title: "Detecting Stego Files with AI",
    summary:
      "We explore how deep learning models like CNNs can be used to detect hidden data in images and audio.",
    date: "April 4, 2025",
    icon: <Brain className="w-6 h-6 text-indigo-500" />,
  },
  {
    title: "Real-World Stego Attacks",
    summary:
      "A look at some real-world cases where steganography was used for malicious purposes.",
    date: "April 5, 2025",
    icon: <ShieldAlert className="w-6 h-6 text-indigo-500" />,
  },
  {
    title: "Understanding Visual Steganalysis",
    summary:
      "How to visually inspect images for stego content using heatmaps and filters.",
    date: "April 6, 2025",
    icon: <Eye className="w-6 h-6 text-indigo-500" />,
  },
  {
    title: "Behind the StegoShield AI Engine",
    summary:
      "Dive deep into the CNN, RNN, and EfficientNet-LSTM models powering StegoShield’s detection.",
    date: "April 7, 2025",
    icon: <Code2 className="w-6 h-6 text-indigo-500" />,
  },
  {
    title: "Manual vs. Automated Detection",
    summary:
      "A comparison between human analysis and AI-driven detection techniques in cybersecurity.",
    date: "April 8, 2025",
    icon: <FileSearch className="w-6 h-6 text-indigo-500" />,
  },
];

const Blog = () => {
    const navigate = useNavigate();
    const { isLoggedIn } = useAuth();
  
    useEffect(() => {
      if (!localStorage.getItem("user")) {
        toast.error("You need to be logged in to access this page.");
        navigate("/login");
      }
    }, [isLoggedIn]);
  
    return (
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
        <Header />
        <motion.div
          className="py-20 px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="max-w-6xl mx-auto">
            <motion.h2
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-5xl font-extrabold text-center bg-gradient-to-r from-gray-900 to-indigo-400 bg-clip-text text-transparent mb-16"
            >
              StegoShield Blog
            </motion.h2>
  
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-10">
              {blogs.map((blog, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  whileHover={{ scale: 1.02 }}
                  className="group relative flex overflow-hidden rounded-2xl shadow-xl bg-white dark:bg-zinc-900 border border-gray-200 dark:border-zinc-700 transition-all duration-300"
                >
                  <div className="w-1.5 bg-gradient-to-b via-purple-500 from-gray-900 group-hover:scale-y-110 transition-transform duration-300" />
                  <div className="p-6 flex flex-col justify-between">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-3 bg-indigo-100 dark:bg-indigo-500/20 rounded-full shadow-inner">
                        {blog.icon}
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 dark:text-white leading-tight">
                        {blog.title}
                      </h3>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                      {blog.date}
                    </p>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {blog.summary}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
  
            <motion.div
              className="mt-20 text-center"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.8 }}
            >
              <motion.button
                whileHover={{ scale: 1.06 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate("/how-it-works")}
                className="bg-gray-900 text-white px-6 py-3 rounded-full font-semibold shadow-lg hover:bg-gray-800 transition-all duration-300"
              >
                Learn How It Works
              </motion.button>
            </motion.div>
          </div>
        </motion.div>
      </div>
    );
  };
  

export default Blog;
