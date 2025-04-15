import { motion } from "framer-motion";
import Header from "../components/Header";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import toast from "react-hot-toast";
import { useAuth } from "../context/AuthContext";

const steps = [
  {
    step: "1. Upload File",
    description: "Upload an image, audio, or video file to begin the scan.",
    icon: "📤",
  },
  {
    step: "2. AI-Based Detection",
    description:
      "Our deep learning models analyze the file to detect steganographic payloads.",
    icon: "🤖",
  },
  {
    step: "3. Get Results",
    description:
      "You’ll get a visual report showing if the file is safe or suspicious.",
    icon: "📊",
  },
  {
    step: "4. Take Action",
    description:
      "Mark, delete, or report suspicious files to help improve platform accuracy.",
    icon: "🛡️",
  },
];

const HowItWorks = () => {
  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

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
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-5xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-5xl font-extrabold text-center bg-gradient-to-r from-gray-900 to-indigo-400 bg-clip-text text-transparent mb-14"
          >
            StegoShield Work Flow
          </motion.h2>

          <div className="grid sm:grid-cols-2 gap-10">
            {steps.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300"
              >
                <div className="text-5xl mb-4">{item.icon}</div>
                <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-3">
                  {item.step}
                </h3>
                <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1, duration: 0.4 }}
            className="mt-16 text-center"
          >
            <button
              onClick={() => navigate("/dashboard")}
              className="bg-gray-900 text-white px-8 py-3 rounded-full hover:bg-gray-800 transition duration-300 font-semibold text-lg shadow-md"
            >
              Try It Now →
            </button>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default HowItWorks;
