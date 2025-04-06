import { motion } from "framer-motion";
import Header from "../components/Header";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { useAuth } from "../context/AuthContext";

const steps = [
  {
    step: "1. Upload File",
    description: "Upload an image, audio, or video file to begin the scan.",
  },
  {
    step: "2. AI-Based Detection",
    description:
      "Our deep learning models analyze the file to detect steganographic payloads.",
  },
  {
    step: "3. Get Results",
    description:
      "You’ll get a visual report showing if the file is safe or suspicious.",
  },
  {
    step: "4. Take Action",
    description:
      "Mark, delete, or report suspicious files to help improve platform accuracy.",
  },
];

const HowItWorks = () => {
  const { isLoggedIn } = useAuth();
  // const [check, setCheck] = useState(false);
  const navigate = useNavigate();
  useEffect(() => {
    if (!localStorage.getItem("user")) {
      toast.error("You need to be logged in to access this page.");
      navigate("/login");
    }
  }, [isLoggedIn])

  return (
    <>
      <Header />
      <motion.div
        className="min-h-screen bg-gray-100 dark:bg-gray-900 py-20 px-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-5xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-gray-800 dark:text-white mb-12">
            How StegoShield Works
          </h2>

          <div className="grid gap-8 sm:grid-cols-2">
            {steps.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-md hover:shadow-xl transition duration-300"
              >
                <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                  {item.step}
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>

          <div className="mt-14 text-center">
            <button
              onClick={() => {
                navigate("/dashboard")
              }}
              className="bg-gray-900 text-white px-6 py-3 rounded-full hover:bg-gray-800 transition duration-300 font-medium shadow"
            >
              Try It Now →
            </button>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default HowItWorks;
