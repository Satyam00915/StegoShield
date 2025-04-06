import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import { useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import toast from "react-hot-toast";

const blogs = [
    {
        title: "What is Steganography?",
        summary:
            "A quick introduction to digital steganography and how attackers hide malicious payloads.",
        date: "April 2, 2025",
    },
    {
        title: "Detecting Stego Files with AI",
        summary:
            "We explore how deep learning models like CNNs can be used to detect hidden data in images and audio.",
        date: "April 4, 2025",
    },
    {
        title: "Real-World Stego Attacks",
        summary:
            "A look at some real-world cases where steganography was used for malicious purposes.",
        date: "April 5, 2025",
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
    }, [isLoggedIn])

    return (
        <>
            <Header />
            <div className="min-h-screen bg-gray-100 dark:bg-gray-900 py-16 px-4">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-8 text-center">
                        StegoShield Blog
                    </h2>
                    <div className="space-y-6">
                        {blogs.map((blog, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md"
                            >
                                <h3 className="text-xl font-semibold text-indigo-600">
                                    {blog.title}
                                </h3>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                                    {blog.date}
                                </p>
                                <p className="text-gray-700 dark:text-gray-200">{blog.summary}</p>
                            </motion.div>
                        ))}
                    </div>

                    {/* CTA Button */}
                    <div className="mt-12 text-center">
                        <button
                            onClick={() => navigate("/how-it-works")}
                            className="bg-gray-900 text-white px-6 py-2 rounded-full hover:bg-gray-800 transition duration-300 font-medium"
                        >
                            Explore How It Works →
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Blog;
