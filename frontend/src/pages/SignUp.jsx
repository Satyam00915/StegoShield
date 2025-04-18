import React, { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { toast } from "react-hot-toast";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { auth, provider, signInWithPopup } from "../firebase"; // Adjust path if needed
import { useNavigate } from "react-router-dom";

const Signup = () => {
    const [formData, setFormData] = useState({
        name: "",
        email: "",
        password: "",
        confirmPassword: "",
    });

    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const navigate = useNavigate(); // ✅ Added for navigation

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    };

    const validateForm = () => {
        const { name, email, password, confirmPassword } = formData;
        const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

        if (!name.trim()) {
            toast.error("Name is required.");
            return false;
        }

        if (!emailRegex.test(email)) {
            toast.error("Please enter a valid email address.");
            return false;
        }

        if (password.length < 8 || confirmPassword.length < 8) {
            toast.error("Password must be at least 8 characters long.");
            return false;
        }

        if (password !== confirmPassword) {
            toast.error("Passwords do not match.");
            return false;
        }

        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!validateForm()) return;

        try {
            const res = await fetch("http://localhost:5000/signup", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    name: formData.name,
                    email: formData.email,
                    password: formData.password,
                }),
            });

            const data = await res.json();

            if (res.ok) {
                toast.success("Signup successful! Please log in.");
                navigate("/login"); // ✅ Redirect to login
            } else {
                toast.error(data.error || "Signup failed.");
            }
        } catch (err) {
            toast.error("Server error.");
            console.error(err);
        }
    };

    const handleGoogleSignup = async () => {
        try {
            const result = await signInWithPopup(auth, provider);
            const user = result.user;

            const payload = {
                name: user.displayName,
                email: user.email,
                uid: user.uid,
            };

            const res = await fetch("http://localhost:5000/google-signup", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    credentials: "include",
                },
                body: JSON.stringify(payload),
            });

            const data = await res.json();

            if (res.ok) {
                navigate("/login");
            } else {
                toast.error(data.error || "Google Signup failed.");
            }
        } catch (error) {
            console.error("Google sign-in error:", error);
            toast.error("Google Sign-In failed.");
        }
    };

    return (
        <div>
            <Header />
            <div className="min-h-screen flex items-center justify-center bg-blue-50 dark:bg-gray-900 px-4 relative">
                <div className="w-full max-w-md bg-white dark:bg-gray-800 p-8 rounded-3xl shadow-2xl">
                    <div className="text-center mb-6">
                        <h2 className="text-4xl font-extrabold text-gray-800 dark:text-gray-300 mb-2">Shield Up</h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Sign up to start protecting your files with AI</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div>
                            <label htmlFor="name" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1">
                                Full Name
                            </label>
                            <input
                                type="text"
                                name="name"
                                id="name"
                                onChange={handleChange}
                                className="w-full px-4 py-3 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-xl focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                placeholder="John Doe"
                            />
                        </div>

                        <div>
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1">
                                Email address
                            </label>
                            <input
                                type="email"
                                name="email"
                                id="email"
                                onChange={handleChange}
                                className="w-full px-4 py-3 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-xl focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                placeholder="you@example.com"
                            />
                        </div>

                        <div className="relative">
                            <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1">
                                Password
                            </label>
                            <input
                                type={showPassword ? "text" : "password"}
                                name="password"
                                id="password"
                                onChange={handleChange}
                                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-xl focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                placeholder="••••••••"
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute top-[38px] right-3 pr-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                            >
                                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                            </button>
                        </div>

                        <div className="relative">
                            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1">
                                Confirm Password
                            </label>
                            <input
                                type={showConfirmPassword ? "text" : "password"}
                                name="confirmPassword"
                                id="confirmPassword"
                                onChange={handleChange}
                                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-xl focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                placeholder="••••••••"
                            />
                            <button
                                type="button"
                                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                className="absolute top-[38px] right-3 pr-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                            >
                                {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                            </button>
                        </div>

                        <button
                            type="submit"
                            className="w-full py-3 text-white bg-[#1f2937] hover:bg-[#111827] dark:bg-[#405c64] dark:hover:bg-[#587d88] rounded-full font-semibold transition-all duration-200 ease-in-out"
                        >
                            Sign Up
                        </button>

                        <div className="flex items-center justify-center gap-2 my-5">
                            <div className="h-px bg-gray-300 flex-1"></div>
                            <span className="text-sm text-gray-500 dark:text-gray-400">or</span>
                            <div className="h-px bg-gray-300 flex-1"></div>
                        </div>

                        <button
                            type="button"
                            onClick={handleGoogleSignup}
                            className="w-full flex items-center justify-center gap-3 py-3 text-white bg-[#1f2937] hover:bg-[#111827] dark:bg-[#405c64] dark:hover:bg-[#587d88] rounded-full font-semibold transition-all duration-200 ease-in-out"
                        >
                            <img
                                src="https://www.svgrepo.com/show/475656/google-color.svg"
                                alt="Google"
                                className="w-5 h-5"
                            />
                            Sign up with Google
                        </button>

                        <p className="text-sm text-center text-gray-600 dark:text-gray-300 mt-3">
                            Already have an account?{" "}
                            <a href="/login" className="text-purple-600 dark:text-[#84b7c7] hover:underline font-medium">
                                Log in
                            </a>
                        </p>
                    </form>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default Signup;
