import React, { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { auth, provider, signInWithPopup } from "../firebase";
import { toast } from "react-hot-toast";

const Login = () => {
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            const res = await fetch("https://stegoshield-3ius.onrender.com/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
                credentials: "include",
            });

            const data = await res.json();

            if (res.ok) {
                localStorage.setItem("user", JSON.stringify(data.user));
                setUser(data.user);
                toast.success("Login successful!");
                navigate("/dashboard");
            } else {
                toast.error(data.message || "Invalid credentials");
            }
        } catch (err) {
            toast.error("Something went wrong!");
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        try {
            const result = await signInWithPopup(auth, provider);
            const idToken = await result.user.getIdToken();

            const res = await fetch("https://stegoshield-3ius.onrender.com/google-login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ idToken }),
                credentials: "include",
            });

            const data = await res.json();

            if (res.ok) {
                const name = localStorage.setItem("user", JSON.stringify(data.user));
                setUser(data.user);
                toast.success(`Welcome, ${user?.name || "USER"}`);
                navigate("/dashboard");
            } else {
                toast.error(data.message || "Login failed");
            }
        } catch (error) {
            console.error("Google login error:", error);
            toast.error("Google login failed.");
        }
    };

    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <section className="flex-grow bg-blue-50 dark:bg-gray-900 flex items-center justify-center px-4 py-12 relative">
                <div className="max-w-md w-full bg-white dark:bg-gray-800 p-8 rounded-3xl shadow-2xl 
                               lg:max-w-xl lg:p-12 2xl:max-w-2xl 2xl:p-16">
                    <h2 className="text-4xl font-extrabold text-gray-900 dark:text-gray-300 mb-2 text-center
                                  lg:text-5xl 2xl:text-6xl">
                        Welcome Back
                    </h2>
                    <p className="text-center text-gray-600 dark:text-gray-400 mb-8 text-sm
                                lg:text-base lg:mb-10 2xl:text-lg">
                        Log in to access your StegoShield dashboard
                    </p>

                    <form onSubmit={handleSubmit} className="space-y-6 lg:space-y-8 2xl:space-y-10">
                        <div>
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1
                                                           lg:text-base 2xl:text-lg">
                                Email Address
                            </label>
                            <input
                                id="email"
                                name="email"
                                type="email"
                                required
                                value={formData.email}
                                onChange={handleChange}
                                placeholder="you@example.com"
                                className="w-full px-4 py-3 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none
                                          lg:py-4 lg:text-base 2xl:py-5 2xl:text-lg"
                            />
                        </div>

                        <div className="relative">
                            <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-1
                                                              lg:text-base 2xl:text-lg">
                                Password
                            </label>
                            <input
                                id="password"
                                name="password"
                                type={showPassword ? "text" : "password"}
                                required
                                value={formData.password}
                                onChange={handleChange}
                                placeholder="••••••••"
                                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:bg-gray-800 dark:text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none
                                          lg:py-4 lg:text-base 2xl:py-5 2xl:text-lg"
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute top-9 right-3 pr-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300
                                          lg:top-10 lg:right-4 2xl:top-12 2xl:right-5"
                            >
                                {showPassword ? <EyeOff size={24} className="lg:w-6 lg:h-6 2xl:w-7 2xl:h-7" /> :
                                    <Eye size={24} className="lg:w-6 lg:h-6 2xl:w-7 2xl:h-7" />}
                            </button>
                        </div>

                        <div className="text-right">
                            <a href="/forgot-password" className="text-sm text-purple-600 dark:text-[#84b7c7] hover:underline
                                                               lg:text-base 2xl:text-lg">
                                Forgot password?
                            </a>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 text-white bg-gray-800 hover:bg-gray-700 rounded-full font-medium transition duration-300 dark:bg-[#405c64] dark:hover:bg-[#587d88]
                                     lg:py-4 lg:text-lg 2xl:py-5 2xl:text-xl"
                        >
                            {loading ? "Logging in..." : "Log In"}
                        </button>

                        <div className="flex items-center justify-center gap-2 my-4 lg:my-6 2xl:my-8">
                            <div className="h-px bg-gray-300 flex-1"></div>
                            <span className="text-sm text-gray-500 dark:text-gray-400 lg:text-base 2xl:text-lg">or</span>
                            <div className="h-px bg-gray-300 flex-1"></div>
                        </div>

                        <button
                            type="button"
                            onClick={handleGoogleLogin}
                            className="w-full py-3 text-white bg-gray-800 hover:bg-gray-700 dark:bg-[#405c64] dark:hover:bg-[#587d88] rounded-full font-medium transition duration-300 flex items-center justify-center gap-2
                                     lg:py-4 lg:text-lg 2xl:py-5 2xl:text-xl"
                        >
                            <img
                                src="https://www.svgrepo.com/show/475656/google-color.svg"
                                alt="Google"
                                className="w-5 h-5 lg:w-6 lg:h-6 2xl:w-7 2xl:h-7"
                            />
                            Continue with Google
                        </button>

                        <p className="text-center text-sm text-gray-600 dark:text-gray-300 mt-4 lg:text-base lg:mt-6 2xl:text-lg 2xl:mt-8">
                            Don't have an account?{" "}
                            <a href="/signup" className="text-purple-600 dark:text-[#84b7c7] hover:underline font-medium">
                                Sign Up
                            </a>
                        </p>
                    </form>
                </div>
            </section>
            <Footer />
        </div>
    );
};

export default Login;