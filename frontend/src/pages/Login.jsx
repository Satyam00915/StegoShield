import React, { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { auth, provider, signInWithPopup  } from "../firebase";
import { toast } from "react-hot-toast";

const Login = () => {
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            const res = await fetch("http://localhost:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            const data = await res.json();

            if (res.ok) {
                localStorage.setItem("user", JSON.stringify(data.user));
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

            const res = await fetch("http://localhost:5000/google-login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ idToken }),
            });

            const data = await res.json();

            if (res.ok) {
                const name = localStorage.setItem("user", JSON.stringify(data.user));
                toast.success(`Welcome, ${name}`);
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
        <section className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center px-4 relative">
            <a
                href="/"
                className="absolute top-6 left-6 flex items-center text-gray-600 hover:text-gray-800"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-5 mr-1"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                >
                    <path
                        fillRule="evenodd"
                        d="M7.707 14.707a1 1 0 01-1.414 0L2.586 11H17a1 1 0 110 2H2.586l3.707 3.707a1 1 0 01-1.414 1.414l-5.121-5.12a1 1 0 010-1.415l5.121-5.12a1 1 0 111.414 1.414L2.586 9H17a3 3 0 110 6H2.586l3.707 3.707a1 1 0 010 1.414z"
                        clipRule="evenodd"
                    />
                </svg>
                Home
            </a>

            <div className="max-w-md w-full bg-white p-10 rounded-3xl shadow-2xl">
                <h2 className="text-4xl font-extrabold text-gray-900 mb-2 text-center">Welcome Back</h2>
                <p className="text-center text-gray-600 mb-8 text-sm">
                    Log in to access your StegoShield dashboard
                </p>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
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
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                        />
                    </div>

                    <div className="relative">
                        <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
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
                            className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                        />
                        <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute top-9 right-3 pr-1 text-gray-500 hover:text-gray-700"
                        >
                            {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                        </button>
                    </div>

                    <div className="text-right">
                        <a href="/forgot-password" className="text-sm text-purple-600 hover:underline">
                            Forgot password?
                        </a>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-3 text-white bg-gray-800 hover:bg-gray-700 rounded-full font-medium transition duration-300"
                    >
                        {loading ? "Logging in..." : "Log In"}
                    </button>

                    <div className="flex items-center justify-center gap-2 my-4">
                        <div className="h-px bg-gray-300 flex-1"></div>
                        <span className="text-sm text-gray-500">or</span>
                        <div className="h-px bg-gray-300 flex-1"></div>
                    </div>

                    <button
                        type="button"
                        onClick={handleGoogleLogin}
                        className="w-full py-3 text-white bg-gray-800 hover:bg-gray-700 rounded-full font-medium transition duration-300 flex items-center justify-center gap-2"
                    >
                        <img
                            src="https://www.svgrepo.com/show/475656/google-color.svg"
                            alt="Google"
                            className="w-5 h-5"
                        />
                        Continue with Google
                    </button>

                    <p className="text-center text-sm text-gray-600 mt-4">
                        Don’t have an account?{" "}
                        <a href="/signup" className="text-purple-600 hover:underline font-medium">
                            Sign Up
                        </a>
                    </p>
                </form>
            </div>
        </section>
    );
};

export default Login;
