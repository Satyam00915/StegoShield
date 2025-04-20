import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { AuthProvider } from "./context/AuthContext";

import FileUpload from "./pages/FileUpload";
import Landing from "./pages/Landing";
import Signup from "./pages/SignUp";
import Login from "./pages/Login";
import ForgotPassword from "./pages/ForgotPassword";
import Dashboard from "./pages/Dashboard";
import Profile from "./pages/Profile";
import Blog from "./pages/Blog";
import HowItWorks from "./pages/HowItWorks";
import UpdatePassword from "./pages/UpdatePassword";

import "./components/landing.css";

function App() {
    return (
        <>
            <Toaster position="top-right" reverseOrder={false} toastOptions={{ duration: 3000 }} />

            <AuthProvider>
                <BrowserRouter>
                    <Routes>
                        <Route path="/" element={<Landing />} />
                        <Route path="/signup" element={<Signup />} />
                        <Route path="/login" element={<Login />} />
                        <Route path="/forgot-password" element={<ForgotPassword />} />
                        <Route path="/update-password" element={<UpdatePassword />} />

                        <Route
                            path="/dashboard"
                            element={
                                <Dashboard />
                            }
                        />
                        <Route
                            path="/upload"
                            element={
                                <FileUpload />
                            }
                        />
                        <Route
                            path="/profile"
                            element={
                                <Profile />
                            }
                        />

                        {/* Public Routes */}
                        <Route path="/blog" element={<Blog />} />
                        <Route path="/how-it-works" element={<HowItWorks />} />

                        {/* Fallback 404 */}
                        <Route
                            path="*"
                            element={<h2 className="text-center mt-10 text-xl">404 - Page Not Found</h2>}
                        />
                    </Routes>
                </BrowserRouter>
            </AuthProvider>
        </>
    );
}

export default App;
