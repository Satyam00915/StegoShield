import { useState } from "react";
import { toast } from "react-hot-toast";
import { Lock, Key, ArrowLeft, CheckCircle } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import axios from "axios";
import { motion } from "framer-motion";

const UpdatePassword = () => {
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  // Get email from location state (passed from ForgotPassword)
  const email = location.state?.email;
  console.log("Email from location state:", email);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!newPassword || !confirmPassword) {
      return toast.error("Please fill in all fields");
    }
    
    if (newPassword !== confirmPassword) {
      return toast.error("Passwords don't match");
    }
    
    if (newPassword.length < 6) {
      return toast.error("Password must be at least 6 characters");
    }

    setLoading(true);
    try {
      const res = await axios.post("http://stegoshield-3ius.onrender.com/api/update-password", {
        email,
        newPassword,
      });
      
      if (res.data.success) {
        toast.success("Password updated successfully");
        setSuccess(true);
        setTimeout(() => navigate("/login"), 2000);
      } else {
        toast.error(res.data.message || "Failed to update password");
      }
    } catch (err) {
      console.error(err);
      toast.error("Error updating password");
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col min-h-screen bg-blue-50 dark:bg-gray-900">
      <Header />
      <main className="flex-grow flex items-center justify-center px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-lg"
        >
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
            <div className="p-8">
              <div className="flex justify-center mb-6">
                <div className={`p-3 rounded-full ${success ? "bg-green-200 dark:bg-green-900/30" : "bg-blue-200 dark:bg-blue-900/30"}`}>
                  {success ? (
                    <CheckCircle className="h-8 w-8 text-green-600 dark:text-green-400" />
                  ) : (
                    <Key className="h-8 w-8 text-gray-800 dark:text-[#668ca5]" />
                  )}
                </div>
              </div>

              <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-gray-300 mb-2">
                {success ? "Success!" : "Set New Password"}
              </h2>
              <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
                {success
                  ? "Your password has been updated successfully. Redirecting to login..."
                  : "Create a new password for your account"}
              </p>

              {!success && (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-2">
                      New Password
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-5 w-5 text-gray-400" />
                      </div>
                      <input
                        id="newPassword"
                        type="password"
                        className="block w-full pl-10 pr-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:text-white"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="Enter new password"
                        required
                        minLength="6"
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-2">
                      Confirm Password
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-5 w-5 text-gray-400" />
                      </div>
                      <input
                        id="confirmPassword"
                        type="password"
                        className="block w-full pl-10 pr-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:text-white"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Confirm new password"
                        required
                        minLength="6"
                      />
                    </div>
                  </div>

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="submit"
                    disabled={loading}
                    className={`w-full py-3 px-4 rounded-lg font-medium text-white transition ${
                      loading
                        ? "bg-gray-400 dark:bg-gray-600 cursor-not-allowed"
                        : "bg-gray-800 hover:bg-gray-700 dark:bg-[#405c64] dark:hover:bg-[#587d88]"
                    }`}
                  >
                    {loading ? (
                      <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Updating...
                      </span>
                    ) : (
                      "Update Password"
                    )}
                  </motion.button>
                </form>
              )}

              <div className="mt-6 text-center">
                <Link
                  to="/login"
                  className="inline-flex items-center text-sm text-gray-800 dark:text-[#68848d] hover:underline font-medium transition"
                >
                  <ArrowLeft className="h-4 w-4 mr-1" />
                  Back to login
                </Link>
              </div>
            </div>
          </div>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
};

export default UpdatePassword;