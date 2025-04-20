import { useState } from "react";
import { toast } from "react-hot-toast";
import { Mail, Lock, ArrowLeft } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import axios from "axios";
import { motion } from "framer-motion";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const Navigate = useNavigate();
  const [otpSent, setOtpSent] = useState(false);
  const [otp, setOtp] = useState(["", "", "", ""]);
  const [loading, setLoading] = useState(false);
  const [otpVerified, setOtpVerified] = useState(false);

  const handleSendOtp = async (e) => {
    e.preventDefault();
    if (!email) return toast.error("Please enter your email");

    setLoading(true);
    try {
      const res = await axios.post("https://stegoshield-3ius.onrender.com/api/send-otp", { email });
      if (res.data.success) {
        toast.success("OTP sent to your email");
        setOtpSent(true);
      } else {
        toast.error(res.data.message || "Failed to send OTP");
      }
    } catch (err) {
      console.error(err);
      toast.error("Error sending OTP");
    }
    setLoading(false);
  };

  const handleOtpChange = (index, value) => {
    if (/^\d?$/.test(value)) {
      const newOtp = [...otp];
      newOtp[index] = value;
      setOtp(newOtp);
      if (value && index < 3) {
        document.getElementById(`otp-${index + 1}`).focus();
      }
    }
  };

  const handleVerifyOtp = async (e) => {
    e.preventDefault();
    const fullOtp = otp.join("");
    if (fullOtp.length !== 4) return toast.error("Enter all 4 digits");

    setLoading(true);
    try {
      const res = await axios.post("https://stegoshield-3ius.onrender.com/api/verify-otp", {
        email,
        otp: fullOtp,
      });
      if (res.data.success) {
        toast.success("OTP verified");
        Navigate("/update-password", { state: { email } });
        setOtpVerified(true);
      } else {
        toast.error(res.data.message || "Invalid OTP");
      }
    } catch (err) {
      console.error(err);
      toast.error("OTP verification failed");
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
                <div className="bg-blue-200 dark:bg-blue-900/30 p-3 rounded-full">
                  <Lock className="h-8 w-8 text-gray-800 dark:text-[#668ca5]" />
                </div>
              </div>

              <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-gray-300 mb-2">
                {otpSent ? "Verify OTP" : "Reset Password"}
              </h2>
              <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
                {otpVerified
                  ? "OTP verified! You can now reset your password."
                  : otpSent
                    ? "Enter the 4-digit code sent to your email"
                    : "Enter your email to receive a verification code"}
              </p>

              <form onSubmit={otpSent ? handleVerifyOtp : handleSendOtp} className="space-y-6">
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-400 mb-2">
                    Email address
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Mail className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      id="email"
                      type="email"
                      className="block w-full pl-10 pr-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent dark:text-white"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                      required
                      disabled={otpSent || loading}
                    />
                  </div>
                </div>

                {otpSent && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Verification code
                    </label>
                    <div className="flex justify-around gap-1">
                      {otp.map((digit, idx) => (
                        <input
                          key={idx}
                          id={`otp-${idx}`}
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          maxLength="1"
                          value={digit}
                          onChange={(e) => handleOtpChange(idx, e.target.value)}
                          className="w-16 h-16 text-center text-2xl font-semibold border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                          disabled={loading || otpVerified}
                        />
                      ))}
                    </div>
                    {!otpVerified && (
                      <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                        Didn't receive code?{" "}
                        <button
                          type="button"
                          onClick={() => {
                            setOtp(["", "", "", ""]);
                            setOtpSent(false);
                          }}
                          className="text-blue-600 dark:text-blue-400 hover:underline focus:outline-none"
                        >
                          Resend
                        </button>
                      </p>
                    )}
                  </motion.div>
                )}

                {!otpVerified && (
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="submit"
                    disabled={loading}
                    className={`w-full py-3 px-4 rounded-lg font-medium text-white transition ${loading
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
                        Processing...
                      </span>
                    ) : otpSent ? "Verify Code" : "Send Verification Code"}
                  </motion.button>
                )}

                {otpVerified && (
                  <p className="text-green-600 dark:text-green-400 text-center font-medium mt-4">
                    OTP verified! Now you can reset your password.
                  </p>
                )}
              </form>

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

export default ForgotPassword;
