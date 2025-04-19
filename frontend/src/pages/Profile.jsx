import { useState, useEffect, useRef } from "react";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Pencil, Eye, EyeOff, Check, X, ChevronDown } from "lucide-react";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { useAuth } from "../context/AuthContext";

const getPasswordStrength = (password) => {
  if (!password) return "Weak";
  if (password.length >= 8 && /\d/.test(password) && /[A-Z]/.test(password)) return "Strong";
  if (password.length >= 6) return "Medium";
  return "Weak";
};

const Profile = () => {
  const [profile, setProfile] = useState({
    id: "",
    name: "",
    email: "",
    password: "",
    oldPassword: "",
    avatar: "",
    theme: "light",
  });
  const [lastUpdated, setLastUpdated] = useState("");
  const [showOldPassword, setShowOldPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [isThemeDropdownOpen, setIsThemeDropdownOpen] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const isLoggedIn = useAuth();

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user")) || {
      id: "",
      name: "CyberNova",
      email: "cybernova@stegoshield.ai",
      avatar: "",
      theme: "light",
    };
    setProfile({ ...user, password: "", oldPassword: "" });
    setLastUpdated(localStorage.getItem("lastUpdated") || "");
  }, []);

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) {
      toast.error("You need to be logged in to access this page.");
      return navigate("/login");
    }
  }, [isLoggedIn]);

  const handleChange = (e) => {
    setProfile({ ...profile, [e.target.name]: e.target.value });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setProfile((prev) => ({ ...prev, avatar: reader.result }));
      toast.success("Profile picture uploaded!");
    };
    reader.readAsDataURL(file);
  };

  const handleSave = async () => {
    if (!profile.name || !profile.email) {
      toast.error("Name and Email are required!");
      return;
    }

    const updatedProfile = {
      name: profile.name,
      email: profile.email,
      oldPassword: profile.oldPassword || null,
      password: profile.password || null,
      avatar: profile.avatar,
      theme: profile.theme,
    };

    try {
      const res = await fetch(`https://stegoshield-3ius.onrender.com/api/user/${profile.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updatedProfile),
      });

      if (!res.ok) {
        const errorData = await res.json();
        toast.error(errorData.message || "Failed to update profile!");
        return;
      }

      const data = await res.json();
      localStorage.setItem("user", JSON.stringify(data));
      const timestamp = new Date().toLocaleString();
      localStorage.setItem("lastUpdated", timestamp);
      setLastUpdated(timestamp);
      toast.success("Profile updated successfully!");

      setTimeout(() => {
        window.location.href = "/dashboard";
      }, 1000);
    } catch (err) {
      console.error(err);
      toast.error("Something went wrong!");
    }
  };

  const strength = getPasswordStrength(profile.password);
  const strengthColors = {
    Strong: "bg-green-500",
    Medium: "bg-yellow-500",
    Weak: "bg-red-500",
  };

  const themes = [
    { name: "Light", value: "light" },
    { name: "Dark", value: "dark" },
    { name: "System", value: "system" },
  ];

  return (
    <div className="min-h-screen bg-blue-50  dark:bg-gray-900">
      {/* Transparent Header */}
      <div className="bg-transparent shadow-none">
        <Header />
      </div>

      <div className="py-10 px-4 ">
        <div className="max-w-3xl mx-auto bg-blue-100/80 dark:bg-gray-800/90 backdrop-blur-sm rounded-2xl shadow-xl p-8 space-y-8 border border-blue-300/50 dark:border-gray-700/50">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col items-center"
          >
            <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-2">
              Profile Settings
            </h2>
            <div className="w-16 h-1 bg-[#1f6175] rounded-full"></div>
          </motion.div>

          {/* Avatar with edit icon */}
          <div className="relative w-28 h-28 mx-auto group">
            <img
              src={
                profile.avatar ||
                `https://ui-avatars.com/api/?name=${encodeURIComponent(profile.name || "User")}&background=4f46e5&color=fff`
              }
              alt="avatar"
              className="w-full h-full rounded-full border-4 border-blue-500 object-cover shadow-lg group-hover:border-blue-600 transition-all duration-300"
            />
            <button
              className="absolute bottom-0 right-0 p-2 bg-[#0e4f63] hover:bg-gray-900 rounded-full shadow-lg  transition-all duration-300 group-hover:scale-110"
              onClick={() => fileInputRef.current.click()}
              title="Edit Photo"
            >
              <Pencil size={18} className="text-white" />
            </button>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              ref={fileInputRef}
              style={{ display: "none" }}
            />
          </div>

          <div className="space-y-6 mt-6">
            {/* Name */}
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Name</label>
              <div className="relative">
                <input
                  type="text"
                  name="name"
                  value={profile.name}
                  onChange={handleChange}
                  className="w-full p-3 mt-1 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Enter your name"
                />
                {profile.name && (
                  <Check className="absolute right-3 top-1/2 transform -translate-y-1/2 text-green-500" size={18} />
                )}
              </div>
            </div>

            {/* Email */}
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Email</label>
              <div className="relative">
                <input
                  type="email"
                  name="email"
                  value={profile.email}
                  onChange={handleChange}
                  className="w-full p-3 mt-1 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Enter your email"
                />
                {profile.email && (
                  <Check className="absolute right-3 top-1/2 transform -translate-y-1/2 text-green-500" size={18} />
                )}
              </div>
            </div>

            {/* Theme Selector */}
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Theme</label>
              <div className="relative">
                <button
                  onClick={() => setIsThemeDropdownOpen(!isThemeDropdownOpen)}
                  className="w-full p-3 mt-1 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white flex justify-between items-center focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                >
                  <span>{themes.find(t => t.value === profile.theme)?.name || 'Select Theme'}</span>
                  <ChevronDown className={`transition-transform duration-200 ${isThemeDropdownOpen ? 'rotate-180' : ''}`} size={18} />
                </button>
                {isThemeDropdownOpen && (
                  <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 rounded-lg shadow-lg border border-gray-200 dark:border-gray-600 overflow-hidden">
                    {themes.map((theme) => (
                      <div
                        key={theme.value}
                        onClick={() => {
                          setProfile({ ...profile, theme: theme.value });
                          setIsThemeDropdownOpen(false);
                        }}
                        className={`px-4 py-2 hover:bg-blue-50 dark:hover:bg-gray-600 cursor-pointer flex items-center justify-between ${profile.theme === theme.value ? 'bg-blue-100 dark:bg-gray-600' : ''}`}
                      >
                        <span>{theme.name}</span>
                        {profile.theme === theme.value && <Check size={16} className="text-[#0e4f63] hover:bg-gray-900 dark:text-gray-300" />}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Old Password */}
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Current Password</label>
              <div className="relative">
                <input
                  type={showOldPassword ? "text" : "password"}
                  name="oldPassword"
                  value={profile.oldPassword}
                  onChange={handleChange}
                  className="w-full p-3 mt-1 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all pr-10"
                  placeholder="Enter current password"
                />
                <button
                  type="button"
                  onClick={() => setShowOldPassword(!showOldPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  {showOldPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">New Password</label>
              <div className="relative">
                <input
                  type={showNewPassword ? "text" : "password"}
                  name="password"
                  value={profile.password}
                  onChange={handleChange}
                  className="w-full p-3 mt-1 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-[#0e4f63] focus:border-transparent transition-all pr-10"
                  placeholder="Enter new password"
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  {showNewPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>

              {profile.password && (
                <div className="mt-2">
                  <div className="flex items-center space-x-2 mb-1">
                    <div className="text-sm font-medium text-gray-700 dark:text-gray-300">Strength:</div>
                    <div className={`text-sm font-medium ${strength === "Strong" ? "text-green-500" : strength === "Medium" ? "text-yellow-500" : "text-red-500"}`}>
                      {strength}
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${strengthColors[strength]} ${
                        strength === "Strong" ? "w-full" : strength === "Medium" ? "w-2/3" : "w-1/3"
                      } transition-all duration-500`}
                    ></div>
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    {strength === "Weak" && "Use at least 6 characters"}
                    {strength === "Medium" && "Add numbers or uppercase letters to strengthen"}
                    {strength === "Strong" && "Strong password!"}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Save Button */}
          <div className="flex justify-center mt-8">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSave}
              className="px-8 py-3 text-white bg-[#0e4f63] hover:bg-gray-900  rounded-full transition-all shadow-lg font-medium flex items-center space-x-2"
            >
              <span>Save Changes</span>
              <Pencil size={18} />
            </motion.button>
          </div>

          {/* Last Updated */}
          {lastUpdated && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-sm text-gray-500 dark:text-gray-400 text-center mt-6"
            >
              Last updated: <span className="font-medium">{lastUpdated}</span>
            </motion.p>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Profile;