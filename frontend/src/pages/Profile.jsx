import { useState, useEffect, useRef } from "react";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Pencil, Eye, EyeOff } from "lucide-react";
import Header from "../components/Header";
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
      const res = await fetch(`http://localhost:5000/api/user/${profile.id}`, {
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

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
      {/* Transparent Header */}
      <div className="bg-transparent shadow-none">
        <Header />
      </div>

      <div className="py-10 px-4">
        <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-6">
          <motion.h2
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl font-bold text-center text-gray-800 dark:text-white"
          >
            Profile Settings
          </motion.h2>

          {/* Avatar with edit icon */}
          <div className="relative w-24 h-24 mx-auto">
            <img
              src={
                profile.avatar ||
                `https://ui-avatars.com/api/?name=${encodeURIComponent(profile.name || "User")}&background=4f46e5&color=fff`
              }
              alt="avatar"
              className="w-24 h-24 rounded-full border-4 border-indigo-500 object-cover"
            />
            <button
              className="absolute bottom-0 right-0 p-1 bg-white rounded-full shadow hover:scale-110 transition"
              onClick={() => fileInputRef.current.click()}
              title="Edit Photo"
            >
              <Pencil size={18} className="text-indigo-600" />
            </button>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              ref={fileInputRef}
              style={{ display: "none" }}
            />
          </div>

          <div className="space-y-4 mt-4">
            {/* Name */}
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">Name</label>
              <input
                type="text"
                name="name"
                value={profile.name}
                onChange={handleChange}
                className="w-full p-2 mt-1 rounded-md border border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
              />
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">Email</label>
              <input
                type="email"
                name="email"
                value={profile.email}
                onChange={handleChange}
                className="w-full p-2 mt-1 rounded-md border border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
              />
            </div>

            {/* Old Password */}
            <div className="relative">
              <label className="block text-sm text-gray-600 dark:text-gray-300">Current Password</label>
              <input
                type={showOldPassword ? "text" : "password"}
                name="oldPassword"
                value={profile.oldPassword}
                onChange={handleChange}
                className="w-full p-2 pr-10 mt-1 rounded-md border border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
              />
              <button
                type="button"
                onClick={() => setShowOldPassword(!showOldPassword)}
                className="absolute top-9 right-3 text-gray-500 hover:text-gray-700 dark:hover:text-white"
              >
                {showOldPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>

            {/* New Password */}
            <div className="relative">
              <label className="block text-sm text-gray-600 dark:text-gray-300">New Password</label>
              <input
                type={showNewPassword ? "text" : "password"}
                name="password"
                value={profile.password}
                onChange={handleChange}
                className="w-full p-2 pr-10 mt-1 rounded-md border border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
              />
              <button
                type="button"
                onClick={() => setShowNewPassword(!showNewPassword)}
                className="absolute top-9 right-3 text-gray-500 hover:text-gray-700 dark:hover:text-white"
              >
                {showNewPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>

              {profile.password && (
                <p
                  className={`text-sm mt-1 ${
                    strength === "Strong"
                      ? "text-green-500"
                      : strength === "Medium"
                      ? "text-yellow-500"
                      : "text-red-500"
                  }`}
                >
                  Password Strength: {strength}
                </p>
              )}
            </div>
          </div>

          {/* Save Button */}
          <div className="flex justify-center">
            <button
              onClick={handleSave}
              className="px-6 py-2 text-white bg-gray-800 hover:bg-gray-700 rounded-full transition"
            >
              Save Changes
            </button>
          </div>

          {/* Last Updated */}
          {lastUpdated && (
            <p className="text-sm text-gray-500 dark:text-gray-400 text-right">
              Last updated on: {lastUpdated}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;
