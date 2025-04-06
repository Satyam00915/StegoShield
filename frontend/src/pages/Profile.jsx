import { useState, useEffect } from "react";
import { toast } from "sonner";
import { motion } from "framer-motion";
import Header from "../components/Header";

const getPasswordStrength = (password) => {
  if (!password) return "Weak";
  if (password.length >= 8 && /\d/.test(password) && /[A-Z]/.test(password)) return "Strong";
  if (password.length >= 6) return "Medium";
  return "Weak";
};

const Profile = () => {
  const [profile, setProfile] = useState({
    name: "",
    email: "",
    password: "",
    avatar: "", // base64 or URL
    theme: "light",
  });

  const [lastUpdated, setLastUpdated] = useState("");

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user")) || {
      name: "CyberNova",
      email: "cybernova@stegoshield.ai",
      avatar: "",
      theme: "light",
    };
    setProfile({ ...user, password: "" });
    setLastUpdated(localStorage.getItem("lastUpdated") || "");
  }, []);

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

  const handleSave = () => {
    if (!profile.name || !profile.email) {
      toast.error("Name and Email are required!");
      return;
    }

    const updatedProfile = { ...profile };
    localStorage.setItem("user", JSON.stringify(updatedProfile));
    const timestamp = new Date().toLocaleString();
    localStorage.setItem("lastUpdated", timestamp);
    setLastUpdated(timestamp);
    toast.success("Profile updated successfully!");
  };

  const strength = getPasswordStrength(profile.password);

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 py-10 px-4">
        <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-6">
          <motion.h2
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl font-bold text-gray-800 dark:text-white"
          >
            Profile Settings
          </motion.h2>

          <div className="space-y-4">
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

            {/* Password */}
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">New Password</label>
              <input
                type="password"
                name="password"
                value={profile.password}
                onChange={handleChange}
                className="w-full p-2 mt-1 rounded-md border border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white"
              />
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

            {/* Avatar Upload */}
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">Upload Profile Picture</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="w-full p-1 mt-1 text-sm text-gray-700 dark:text-gray-200"
              />
            </div>

            {/* Avatar Preview */}
            {profile.avatar && (
              <div className="mt-3">
                <img
                  src={profile.avatar}
                  alt="avatar"
                  className="w-24 h-24 rounded-full border-4 border-indigo-500 object-cover"
                />
              </div>
            )}

          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            className="px-6 py-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-500 transition"
          >
            Save Changes
          </button>

          {/* Last Updated */}
          {lastUpdated && (
            <p className="text-sm text-gray-500 dark:text-gray-400 text-right">
              Last updated on: {lastUpdated}
            </p>
          )}
        </div>
      </div>
    </>
  );
};

export default Profile;
