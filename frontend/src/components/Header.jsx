import { useEffect, useState } from "react";
import logo from "../assets/logo.png";
import "./landing.css";
import { useAuth } from "../context/AuthContext";

const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  // const [isLoggedIn, setIsLoggedIn] = useState(true);
  const { isLoggedIn, setIsLoggedIn } = useAuth()

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (!e.target.closest(".menu-btn")) setIsOpen(false);
    };
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  useEffect(() => {
    const user = localStorage.getItem("user");
    if (user) {
      setIsLoggedIn(true);
    } else {
      setIsLoggedIn(false);
    }
  }, [])

  const handleLogout = () => {
    setIsLoggedIn(false);
    localStorage.removeItem("user");
    history.push("/login");
    alert("Logged out successfully!");
  };

  const Brand = () => (
    <div className="flex items-center custom-container custom-padding justify-between py-4 md:block">
      <a href="/">
        <img src={logo} alt="StegoShield logo" className="w-32 h-auto" />
      </a>
      <div className="md:hidden">
        <button
          className="menu-btn text-gray-600 hover:text-gray-800"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="h-6 w-6">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16m-7 6h7" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );

  // Dynamic Navigation
  const navigation = isLoggedIn
    ? [
      { title: "Dashboard", path: "/dashboard" },
      { title: "Predict", path: "/upload" },
      { title: "Blog", path: "/blog" },
      { title: "How It Works", path: "/how-it-works" },
    ]
    : [
      { title: "Home", path: "/" },
      { title: "Features", path: "#features" },
      { title: "Customers", path: "#customers" },
      { title: "Contact", path: "#contact" },
    ];

  return (
    <header className="relative z-10">
      {/* Mobile Brand */}
      <div className={`md:hidden ${isOpen ? "mx-2 pb-5" : "hidden"}`}>
        <Brand />
      </div>

      <nav
        className={`pb-5 md:text-sm ${isOpen
          ? "absolute top-0 inset-x-0 bg-white shadow-lg rounded-xl border mx-2 mt-2 md:static md:shadow-none md:border-none md:mx-0"
          : ""
          }`}
      >
        <div className="max-w-screen-1xl mx-auto px-4 sm:px-6 lg:px-3 xl:px-4 md:flex md:items-center md:justify-between md:px-6 md:py-4">
          <Brand />

          {/* Nav Links */}
          <div
            className={`md:flex items-center space-y-6 md:space-y-0 md:space-x-6 ${isOpen
              ? "flex flex-col items-center justify-center mt-6"
              : "hidden md:block"
              }`}
          >
            <ul className="flex flex-col items-center space-y-6 md:flex-row md:space-y-0 md:space-x-6">
              {navigation.map((item, idx) => (
                <li key={idx}>
                  <a
                    href={item.path}
                    className="block text-gray-700 hover:text-gray-900 text-center"
                  >
                    {item.title}
                  </a>
                </li>
              ))}
            </ul>

            {/* Auth Links */}
            {!isLoggedIn ? (
              <a
                href="/signup"
                className="inline-flex items-center justify-center px-4 py-2 text-white bg-gray-800 hover:bg-gray-700 rounded-full text-sm mt-4 md:mt-0"
              >
                Sign Up
              </a>
            ) : (
              <div className="flex flex-col items-center gap-4 mt-4 md:flex-row md:gap-4 md:mt-0">
                <button
                  onClick={handleLogout}
                  className="inline-flex items-center justify-center px-4 py-2 text-white bg-gray-800 hover:bg-gray-700 rounded-full text-sm"
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
