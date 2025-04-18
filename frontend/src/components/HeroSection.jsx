import Header from "./Header";
import "./landing.css";
import { useState } from 'react';

const HeroSection = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen);
  };

  return (
    <div className="relative overflow-hidden">
      {/* Enhanced gradient background with animation */}
      <div className="absolute inset-0 -z-10 bg-blue-50 dark:bg-gray-900 animate-gradient"></div>

      <Header />

      <section id="hero">
        {/* Increased max-width for larger screens */}
        <div className="max-w-screen-xl 2xl:max-w-[1800px] mx-auto px-4 sm:px-6 md:px-8 py-16 sm:py-24 flex flex-col-reverse md:flex-row items-center gap-12 text-center md:text-left">
          <div className="w-full max-w-xl 2xl:max-w-2xl space-y-6">
            <a
              href="/login"
              className="inline-flex gap-x-2 items-center text-sm font-medium border border-indigo-200 px-3 py-1.5 dark:text-gray-400 rounded-full hover:bg-white dark:hover:bg-gray-800 transition hover:shadow-md group"
            >
              AI Powered Steganography Detection
              <svg 
                className="w-4 h-4 transition-transform group-hover:translate-x-1" 
                fill="currentColor" 
                viewBox="0 0 20 20"
              >
                <path d="M7.21 14.77a.75.75 0 01.02-1.06L11.17 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" />
              </svg>
            </a>
            
            <h1 className="text-4xl sm:text-5xl 2xl:text-6xl font-extrabold text-gray-800 leading-tight dark:text-gray-500">
              Protecting Your Data, <span className="text-[#0e4f63]">One Pixel at a Time</span>
            </h1>
            
            <p className="text-gray-600 text-lg 2xl:text-xl">
              StegoShield detects hidden payloads inside everyday media files — images, audio, and video — using cutting-edge AI.
            </p>
            
            <div className="flex flex-col sm:flex-row justify-center md:justify-start gap-3 sm:gap-4">
              <a
                href="/signup"
                className="inline-flex items-center justify-center gap-x-2 px-6 py-3 2xl:px-8 2xl:py-4 text-white bg-[#0e4f63] hover:bg-[#2c6070] rounded-full text-sm 2xl:text-base font-medium transition-all shadow-md hover:shadow-lg"
              >
                Get started
                <svg className="w-4 h-4 2xl:w-5 2xl:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
              <button
                onClick={toggleModal}
                className="inline-flex items-center justify-center gap-x-2 px-6 py-3 2xl:px-8 2xl:py-4 text-gray-700 hover:text-gray-900 rounded-full text-sm 2xl:text-base font-medium border border-gray-300 dark:text-gray-400 hover:border-gray-400 transition-all"
              >
                Learn more
              </button>
            </div>
            
            {/* Trust indicators */}
            <div className="pt-4 flex flex-col sm:flex-row items-center gap-4 text-sm 2xl:text-base text-gray-500">
              <div className="flex items-center gap-2">
                <div className="flex -space-x-2">
                  <img 
                    className="w-6 h-6 2xl:w-8 2xl:h-8 rounded-full border-2 border-white" 
                    src="https://randomuser.me/api/portraits/women/12.jpg" 
                    alt="User" 
                  />
                  <img 
                    className="w-6 h-6 2xl:w-8 2xl:h-8 rounded-full border-2 border-white" 
                    src="https://randomuser.me/api/portraits/men/43.jpg" 
                    alt="User" 
                  />
                  <img 
                    className="w-6 h-6 2xl:w-8 2xl:h-8 rounded-full border-2 border-white" 
                    src="https://randomuser.me/api/portraits/women/34.jpg" 
                    alt="User" 
                  />
                </div>
                <span>Trusted by 500+ users</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center text-yellow-400">
                  {[...Array(5)].map((_, i) => (
                    <svg key={i} className="w-4 h-4 2xl:w-5 2xl:h-5 fill-current" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                  ))}
                </div>
                <span>4.9/5 rating</span>
              </div>
            </div>
          </div>

          <div className="w-full md:w-1/2 flex justify-center relative">
            {/* Floating animation container */}
            <div className="relative animate-float">
              <img
                src="https://raw.githubusercontent.com/sidiDev/remote-assets/c86a7ae02ac188442548f510b5393c04140515d7/undraw_progressive_app_m-9-ms_oftfv5.svg"
                alt="Hero illustration"
                className="w-full max-w-md sm:max-w-xl 2xl:max-w-2xl relative z-10"
              />
              {/* Decorative elements */}
              <div className="absolute -top-6 -left-6 w-32 h-32 2xl:w-40 2xl:h-40 rounded-full bg-indigo-100 opacity-50"></div>
              <div className="absolute -bottom-6 -right-6 w-24 h-24 2xl:w-32 2xl:h-32 rounded-full bg-pink-100 opacity-50"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Modal Overlay - Also adjusted for larger screens */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 backdrop-blur-sm transition-opacity duration-300">
          <div 
            className="bg-gray-100 dark:bg-gray-800 rounded-xl max-w-2xl 2xl:max-w-4xl w-full p-6 md:p-8 max-h-[90vh] overflow-y-auto shadow-2xl transform transition-all duration-300 scale-95 hover:scale-100"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-2xl md:text-3xl 2xl:text-4xl font-bold text-gray-800 dark:text-[#e2e7e9d7] flex items-center gap-2">
                  <svg className="w-6 h-6 2xl:w-8 2xl:h-8 text-[#0e4f63] dark:text-[#b2d6e0a4]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9z" clipRule="evenodd" />
                  </svg>
                  About StegoShield
                </h2>
                <p className="text-[#0e4f63] dark:text-[#b2d6e0a4] text-sm 2xl:text-base mt-1">Advanced Steganography Detection</p>
              </div>
              <button 
                onClick={toggleModal}
                className="text-gray-400 hover:text-gray-700 hover:bg-gray-100 p-1 rounded-full transition"
                aria-label="Close modal"
              >
                <svg className="w-6 h-6 2xl:w-8 2xl:h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="space-y-6 text-gray-600 dark:text-gray-100 2xl:text-lg">
              <div className="bg-indigo-50 dark:bg-gray-700 p-4 rounded-lg border-l-4 border-[#0e4f63] dark:border-[#b2d6e0a4]">
                <p className="font-medium">
                  StegoShield is an advanced AI-powered platform designed to detect hidden payloads and steganographic content in digital media files. Our technology helps organizations protect against data leaks and cyber threats hidden within seemingly innocent files.
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-800 dark:text-gray-50 flex items-center gap-2 mb-3">
                    <svg className="w-5 h-5 2xl:w-6 2xl:h-6 text-[#0e4f63] dark:text-[#b2d6e0a4]" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" />
                    </svg>
                    Key Features
                  </h3>
                  <ul className="space-y-3">
                    {[
                      "Detects hidden data in images (JPEG, PNG, BMP)",
                      "Analyzes audio files (MP3, WAV) for concealed information",
                      "Scans video files (MP4, AVI) for embedded payloads",
                      "Provides detailed analysis reports with confidence scores",
                      "Enterprise-grade security with end-to-end encryption"
                    ].map((feature, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <svg className="w-5 h-5 2xl:w-6 2xl:h-6 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold text-gray-800 dark:text-gray-50 flex items-center gap-2 mb-3">
                    <svg className="w-5 h-5 2xl:w-6 2xl:h-6 text-[#0e4f63] dark:text-[#b2d6e0a4]" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                    </svg>
                    How It Works
                  </h3>
                  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                    <p className="mb-3">
                      Our proprietary algorithms use deep learning to analyze the statistical properties of files, identifying anomalies that indicate hidden data.
                    </p>
                    <div className="flex items-center gap-2 text-sm 2xl:text-base text-[#0e4f63] dark:text-[#b2d6e0a4]">
                      <svg className="w-4 h-4 2xl:w-5 2xl:h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z" clipRule="evenodd" />
                      </svg>
                      <span>Learn more in our documentation</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <button
                onClick={toggleModal}
                className="px-6 py-2 2xl:px-8 2xl:py-3 bg-[#0e4f63] dark:bg-[#527f8ba4] text-white rounded-full hover:bg-[#305966] transition flex-1 flex items-center justify-center gap-2 text-sm 2xl:text-base"
              >
                <svg className="w-5 h-5 2xl:w-6 2xl:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Got it!
              </button>
              <a
                href="/signup"
                className="px-6 py-2 2xl:px-8 2xl:py-3 border border-[#0e4f63] dark:border-[#527f8ba4] text-[#0e4f63] dark:text-[#aee6f5a4] rounded-full hover:bg-indigo-50 dark:hover:bg-gray-700 transition flex-1 text-center text-sm 2xl:text-base"
              >
                Try it now
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeroSection;