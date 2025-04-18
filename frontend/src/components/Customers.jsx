import React from "react";
import { Quote } from 'lucide-react';

const testimonials = [
  {
    name: "Neha Patel",
    role: "Security Engineer",
    review: "StegoShield is an innovative tool that helped me detect hidden payloads in image files with impressive accuracy. Loved the intuitive UI!",
    avatar: "https://randomuser.me/api/portraits/women/68.jpg",
    rating: 5
  },
  {
    name: "Karan Mehta",
    role: "Penetration Tester",
    review: "Finally a steganography detection tool that actually works and is easy to use. It's now a part of my go-to toolkit for CTFs and audits.",
    avatar: "https://randomuser.me/api/portraits/men/52.jpg",
    rating: 5
  },
  {
    name: "Ayesha Iqbal",
    role: "Cybersecurity Intern",
    review: "Used StegoShield during my internship and was amazed by how fast and accurate it was. It really helped me validate suspicious media files.",
    avatar: "https://randomuser.me/api/portraits/women/91.jpg",
    rating: 4
  },
];

const Customers = () => {
  return (
    <section className="bg-blue-50 dark:bg-gray-900 py-16 px-4" id="customers">
      <div className="max-w-screen-xl mx-auto text-center">
        <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-400 mb-4">
          TESTIMONIALS
        </div>
        <h2 className="text-4xl font-extrabold text-gray-800 dark:text-[#0e4f63] mb-4">
          Trusted by Security Professionals
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-12 max-w-3xl mx-auto text-lg">
          Feedback from cybersecurity professionals and researchers who've tried StegoShield.
        </p>

        <div className="grid gap-8 md:grid-cols-3">
          {testimonials.map((user, idx) => (
            <div
              key={idx}
              className="bg-white dark:bg-gray-800 shadow-md rounded-2xl p-6 text-left hover:shadow-xl transition-all transform hover:-translate-y-1"
            >
              <Quote className="w-8 h-8 text-indigo-100 dark:text-indigo-300 mb-4" />
              <div className="flex items-center gap-4 mb-4">
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div className="text-left">
                  <h4 className="font-semibold text-gray-800 dark:text-white">{user.name}</h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{user.role}</p>
                </div>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-4">"{user.review}"</p>
              <div className="flex items-center">
                {[...Array(5)].map((_, i) => (
                  <svg
                    key={i}
                    className={`w-4 h-4 ${i < user.rating ? 'text-yellow-400' : 'text-gray-300 dark:text-gray-600'}`}
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">{user.rating}/5</span>
              </div>
            </div>
          ))}
        </div>

        {/* Client logos section */}
        <div className="mt-16">
          <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-6">
            TRUSTED BY SECURITY COMPANIES
          </h3>
          <div className="flex flex-wrap justify-center gap-8 md:gap-16 items-center opacity-70">
            {/* ShieldSec */}
            <div className="flex items-center justify-center h-8">
              <svg width="120" height="40" viewBox="0 0 120 40" className="h-full w-auto">
                <rect x="10" y="5" width="30" height="30" rx="4" className="fill-indigo-400 dark:fill-indigo-300" opacity="0.2"/>
                <path d="M25 15L28 20L25 25" stroke="#6366F1" strokeWidth="2" fill="none"/>
                <text x="50" y="25" fontFamily="Arial" fontSize="14" fontWeight="bold" className="fill-gray-800 dark:fill-white">ShieldSec</text>
              </svg>
            </div>

            {/* DataGuard */}
            <div className="flex items-center justify-center h-8">
              <svg width="120" height="40" viewBox="0 0 120 40" className="h-full w-auto">
                <circle cx="20" cy="20" r="12" className="fill-emerald-400 dark:fill-emerald-300" opacity="0.2"/>
                <path d="M15 20H25M20 15V25" stroke="#10B981" strokeWidth="2"/>
                <text x="40" y="25" fontFamily="Arial" fontSize="14" fontWeight="bold" className="fill-gray-800 dark:fill-white">DataGuard</text>
              </svg>
            </div>

            {/* SecureNet */}
            <div className="flex items-center justify-center h-8">
              <svg width="120" height="40" viewBox="0 0 120 40" className="h-full w-auto">
                <polygon points="20,5 35,35 5,35" className="fill-amber-400 dark:fill-amber-300" opacity="0.2"/>
                <path d="M20 10V30M10 20H30" stroke="#F59E0B" strokeWidth="2"/>
                <text x="40" y="25" fontFamily="Arial" fontSize="14" fontWeight="bold" className="fill-gray-800 dark:fill-white">SecureNet</text>
              </svg>
            </div>

            {/* CyberLock */}
            <div className="flex items-center justify-center h-8">
              <svg width="120" height="40" viewBox="0 0 120 40" className="h-full w-auto">
                <rect x="5" y="10" width="30" height="20" rx="2" className="fill-red-400 dark:fill-red-300" opacity="0.2"/>
                <circle cx="20" cy="20" r="4" fill="#EF4444"/>
                <text x="40" y="25" fontFamily="Arial" fontSize="14" fontWeight="bold" className="fill-gray-800 dark:fill-white">CyberLock</text>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Customers;
