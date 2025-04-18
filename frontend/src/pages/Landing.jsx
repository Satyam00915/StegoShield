import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import HeroSection from "../components/HeroSection";
import Feature from "../components/Features";
import Customers from "../components/Customers";
import About from "../components/About";
import Contact from "../components/Contact";
import Fotter from "../components/Footer";
import "../components/landing.css";

const Landing = () => {
  const location = useLocation();

  useEffect(() => {
    // Scroll to section if URL contains hash (e.g. #features)
    if (location.hash) {
      const targetElement = document.querySelector(location.hash);
      if (targetElement) {
        setTimeout(() => {
          targetElement.scrollIntoView({ behavior: 'smooth' });
        }, 100); // delay to ensure DOM is fully rendered
      }
    }
  }, [location]);

  useEffect(() => {
    // Smooth scroll for internal links inside landing page
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }, []);

  return (
    <>
      <HeroSection />
      <Feature />
      <Customers />
      <About />
      <Contact />
      <Fotter />

      {/* Floating Action Button */}
      <div className="fixed bottom-8 right-8 z-50">
        <a 
          href="#hero" 
          className="w-14 h-14 bg-[#0e4f63] hover:bg-[#286577] text-white rounded-full flex items-center justify-center shadow-lg transition-all duration-300 transform hover:scale-110"
          aria-label="Back to top"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </a>
      </div>
    </>
  );
};

export default Landing;
