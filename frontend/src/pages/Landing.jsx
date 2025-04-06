import React from 'react'
import HeroSection from "../components/HeroSection";
import Feature from "../components/Features";
import Customers from "../components/Customers";
import About from "../components/About";
import Contact from "../components/Contact";
import Fotter from "../components/Footer";
import "../components/landing.css";

const Landing = () => {
  return (
    <>
        <HeroSection />
        <Feature />
        <Customers />
        <About />
        <Contact />
        <Fotter />
    </>
  )
}

export default Landing