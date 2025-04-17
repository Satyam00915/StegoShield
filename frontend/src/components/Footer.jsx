import React from "react";
import { Mail, Linkedin, Github } from "lucide-react";

const Footer = () => {
    return (
        <footer className="bg-blue-50 dark:bg-gray-900 text-[#0e4f63] dark:text-gray-300 px-10 py-10">
            <div className="max-w-7xl mx-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-10 border-b border-gray-700 pb-8 my-20">
                {/* Logo + Tagline */}
                <div>
                    <h2 className="text-2xl font-bold text-[#0e4f63] dark:text-gray-300 mb-2">StegoShield</h2>
                    <p className="text-sm text-[#0e4f63] dark:text-gray-300">
                        AI-powered steganalysis to uncover hidden threats in images, audio, and video.
                    </p>
                </div>

                {/* Navigation Links */}
                <div>
                    <h3 className="text-lg font-semibold text-[#0e4f63] dark:text-gray-300 mb-3">Quick Links</h3>
                    <ul className="space-y-2 text-sm">
                        <li><a href="/" className="hover:text-gray-500">Home</a></li>
                        <li><a href="#features" className="hover:text-gray-500">Features</a></li>
                        <li><a href="#customers" className="hover:text-gray-500">Customers</a></li>
                        <li><a href="#about" className="hover:text-gray-500">About</a></li>
                        <li><a href="#contact" className="hover:text-gray-500">Contact</a></li>
                    </ul>
                </div>

                {/* Developer Info */}
                <div>
                    <h3 className="text-lg font-semibold text-[#0e4f63] dark:text-gray-300 mb-3">Developer</h3>
                    <p className="text-sm text-[#0e4f63] dark:text-gray-300">
                        Sneha<br />
                        B.Tech CSE | Cybersecurity Enthusiast<br />
                        ICFAITech, Hyderabad
                    </p>
                </div>

                {/* Contact / Socials */}
                <div>
                    <h3 className="text-lg font-semibold text-[#0e4f63] dark:text-gray-300 mb-3">Contact</h3>
                    <ul className="space-y-2 text-sm text-[#0e4f63] dark:text-gray-300">
                        <li className="flex items-center gap-2">
                            <Mail size={16} />
                            <a href="mailto:itssneha45@gmail.com" className="hover:text-gray-500">
                                itssneha45@gmail.com
                            </a>
                        </li>
                        <li className="flex items-center gap-2">
                            <Linkedin size={16} />
                            <a
                                href="https://www.linkedin.com/in/sneha-sah-760b40250/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="hover:text-gray-500"
                            >
                                LinkedIn
                            </a>
                        </li>
                        <li className="flex items-center gap-2">
                            <Github size={16} />
                            <a
                                href="https://github.com/amyy45"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="hover:text-gray-500"
                            >
                                GitHub
                            </a>
                        </li>
                    </ul>
                </div>
            </div>

            {/* Bottom note */}
            <div className="text-center mt-6 text-sm text-[#0e4f63] dark:text-gray-300">
                © {new Date().getFullYear()} <span className="text-[#0e4f63] dark:text-gray-300font-medium">StegoShield</span>. All rights reserved.
            </div>
        </footer>
    );
};

export default Footer;
