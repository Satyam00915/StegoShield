import React from "react";
import { Mail, Linkedin, Github } from "lucide-react";

const Footer = () => {
    return (
        <footer className="bg-gray-900 text-gray-300 px-10 py-10">
            <div className="max-w-7xl mx-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-10 border-b border-gray-700 pb-8">
                {/* Logo + Tagline */}
                <div>
                    <h2 className="text-2xl font-bold text-white mb-2">StegoShield</h2>
                    <p className="text-sm text-gray-400">
                        AI-powered steganalysis to uncover hidden threats in images, audio, and video.
                    </p>
                </div>

                {/* Navigation Links */}
                <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Quick Links</h3>
                    <ul className="space-y-2 text-sm">
                        <li><a href="/" className="hover:text-white">Home</a></li>
                        <li><a href="#features" className="hover:text-white">Features</a></li>
                        <li><a href="#customers" className="hover:text-white">Customers</a></li>
                        <li><a href="#about" className="hover:text-white">About</a></li>
                        <li><a href="#contact" className="hover:text-white">Contact</a></li>
                    </ul>
                </div>

                {/* Developer Info */}
                <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Developer</h3>
                    <p className="text-sm text-gray-400">
                        Sneha<br />
                        B.Tech CSE | Cybersecurity Enthusiast<br />
                        ICFAITech, Hyderabad
                    </p>
                </div>

                {/* Contact / Socials */}
                <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Contact</h3>
                    <ul className="space-y-2 text-sm text-gray-400">
                        <li className="flex items-center gap-2">
                            <Mail size={16} />
                            <a href="mailto:itssneha45@gmail.com" className="hover:text-white">
                                itssneha45@gmail.com
                            </a>
                        </li>
                        <li className="flex items-center gap-2">
                            <Linkedin size={16} />
                            <a
                                href="https://www.linkedin.com/in/sneha-sah-760b40250/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="hover:text-white"
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
                                className="hover:text-white"
                            >
                                GitHub
                            </a>
                        </li>
                    </ul>
                </div>
            </div>

            {/* Bottom note */}
            <div className="text-center mt-6 text-sm text-gray-500">
                © {new Date().getFullYear()} <span className="text-white font-medium">StegoShield</span>. All rights reserved.
            </div>
        </footer>
    );
};

export default Footer;
