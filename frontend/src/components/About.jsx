import React from "react";
import { Shield, Code2, GitBranch, Database, Server, Globe, Rocket } from 'lucide-react';


const About = () => {
    return (
        <section className="bg-blue-50 dark:bg-gray-900 py-16 px-6 sm:px-12 md:px-24 text-center" id="about">
            <div className="max-w-4xl 2xl:max-w-[1800px] mx-auto">
                <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-400 mb-4">
                    ABOUT STEGOSHIELD
                </div>
                <h2 className="text-4xl font-extrabold text-gray-800 dark:text-[#0e4f63] mb-6">
                    Advanced Steganography Detection
                </h2>
                <p className="text-gray-600 dark:text-gray-300 mb-10 text-lg text-justify">
                    StegoShield is an advanced cybersecurity tool designed to detect hidden steganographic content inside images, audio, and video files. Using cutting-edge AI models, it ensures that malicious payloads embedded within media files are quickly identified and mitigated.
                </p>

                {/* Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
                    {[
                        { value: '99%', label: 'Detection Accuracy' },
                        { value: '500+', label: 'Daily Scans' },
                        { value: '10+', label: 'File Formats' },
                        { value: '24/7', label: 'Monitoring' }
                    ].map((stat, idx) => (
                        <div key={idx} className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                            <div className="text-3xl font-bold text-[#0e4f63] mb-2">{stat.value}</div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
                        </div>
                    ))}
                </div>

                {/* Technology stack */}
                <div className="mb-12">
                    <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-400 mb-4">Built With Modern Technology</h3>
                    <div className="flex flex-wrap justify-center gap-4">
                        {[
                            { icon: Shield, label: 'React.js' },
                            { icon: Code2, label: 'Tailwind CSS' },
                            { icon: GitBranch, label: 'Flask (Python)' },
                            { icon: Database, label: 'PostgreSQL' },
                            { icon: Server, label: 'Local Model Hosting' },
                            { icon: Globe, label: 'Render (Backend)' },
                            { icon: Rocket, label: 'Vercel (Frontend)' }
                        ].map(({ icon: Icon, label }, i) => (
                            <div key={i} className="flex items-center gap-2 px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-full">
                                <Icon className="w-4 h-4 text-[#0e4f63]" />
                                <span className="text-sm text-gray-800 dark:text-gray-300">{label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Developer info */}
                <div className="mt-12">
                    <h3 className="text-3xl font-bold text-gray-800 dark:text-gray-400 mb-4">Meet the Developer</h3>
                    <div className="max-w-3xl mx-auto bg-gray-50 dark:bg-gray-800 rounded-xl p-8 shadow-md relative overflow-hidden">
                        <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-indigo-100 dark:bg-indigo-400 opacity-20"></div>
                        <div className="absolute -bottom-10 -left-10 w-24 h-24 rounded-full bg-pink-100 dark:bg-pink-400 opacity-20"></div>
                        <div className="relative z-10">
                            <h4 className="text-xl font-semibold text-gray-700 dark:text-white mb-2">Sneha Sah</h4>
                            <p className="text-gray-600 dark:text-gray-300 text-justify">
                                Hi! I'm a 3rd-year B.Tech CSE student at IcfaiTech Hyderabad with a strong passion for cybersecurity and ethical hacking. I developed StegoShield as part of my final-year project to tackle the growing threat of steganography-based cyberattacks. I'm deeply invested in building secure systems that protect user data and believe in the power of AI to make that possible.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default About;
