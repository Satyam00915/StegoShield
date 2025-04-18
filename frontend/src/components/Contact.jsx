import React from 'react';
import { Mail, Phone, MapPin, Send } from 'lucide-react';

const Contact = () => {
    return (
        <section id="contact" className="pt-20 pb-10 px-6 md:px-20 bg-blue-50 dark:bg-gray-900 text-gray-800 dark:text-gray-100">
            <div className="max-w-6xl 2xl:max-w-[1800px] mx-auto">
                <div className="text-center mb-20">
                    <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-400 mb-4">
                        GET IN TOUCH
                    </div>
                    <h2 className="text-4xl font-extrabold mb-4 text-gray-800 dark:text-[#0e4f63] ">Contact Us</h2>
                    <p className="max-w-2xl mx-auto text-lg text-gray-700 dark:text-gray-300">
                        Have questions or need support? Our team is here to help you with any inquiries.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 gap-12">
                    <div className="space-y-6">
                        {/* Mail */}
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-lg bg-indigo-100 dark:bg-[#1f2937] text-[#0e4f63] dark:text-indigo-300">
                                <Mail className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-lg mb-1 dark:text-[#4b9ab3]">Email Us</h3>
                                <p className="text-gray-600 dark:text-gray-300">support@stegoshield.com</p>
                                <p className="text-gray-600 dark:text-gray-300">info@stegoshield.com</p>
                            </div>
                        </div>

                        {/* Phone */}
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-lg bg-indigo-100 dark:bg-[#1f2937] text-[#0e4f63] dark:text-indigo-300">
                                <Phone className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-lg mb-1 dark:text-[#4b9ab3]">Call Us</h3>
                                <p className="text-gray-600 dark:text-gray-300">+1 (555) 123-4567</p>
                                <p className="text-gray-600 dark:text-gray-300">Mon-Fri: 9am-5pm</p>
                            </div>
                        </div>

                        {/* Address */}
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-lg bg-indigo-100 dark:bg-[#1f2937] text-[#0e4f63] dark:text-indigo-300">
                                <MapPin className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-lg mb-1 dark:text-[#4b9ab3]">Visit Us</h3>
                                <p className="text-gray-600 dark:text-gray-300">123 Security Lane</p>
                                <p className="text-gray-600 dark:text-gray-300">Hyderabad, TS 500032</p>
                            </div>
                        </div>

                         {/* Social links */}
                         <div className="pt-4">
                            <h3 className="font-medium mb-3 dark:text-[#4b9ab3]">Follow Us</h3>
                            <div className="flex gap-4">
                                <a href="#" className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center hover:bg-indigo-100 dark:bg-[#1f2937] dark:hover:bg-[#0e191d] transition">
                                    <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z" />
                                    </svg>
                                </a>
                                <a href="#" className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center hover:bg-indigo-100 dark:bg-[#1f2937] dark:hover:bg-[#0e191d] transition">
                                    <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                                    </svg>
                                </a>
                                <a href="#" className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center hover:bg-indigo-100 dark:bg-[#1f2937] dark:hover:bg-[#0e191d] transition">
                                    <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                                    </svg>
                                </a>
                            </div>
                        </div>
                    </div>

                    {/* Form */}
                    <form 
                        className="bg-white dark:bg-[#1e293b] p-8 rounded-2xl shadow-lg space-y-6"
                        action="https://getform.io/f/bnlqzgjb"
                        method="POST"
                    >
                        <div>
                            <label className="block text-sm font-medium mb-1 dark:text-gray-200" htmlFor="name">Name</label>
                            <input
                                id="name"
                                name="name"
                                type="text"
                                placeholder="Your name"
                                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1 dark:text-gray-200" htmlFor="email">Email</label>
                            <input
                                id="email"
                                name="email"
                                type="email"
                                placeholder="you@example.com"
                                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1 dark:text-gray-200" htmlFor="message">Message</label>
                            <textarea
                                id="message"
                                name="message"
                                rows="5"
                                placeholder="Write your message..."
                                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                                required
                            ></textarea>
                        </div>
                        <button
                            type="submit"
                            className="inline-flex items-center justify-center gap-x-2 px-6 py-3 text-white bg-[#0e4f63] hover:bg-gray-800 dark:hover:bg-indigo-700 rounded-full text-sm font-medium transition-all shadow-md hover:shadow-lg w-full"
                        >
                            Send Message
                            <Send className="w-4 h-4" />
                        </button>
                    </form>
                </div>
            </div>
        </section>

    );
};

export default Contact;