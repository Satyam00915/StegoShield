import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { useEffect, useState, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import toast from "react-hot-toast";
import {
  BookText,
  Brain,
  ShieldAlert,
  Eye,
  Code2,
  FileSearch,
  ArrowRight,
  Clock,
  ChevronRight,
  User,
  Lock,
  Network,
  Terminal,
  Shield,
  FileCode,
  Search,
  X,
  Bookmark,
  Share2,
  Heart,
  BookOpen,
  ArrowLeft,
  ArrowUp,
} from "lucide-react";
import BlogDetail from "../components/BlogDetail";

const blogsData = [
  // Fundamentals
  {
    id: 1,
    title: "What is Steganography?",
    summary: "A quick introduction to digital steganography and how attackers hide malicious payloads.",
    content: "Steganography is the practice of concealing a file, message, image, or video within another file, message, image, or video. Unlike cryptography, which conceals the contents of a secret message, steganography conceals the very existence of the message. This article explores the basic techniques used in digital steganography and how they're applied in modern cybersecurity threats.",
    date: "April 2, 2025",
    icon: <BookText className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "5 min read",
    category: "Fundamentals",
    author: "Dr. Sarah Chen",
    authorRole: "Security Researcher",
    likes: 42,
    bookmarks: 18,
    views: 256,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 2,
    title: "History of Steganography",
    summary: "From ancient techniques to modern digital implementations.",
    content: "The history of steganography dates back to ancient Greece, where messages were hidden on wax tablets or by tattooing them on slaves' heads under grown-out hair. In World War II, the Germans used microdots to hide information. Today, digital steganography hides data in images, audio files, videos, and even network protocols.",
    date: "April 3, 2025",
    icon: <BookText className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "7 min read",
    category: "Fundamentals",
    author: "Prof. James Wilson",
    authorRole: "Cryptography Historian",
    likes: 38,
    bookmarks: 12,
    views: 198,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 3,
    title: "Basic Steganography Techniques",
    summary: "Learn the fundamental methods used to hide data in digital files.",
    content: "This article covers the most common steganography techniques including LSB (Least Significant Bit) substitution, transform domain techniques, and spread spectrum methods. Each technique is explained with practical examples and visual demonstrations to help you understand how data can be concealed within various file types.",
    date: "April 4, 2025",
    icon: <BookText className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "6 min read",
    category: "Fundamentals",
    author: "Alex Morgan",
    authorRole: "Security Analyst",
    likes: 56,
    bookmarks: 22,
    views: 312,
    isBookmarked: false,
    isLiked: false
  },

  // AI Research
  {
    id: 4,
    title: "Detecting Stego Files with AI",
    summary: "How deep learning models can detect hidden data in images and audio.",
    content: "Artificial intelligence has revolutionized steganalysis by enabling the detection of subtle statistical anomalies that human analysts might miss. Convolutional Neural Networks (CNNs) can analyze thousands of image features simultaneously, looking for patterns that indicate hidden data.",
    date: "April 5, 2025",
    icon: <Brain className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "8 min read",
    category: "AI Research",
    author: "Dr. Emily Zhang",
    authorRole: "AI Specialist",
    likes: 89,
    bookmarks: 45,
    views: 521,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 5,
    title: "Neural Networks for Audio Steganalysis",
    summary: "Applying RNNs to detect hidden data in audio files.",
    content: "Recurrent Neural Networks (RNNs) are particularly effective for analyzing temporal patterns in audio files that may indicate steganographic content. This research paper presents our findings on using LSTM networks to achieve 98.7% detection accuracy on common audio steganography techniques.",
    date: "April 6, 2025",
    icon: <Brain className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "10 min read",
    category: "AI Research",
    author: "Dr. Raj Patel",
    authorRole: "AI Researcher",
    likes: 76,
    bookmarks: 38,
    views: 487,
    isBookmarked: false,
    isLiked: false
  },

  // Case Studies
  {
    id: 6,
    title: "Real-World Stego Attacks",
    summary: "Cases where steganography was used for malicious purposes.",
    content: "From malware distribution to corporate espionage, steganography has been used in numerous high-profile attacks. One notable case involved hackers embedding command-and-control instructions in Instagram images. Another used audio files on YouTube to exfiltrate data.",
    date: "April 7, 2025",
    icon: <ShieldAlert className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "6 min read",
    category: "Case Studies",
    author: "Mark Johnson",
    authorRole: "Threat Intelligence Analyst",
    likes: 65,
    bookmarks: 29,
    views: 402,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 7,
    title: "The StegoMalware Epidemic",
    summary: "How steganography is being used in modern malware campaigns.",
    content: "Recent malware campaigns have increasingly used steganography to hide malicious payloads in seemingly innocent files. This case study examines three major campaigns from the past year, showing their techniques and how they were eventually detected.",
    date: "April 8, 2025",
    icon: <ShieldAlert className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "9 min read",
    category: "Case Studies",
    author: "Lisa Wong",
    authorRole: "Malware Analyst",
    likes: 72,
    bookmarks: 31,
    views: 435,
    isBookmarked: false,
    isLiked: false
  },

  // Techniques
  {
    id: 8,
    title: "Understanding Visual Steganalysis",
    summary: "How to visually inspect images for stego content.",
    content: "Visual steganalysis techniques allow analysts to detect hidden data without complex algorithms. By applying specific filters and color adjustments, hidden patterns become visible. This guide walks through practical techniques using open-source tools.",
    date: "April 9, 2025",
    icon: <Eye className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "7 min read",
    category: "Techniques",
    author: "Lisa Rodriguez",
    authorRole: "Forensic Analyst",
    likes: 48,
    bookmarks: 19,
    views: 287,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 9,
    title: "Advanced Statistical Analysis Methods",
    summary: "Statistical techniques for detecting hidden data.",
    content: "This article dives deep into statistical methods for steganalysis, including chi-square analysis, RS analysis, and sample pair analysis. Each method is explained with mathematical foundations and practical implementation examples.",
    date: "April 10, 2025",
    icon: <Terminal className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "12 min read",
    category: "Techniques",
    author: "Dr. Michael Brown",
    authorRole: "Data Scientist",
    likes: 53,
    bookmarks: 24,
    views: 321,
    isBookmarked: false,
    isLiked: false
  },

  // Technology
  {
    id: 10,
    title: "Behind the StegoShield AI Engine",
    summary: "The models powering StegoShield's detection.",
    content: "Our proprietary detection system combines multiple AI architectures to achieve industry-leading accuracy. The CNN analyzes spatial patterns, the RNN processes sequential data in files, and the EfficientNet-LSTM hybrid model provides efficient large-scale scanning.",
    date: "April 11, 2025",
    icon: <Code2 className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "10 min read",
    category: "Technology",
    author: "StegoShield Team",
    authorRole: "Engineering",
    likes: 94,
    bookmarks: 47,
    views: 589,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 11,
    title: "Optimizing Detection Performance",
    summary: "How we achieve real-time steganalysis at scale.",
    content: "Processing millions of files per day requires careful optimization. This technical article explains our architecture decisions, including distributed computing, model quantization, and efficient feature extraction that make large-scale steganalysis possible.",
    date: "April 12, 2025",
    icon: <Network className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "11 min read",
    category: "Technology",
    author: "Sarah Kim",
    authorRole: "Systems Architect",
    likes: 81,
    bookmarks: 39,
    views: 512,
    isBookmarked: false,
    isLiked: false
  },

  // Analysis
  {
    id: 12,
    title: "Manual vs. Automated Detection",
    summary: "Comparing human analysis and AI-driven techniques.",
    content: "While AI has transformed steganalysis, human expertise remains valuable. This article compares both approaches across several dimensions: detection rates, false positives, resource requirements, and adaptability.",
    date: "April 13, 2025",
    icon: <FileSearch className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "9 min read",
    category: "Analysis",
    author: "Dr. Alan Turing",
    authorRole: "Senior Researcher",
    likes: 67,
    bookmarks: 28,
    views: 398,
    isBookmarked: false,
    isLiked: false
  },
  {
    id: 13,
    title: "The Future of Steganalysis",
    summary: "Emerging trends and future directions in detection.",
    content: "As steganography techniques evolve, so must detection methods. This forward-looking article examines promising research directions including quantum steganalysis, adversarial learning defenses, and explainable AI for forensic applications.",
    date: "April 14, 2025",
    icon: <FileCode className="w-6 h-6 text-[#0e4f63] dark:text-gray-400" />,
    readTime: "8 min read",
    category: "Analysis",
    author: "Dr. Priya Singh",
    authorRole: "Research Director",
    likes: 78,
    bookmarks: 36,
    views: 467,
    isBookmarked: false,
    isLiked: false
  },
  
];

const categories = ["All", ...new Set(blogsData.map(blog => blog.category))];

const Blog = () => {
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [filteredBlogs, setFilteredBlogs] = useState(blogsData);
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showScrollToTop, setShowScrollToTop] = useState(false);
  const [blogs, setBlogs] = useState(blogsData);
  const searchInputRef = useRef(null);

  useEffect(() => {
    if (!localStorage.getItem("user")) {
      toast.error("You need to be logged in to access this page.");
      navigate("/login");
    }

    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowScrollToTop(true);
      } else {
        setShowScrollToTop(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [isLoggedIn]);

  useEffect(() => {
    let result = blogsData;
    
    if (selectedCategory !== "All") {
      result = result.filter(blog => blog.category === selectedCategory);
    }
    
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        blog =>
          blog.title.toLowerCase().includes(query) ||
          blog.summary.toLowerCase().includes(query) ||
          blog.content.toLowerCase().includes(query) ||
          blog.author.toLowerCase().includes(query)
      );
    }
    
    setFilteredBlogs(result);
  }, [selectedCategory, searchQuery]);

  const handleReadMore = (blog) => {
    // Increment views when opening a blog
    const updatedBlogs = blogs.map(b => 
      b.id === blog.id ? { ...b, views: b.views + 1 } : b
    );
    setBlogs(updatedBlogs);
    setFilteredBlogs(updatedBlogs.filter(b => 
      selectedCategory === "All" || b.category === selectedCategory
    ));
    
    setSelectedBlog(blog);
    setIsModalOpen(true);
    document.body.style.overflow = 'hidden';
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedBlog(null);
    document.body.style.overflow = 'auto';
  };

  const handleBookmark = (id, e) => {
    e.stopPropagation();
    const updatedBlogs = blogs.map(blog => {
      if (blog.id === id) {
        return {
          ...blog,
          isBookmarked: !blog.isBookmarked,
          bookmarks: blog.isBookmarked ? blog.bookmarks - 1 : blog.bookmarks + 1
        };
      }
      return blog;
    });
    setBlogs(updatedBlogs);
    setFilteredBlogs(updatedBlogs.filter(b => 
      selectedCategory === "All" || b.category === selectedCategory
    ));
    
    if (selectedBlog?.id === id) {
      setSelectedBlog(updatedBlogs.find(blog => blog.id === id));
    }
    
    toast.success(
      updatedBlogs.find(blog => blog.id === id).isBookmarked 
        ? "Article bookmarked!" 
        : "Bookmark removed"
    );
  };

  const handleLike = (id, e) => {
    e.stopPropagation();
    const updatedBlogs = blogs.map(blog => {
      if (blog.id === id) {
        return {
          ...blog,
          isLiked: !blog.isLiked,
          likes: blog.isLiked ? blog.likes - 1 : blog.likes + 1
        };
      }
      return blog;
    });
    setBlogs(updatedBlogs);
    setFilteredBlogs(updatedBlogs.filter(b => 
      selectedCategory === "All" || b.category === selectedCategory
    ));
    
    if (selectedBlog?.id === id) {
      setSelectedBlog(updatedBlogs.find(blog => blog.id === id));
    }
  };

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  const clearSearch = () => {
    setSearchQuery("");
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  };

  return (
    <div className="min-h-screen bg-blue-50 dark:bg-gray-900">
      <Header />
      <motion.div
        className="py-10 px-4 sm:px-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="max-w-7xl 2xl:max-w-[1800px] mx-auto">
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <span className="inline-block px-3 py-1 text-xs font-semibold text-[#0e4f63] dark:text-gray-400 bg-indigo-100 dark:bg-indigo-900/30 rounded-full mb-4">
              Latest Articles
            </span>
            <h2 className="text-5xl font-extrabold bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent mb-4">
              StegoShield Blog
            </h2>
            <p className="max-w-2xl mx-auto text-gray-600 dark:text-gray-400">
              Insights, research, and updates on steganography detection and
              cybersecurity trends.
            </p>
          </motion.div>

          {/* Search and Filter Section */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
            <div className="flex overflow-x-auto pb-2 scrollbar-hide w-full sm:w-auto">
              <div className="flex space-x-2">
                {categories.map((category) => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`px-4 py-2 text-sm font-medium rounded-full whitespace-nowrap transition-colors ${
                      category === selectedCategory
                        ? "bg-[#0e4f63] text-white"
                        : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>

            <div className="relative w-full sm:w-64">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search articles..."
              className="block w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-full bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#0e4f63] focus:border-transparent"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                onClick={clearSearch}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                <X className="h-5 w-5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
              </button>
            )}
          </div>
          </div>

          

          {/* Blog Grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {filteredBlogs.map((blog) => (
              <motion.div
                key={blog.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                whileHover={{ y: -5 }}
                className="group relative flex flex-col overflow-hidden rounded-xl shadow-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:shadow-xl dark:hover:shadow-2xl dark:hover:bg-gray-700"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-700 dark:to-gray-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <div className="relative p-6 flex flex-col justify-between h-full">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-[#0e4f63] dark:text-gray-400 rounded-full">
                        {blog.category}
                      </span>
                      <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
                        <Clock className="w-3 h-3 mr-1" />
                        {blog.readTime}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-3 bg-indigo-100 dark:bg-indigo-500/20 rounded-full shadow-inner">
                        {blog.icon}
                      </div>
                      <h3 className="text-lg font-bold text-gray-800 dark:text-white leading-tight">
                        {blog.title}
                      </h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                      {blog.summary}
                    </p>
                  </div>
                  <div className="flex items-center justify-between pt-4 border-t border-gray-100 dark:border-gray-700">
                    <div className="flex items-center space-x-3">
                      <button 
                        onClick={(e) => handleLike(blog.id, e)}
                        className="flex items-center text-xs text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                      >
                        <Heart 
                          className={`w-4 h-4 mr-1 ${blog.isLiked ? 'fill-red-500 text-red-500 dark:fill-red-400 dark:text-red-400' : ''}`} 
                        />
                        {blog.likes}
                      </button>
                      <button 
                        onClick={(e) => handleBookmark(blog.id, e)}
                        className="flex items-center text-xs text-gray-500 dark:text-gray-400 hover:text-[#0e4f63] dark:hover:text-gray-300 transition-colors"
                      >
                        <Bookmark 
                          className={`w-4 h-4 mr-1 ${blog.isBookmarked ? 'fill-[#0e4f63] text-[#0e4f63] dark:fill-gray-400 dark:text-gray-400' : ''}`} 
                        />
                        {blog.bookmarks}
                      </button>
                      <span className="flex items-center text-xs text-gray-500 dark:text-gray-400">
                        <Eye className="w-4 h-4 mr-1" />
                        {blog.views}
                      </span>
                    </div>
                    <button 
                      onClick={() => handleReadMore(blog)}
                      className="flex items-center text-sm font-medium text-[#0e4f63] dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300 transition-colors"
                    >
                      Read more
                      <ChevronRight className="w-4 h-4 ml-1" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {filteredBlogs.length === 0 && (
            <div className="text-center py-12">
              <h3 className="text-xl font-medium text-gray-600 dark:text-gray-400">
                No articles found matching your criteria
              </h3>
              <button
                onClick={() => {
                  setSelectedCategory("All");
                  setSearchQuery("");
                }}
                className="mt-4 text-indigo-600 dark:text-indigo-400 hover:underline"
              >
                View all articles
              </button>
            </div>
          )}

          {/* Newsletter Subscription */}
          <div className="mt-16 bg-gradient-to-r from-[#0e4f63] to-[#34737e] dark:from-gray-800 dark:to-gray-700 rounded-xl p-8 text-white">
            <div className="max-w-2xl mx-auto text-center">
              <h3 className="text-3xl font-bold mb-2">Stay Updated</h3>
              <p className="mb-6 text-lg">
                Subscribe to our newsletter for the latest articles, research, and security insights.
              </p>
              <div className="flex flex-col sm:flex-row gap-2 max-w-md mx-auto">
                <input
                  type="email"
                  placeholder="Your email address"
                  className="flex-grow px-4 py-2 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-white bg-blue-100 dark:bg-gray-300"
                />
                <button className="px-6 py-2 bg-blue-100 dark:bg-gray-300 text-[#0e4f63] font-semibold rounded-lg hover:bg-gray-100 transition-colors">
                  Subscribe
                </button>
              </div>
              <p className="text-xs opacity-70 mt-3">
                We respect your privacy. Unsubscribe at any time.
              </p>
            </div>
          </div>

          <motion.div
            className="mt-20 text-center"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.8 }}
          >
            <div className="mb-8">
              <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                Want to learn more about our technology?
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                Discover how StegoShield's advanced detection algorithms work to
                protect your digital assets.
              </p>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate("/how-it-works")}
              className="inline-flex items-center bg-gradient-to-r from-[#113742] to-[#34737e] dark:from-gray-800 dark:to-gray-700 text-white px-6 py-3 rounded-full font-semibold shadow-lg hover:shadow-xl transition-all duration-300 group"
            >
              Learn How It Works
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </motion.div>
        </div>
      </motion.div>

      {/* Scroll to Top Button */}
      {showScrollToTop && (
        <motion.button
          onClick={scrollToTop}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed bottom-6 right-6 p-3 bg-[#0e4f63] text-white rounded-full shadow-lg hover:bg-[#113742] transition-colors z-50"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <ArrowUp className="w-5 h-5" />
        </motion.button>
      )}

      {/* Blog Detail Modal */}
      <AnimatePresence>
        {isModalOpen && selectedBlog && (
          <BlogDetail 
            blog={selectedBlog} 
            onClose={closeModal} 
            onBookmark={handleBookmark}
            onLike={handleLike}
          />
        )}
      </AnimatePresence>

      <Footer />
    </div>
  );
};

export default Blog;