import { motion } from "framer-motion";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { useAuth } from "../context/AuthContext";

const steps = [
  {
    step: "1. Upload File",
    description: "Upload an image, audio, or video file to begin the scan. Supported formats include JPG, PNG, MP3, WAV, MP4, and more.",
    icon: "📤",
    color: "from-blue-400 to-cyan-400",
    features: ["Drag & Drop Interface", "Batch Upload", "File Type Validation"]
  },
  {
    step: "2. AI-Based Detection",
    description: "Our advanced deep learning models analyze the file to detect steganographic payloads with 95%+ accuracy across multiple file types.",
    icon: "🤖",
    color: "from-purple-400 to-pink-400",
    features: ["Multi-model Analysis", "Real-time Processing", "Anomaly Detection"]
  },
  {
    step: "3. Get Results",
    description: "You'll receive a detailed visual report showing if the file is safe or suspicious, including confidence levels and detected patterns.",
    icon: "📊",
    color: "from-green-400 to-teal-400",
    features: ["Interactive Visualizations", "Detailed Threat Breakdown", "Exportable Reports"]
  },
  {
    step: "4. Take Action",
    description: "Mark, delete, or report suspicious files. Contribute to our growing database to help improve platform accuracy for everyone.",
    icon: "🛡️",
    color: "from-amber-400 to-orange-400",
    features: ["One-click Quarantine", "Community Reporting", "False Positive Feedback"]
  },
];

const testimonials = [
  {
    quote: "StegoShield detected hidden data in files that other tools missed. Essential for our security team.",
    author: "Alex K., Security Analyst",
    role: "Fortune 500 Company"
  },
  {
    quote: "The visual reports make it easy to explain risks to non-technical stakeholders.",
    author: "Maria S., IT Director",
    role: "Healthcare Provider"
  },
  {
    quote: "Saves us hours of manual inspection with its accurate AI detection.",
    author: "James T., Forensic Investigator",
    role: "Law Enforcement"
  }
];

const visualizationDetails = {
  title: "Advanced Threat Visualization",
  description: "Our visualization system provides real-time analysis of potential steganographic content with the following features:",
  features: [
    "Heatmap of suspicious areas in files",
    "Color-coded threat levels (Red = High, Yellow = Medium, Green = Low)",
    "Interactive zoom and pan capabilities",
    "Detailed metadata analysis",
    "Historical comparison with similar files"
  ],
  sampleData: [
    { x: 10, y: 15, value: 0.9, type: "LSB" },
    { x: 25, y: 30, value: 0.7, type: "DCT" },
    { x: 45, y: 20, value: 0.4, type: "EOF" },
    { x: 60, y: 50, value: 0.2, type: "None" }
  ]
};

  
const HowItWorks = () => {
  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();
  const [showDetails, setShowDetails] = useState(false);
  const [hoveredCell, setHoveredCell] = useState(null);
  const [showAboutModal, setShowAboutModal] = useState(false);

  useEffect(() => {
    if (!localStorage.getItem("user")) {
      toast.error("You need to be logged in to access this page.");
      navigate("/login");
    }
  }, [isLoggedIn]);

  const getThreatLevel = (value) => {
    if (value > 0.7) return { color: "bg-red-500", label: "High Risk" };
    if (value > 0.4) return { color: "bg-yellow-500", label: "Medium Risk" };
    return { color: "bg-green-500", label: "Low Risk" };
  };

  return (
    <div className="min-h-screen bg-blue-50 dark:bg-gray-900">
      <Header />

      <motion.div
        className="py-10 sm:py-10 px-4 sm:px-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-7xl mx-auto">
          {/* Hero Section */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16 px-4"
          >
            <span className="inline-block bg-blue-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-300 px-4 py-1.5 rounded-full text-sm font-medium mb-4 shadow-sm">
              Advanced Steganography Detection
            </span>
            <motion.h2
              className="text-4xl sm:text-5xl font-extrabold bg-gradient-to-r from-[#113742] to-[#8fbcc4] bg-clip-text text-transparent mb-4 leading-tight"
            >
              Uncover Hidden Threats with <br />AI-Powered Analysis
            </motion.h2>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              StegoShield combines cutting-edge deep learning with intuitive visualization to detect hidden data in your files with industry-leading accuracy.
            </p>
          </motion.div>

          {/* Steps Grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-24">
            {steps.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                className="group relative"
              >
                <div className={`absolute -inset-0.5 bg-gradient-to-br ${item.color} rounded-2xl opacity-75 blur transition-all duration-500 group-hover:opacity-100 group-hover:duration-200`} />
                <div className="relative bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-blue-50 dark:border-gray-700 h-full flex flex-col">
                  <div className={`text-4xl mb-4 w-14 h-14 rounded-full bg-gradient-to-br ${item.color} flex items-center justify-center shadow-md`}>
                    {item.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                    {item.step}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed mb-4 flex-grow">
                    {item.description}
                  </p>
                  <div className="space-y-2 mt-3">
                    {item.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center">
                        <svg className="w-4 h-4 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Enhanced Visualization Demo */}
          <motion.div 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="mb-24 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-3xl p-8 sm:p-12 shadow-inner"
          >
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-8">
                <h3 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-3">Interactive Threat Visualization</h3>
                <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                  See exactly where potential threats are detected with our heatmap visualization and detailed analysis reports.
                </p>
              </div>
              
              <div className="relative bg-white dark:bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-200 dark:border-gray-700 h-96">
                <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />
                <div className="relative z-10 h-full p-4 flex flex-col">
                  <div className="flex justify-between items-center mb-4">
                    <div className="inline-block bg-blue-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-300 px-3 py-1 rounded-full text-xs font-medium">
                      Sample Visualization
                    </div>
                    <button 
                      onClick={() => setShowDetails(true)}
                      className="text-sm font-medium text-[#0e4f63] dark:text-gray-400 hover:underline flex items-center"
                    >
                      Learn More
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                  
                  <div className="flex-grow relative">
                    {/* Heatmap Grid */}
                    <div className="absolute inset-0 grid grid-cols-10 grid-rows-10 gap-1 p-4">
                      {[...Array(100)].map((_, i) => {
                        const x = i % 10;
                        const y = Math.floor(i / 10);
                        const value = Math.random() * 0.8;
                        const threat = getThreatLevel(value);
                        const isHotspot = Math.random() > 0.85;
                        const hotspotValue = isHotspot ? Math.min(value + 0.3, 1) : value;
                        
                        return (
                          <motion.div
                            key={i}
                            className={`${getThreatLevel(hotspotValue).color} rounded cursor-pointer transition-all duration-300`}
                            initial={{ opacity: 0.7 }}
                            whileHover={{ 
                              opacity: 1,
                              scale: 1.2,
                              zIndex: 10,
                              boxShadow: "0 0 10px rgba(0,0,0,0.3)"
                            }}
                            onMouseEnter={() => setHoveredCell({
                              x: x * 10,
                              y: y * 10,
                              value: hotspotValue.toFixed(2),
                              threat: getThreatLevel(hotspotValue).label
                            })}
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{
                              opacity: 0.7 + (hotspotValue * 0.3),
                              animation: `pulse ${3 - (hotspotValue * 2)}s infinite`
                            }}
                          />
                        );
                      })}
                    </div>
                    
                    {/* Hover Tooltip */}
                    {hoveredCell && (
                      <motion.div
                        className="absolute bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                          left: `${hoveredCell.x}%`,
                          top: `${hoveredCell.y}%`
                        }}
                      >
                        <div className="text-sm font-medium text-gray-900 dark:text-white">
                          Threat Detected
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          Position: {hoveredCell.x/10}, {hoveredCell.y/10}
                        </div>
                        <div className="text-xs mt-1">
                          Confidence: <span className="font-bold">{hoveredCell.value}</span>
                        </div>
                        <div className="text-xs ">
                          Level: <span className={`font-bold ${
                            hoveredCell.threat === "High Risk" ? "text-red-500" :
                            hoveredCell.threat === "Medium Risk" ? "text-yellow-500" : "text-green-500"
                          }`}>
                            {hoveredCell.threat}
                          </span>
                        </div>
                      </motion.div>
                    )}
                    
                    {/* Legend */}
                    <div className="absolute bottom-4 left-4 right-4 bg-white dark:bg-gray-800 p-2 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex justify-center">
                      <div className="flex items-center space-x-4 text-xs">
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-red-500 rounded mr-1"></div>
                          <span>High Risk</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-yellow-500 rounded mr-1"></div>
                          <span>Medium Risk</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-green-500 rounded mr-1"></div>
                          <span>Low Risk</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-4 text-center">
                Hover over the heatmap to see detailed threat information
              </p>
            </div>
          </motion.div>

          {/* Testimonials */}
          <motion.div 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="mb-24"
          >
            <h3 className="text-2xl sm:text-3xl font-bold text-center text-gray-900 dark:text-white mb-12">Trusted by Security Professionals</h3>
            <div className="grid sm:grid-cols-3 gap-8">
              {testimonials.map((testimonial, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.2 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300 border border-gray-100 dark:border-gray-700"
                >
                  <div className="flex items-center mb-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-r from-[#113742] to-[#3a7d8c] dark:from-gray-800 dark:to-gray-700 flex items-center justify-center text-white font-bold">
                      {testimonial.author.charAt(0)}
                    </div>
                    <div className="ml-3">
                      <h4 className="font-medium text-gray-900 dark:text-white">{testimonial.author}</h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400">{testimonial.role}</p>
                    </div>
                  </div>
                  <p className="text-gray-600 dark:text-gray-300 italic">"{testimonial.quote}"</p>
                  <div className="mt-4 flex">
                    {[...Array(5)].map((_, i) => (
                      <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Final CTA */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-3xl p-8 sm:p-12 shadow-inner mb-8">
              <h3 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-4">Ready to Secure Your Digital Assets?</h3>
              <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-8">
                Join thousands of security professionals and organizations who trust StegoShield for advanced steganalysis.
              </p>
              <div className="flex flex-col sm:flex-row justify-center gap-4">
                <button
                  onClick={() => navigate("/dashboard")}
                  className="relative inline-flex items-center justify-center px-8 py-3.5 overflow-hidden font-medium text-white transition-all duration-300 bg-gradient-to-r from-[#113742] to-[#3a7d8c] dark:from-gray-800 dark:to-gray-700 rounded-full hover:shadow-xl group"
                >
                  <span className="relative z-10 flex items-center">
                    Start Free Scan
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2 transition-transform group-hover:translate-x-1" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </span>
                </button>
                <button
                  onClick={() => setShowAboutModal(true)}
                  className="relative inline-flex items-center justify-center px-8 py-3.5 overflow-hidden font-medium text-gray-900 dark:text-white transition-all duration-300 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full hover:shadow-lg group"
                >
                  <span className="relative z-10 flex items-center">
                    Learn About Us
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  </span>
                </button>

              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-6">
                No credit card required. Get started in seconds.
              </p>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Visualization Details Modal */}
      {showDetails && (
        <motion.div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setShowDetails(false)}
        >
          <motion.div 
            className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {visualizationDetails.title}
                </h3>
                <button 
                  onClick={() => setShowDetails(false)}
                  className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                >
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                {visualizationDetails.description}
              </p>
              
              <div className="space-y-4 mb-6">
                {visualizationDetails.features.map((feature, i) => (
                  <div key={i} className="flex items-start">
                    <div className="flex-shrink-0 mt-1">
                      <div className="w-5 h-5 bg-blue-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                    <p className="ml-3 text-gray-700 dark:text-gray-300">{feature}</p>
                  </div>
                ))}
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">Sample Detection Data</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
                    <thead className="bg-gray-100 dark:bg-gray-600">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">X</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Y</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Confidence</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Type</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-700 divide-y divide-gray-200 dark:divide-gray-600">
                      {visualizationDetails.sampleData.map((data, i) => (
                        <tr key={i}>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">{data.x}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">{data.y}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              data.value > 0.7 ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                              data.value > 0.4 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                              'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            }`}>
                              {data.value}
                            </span>
                          </td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">{data.type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              <div className="flex justify-center">
                <button
                  onClick={() => setShowDetails(false)}
                  className="px-4 py-2 bg-[#113742] text-white rounded-lg hover:bg-[#0e2e38] transition-colors"
                >
                  Got It!
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* About Us Modal */}
      {showAboutModal && (
        <motion.div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          onClick={() => setShowAboutModal(false)}
        >
          <motion.div 
            className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white rounded-xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
            initial={{ scale: 0.95, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-8 relative">
              {/* Close Button */}
              <button
                onClick={() => setShowAboutModal(false)}
                className="absolute top-4 right-4 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>

              {/* Header */}
              <div className="text-center mb-8">
                <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-[#0e4f63] to-[#8fbcc4] rounded-full flex items-center justify-center text-white text-3xl">
                  🛡️
                </div>
                <h2 className="text-3xl font-bold bg-gradient-to-r from-[#0e4f63] to-[#8fbcc4] bg-clip-text text-transparent">
                  About StegoShield
                </h2>
                <p className="text-lg text-gray-600 dark:text-gray-300 mt-2">
                  Your AI-powered defense against hidden digital threats
                </p>
              </div>

              {/* Content Sections */}
              <div className="space-y-6">
                <div className="bg-blue-50 dark:bg-gray-800 p-5 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 flex items-center">
                    <svg className="w-5 h-5 mr-2 text-[#0e4f63] dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    What is StegoShield?
                  </h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    StegoShield is an advanced security platform that uses deep learning to detect steganographic payloads hidden within image, audio, and video files. Our technology identifies malicious content that traditional security tools often miss.
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <h3 className="text-lg font-semibold mb-3 flex items-center">
                      <svg className="w-5 h-5 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Key Features
                    </h3>
                    <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Detects LSB, DCT, and other steganography techniques</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>95%+ detection accuracy across file types</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Interactive threat visualization</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Batch file processing</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Detailed analysis reports</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <h3 className="text-lg font-semibold mb-3 flex items-center">
                      <svg className="w-5 h-5 mr-2 text-[#093947] dark:text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                        <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                      </svg>
                      Our Technology
                    </h3>
                    <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="text-[#093947] dark:text-gray-300 mr-2">•</span>
                        <span>Deep neural networks trained on millions of samples</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-[#093947] dark:text-gray-300 mr-2">•</span>
                        <span>Proprietary ensemble detection algorithms</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-[#093947] dark:text-gray-300 mr-2">•</span>
                        <span>Continuous learning from user reports</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-[#093947] dark:text-gray-300 mr-2">•</span>
                        <span>Cloud-based and on-premise deployment options</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-[#093947] dark:text-gray-300 mr-2">•</span>
                        <span>Military-grade encryption for all processing</span>
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-gray-800 p-5 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 flex items-center">
                    <svg className="w-5 h-5 mr-2 text-[#0e4f63] dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    Why It Matters
                  </h3>
                  <p className="text-gray-700 dark:text-gray-300 mb-4">
                    Steganography is increasingly used by cybercriminals to bypass traditional security measures. StegoShield provides the critical detection layer needed in today's threat landscape.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-blue-100 dark:bg-gray-700 text-[#093947] dark:text-gray-300 rounded-full text-sm">Data Exfiltration</span>
                    <span className="px-3 py-1 bg-blue-100 dark:bg-gray-700 text-[#093947] dark:text-gray-300 rounded-full text-sm">Malware Delivery</span>
                    <span className="px-3 py-1 bg-blue-100 dark:bg-gray-700 text-[#093947] dark:text-gray-300 rounded-full text-sm">Corporate Espionage</span>
                    <span className="px-3 py-1 bg-blue-100 dark:bg-gray-700 text-[#093947] dark:text-gray-300 rounded-full text-sm">Covert Communication</span>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                  <h3 className="text-lg font-semibold mb-3">Supported File Formats</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    {['JPEG', 'PNG', 'BMP', 'GIF', 'MP3', 'WAV', 'MP4', 'AVI', 'PDF'].map((format) => (
                      <div key={format} className="flex items-center">
                        <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="text-gray-700 dark:text-gray-300">{format}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row justify-between items-center">
                <div className="mb-4 sm:mb-0">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Version 2.1.0 • Last updated April 2025
                  </p>
                </div>
                <button
                  onClick={() => setShowAboutModal(false)}
                  className="px-6 py-2 bg-gradient-to-r from-[#0e4f63] to-[#3a7d8c] hover:from-[#093947] hover:to-[#2c6a7a] text-white rounded-lg transition-all shadow-md hover:shadow-lg"
                >
                  Got It!
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      <style jsx global>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.7; }
          50% { opacity: 0.4; }
        }
      `}</style>

      <Footer />
    </div>
  );
};

export default HowItWorks;