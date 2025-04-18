import React from "react";
import {
  ShieldCheck,
  Image,
  Mic,
  Video,
  Zap,
  BarChart2,
  Code,
  Lock,
} from "lucide-react";

const features = [
  {
    title: "Image Steganalysis",
    description: "Detect hidden payloads in PNG, JPEG, and BMP files using advanced CNN models.",
    icon: <Image className="w-8 h-8 text-[#0e4f63] dark:text-indigo-400" />,
    stats: "99.2% accuracy",
    color: "bg-indigo-50 dark:bg-indigo-900",
  },
  {
    title: "Audio Steganalysis",
    description: "Analyze spectrograms to identify steganography in MP3 and WAV audio formats.",
    icon: <Mic className="w-8 h-8 text-[#0e4f63] dark:text-indigo-400" />,
    stats: "98.7% accuracy",
    color: "bg-pink-50 dark:bg-pink-900",
  },
  {
    title: "Video Steganalysis",
    description: "Detect encoded information in MP4 and AVI files with hybrid CNN-LSTM models.",
    icon: <Video className="w-8 h-8 text-[#0e4f63] dark:text-indigo-400" />,
    stats: "97.9% accuracy",
    color: "bg-purple-50 dark:bg-purple-900",
  },
  {
    title: "Real-time Detection",
    description: "Get instant analysis reports and payload suspicion score with every upload.",
    icon: <ShieldCheck className="w-8 h-8 text-[#0e4f63] dark:text-indigo-400" />,
    stats: "<1s response",
    color: "bg-blue-50 dark:bg-blue-900",
  },
];

const Feature = () => {
  return (
    <section className="py-20 bg-blue-50 dark:bg-gray-900" id="features">
      <div className="max-w-screen-xl mx-auto px-4 text-center">
        <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 dark:bg-gray-800 text-[#0e4f63] dark:text-gray-400 mb-4">
          <Zap className="w-4 h-4 mr-1" /> POWERFUL FEATURES
        </div>
        <h2 className="text-4xl font-extrabold text-gray-900 dark:text-[#0e4f63] mb-6">
          Advanced Detection Capabilities
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-12 max-w-2xl mx-auto text-lg">
          StegoShield provides multi-model steganalysis using deep learning to uncover hidden data in your files.
        </p>

        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`${feature.color} p-6 rounded-2xl shadow hover:shadow-md transition-all group hover:-translate-y-1 dark:shadow-md`}
            >
              <div className="flex justify-center mb-4">
                <div className="p-3 rounded-lg bg-[#cdecf5] dark:bg-gray-800 shadow-sm group-hover:shadow-md">
                  {feature.icon}
                </div>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {feature.description}
              </p>
              <span className="text-xs font-medium text-[#0e4f63] dark:text-indigo-300">
                {feature.stats}
              </span>
            </div>
          ))}
        </div>

        {/* Additional feature highlights */}
        <div className="mt-16 grid md:grid-cols-3 gap-8">
          <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl">
            <div className="flex items-center gap-3 mb-3">
              <BarChart2 className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white">Detailed Analytics</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              Comprehensive reports with visualizations showing exactly where potential hidden data might be located.
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl">
            <div className="flex items-center gap-3 mb-3">
              <Code className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white">Developer API</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              Integrate our detection models directly into your applications with our robust API.
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl">
            <div className="flex items-center gap-3 mb-3">
              <Lock className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white">Enterprise Security</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              Military-grade encryption for all your file analysis with zero data retention policy.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Feature;
