import { motion, AnimatePresence } from "framer-motion";
import { X, Clock } from "lucide-react";

const BlogDetail = ({ blog, onClose }) => {
  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 overflow-y-auto">
        {/* Background overlay */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.75 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-gray-500 dark:bg-gray-900"
          onClick={onClose}
        />

        {/* Modal content */}
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative w-full max-w-2xl bg-white dark:bg-gray-800 rounded-lg shadow-xl overflow-hidden"
        >
          <div className="absolute top-4 right-4">
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
            </button>
          </div>

          <div className="p-6 overflow-y-auto max-h-[80vh]">
            <div className="mb-6">
              <span className="inline-block px-3 py-1 text-xs font-semibold bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-gray-400 rounded-full mb-3">
                {blog.category}
              </span>
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                {blog.title}
              </h2>
              <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-4">
                <span>{blog.date}</span>
                <span className="mx-2">•</span>
                <span className="flex items-center">
                  <Clock className="w-4 h-4 mr-1" />
                  {blog.readTime}
                </span>
                <span className="mx-2">•</span>
                <span>By {blog.author}</span>
              </div>
            </div>

            <div className="flex items-start gap-4 mb-6">
              <div className="p-3 bg-indigo-100 dark:bg-indigo-500/20 rounded-full shadow-inner flex-shrink-0">
                {blog.icon}
              </div>
              <div>
                <p className="font-medium text-gray-800 dark:text-white">{blog.author}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">{blog.authorRole}</p>
              </div>
            </div>

            <div className="prose dark:prose-invert max-w-none">
              <p className="text-gray-700 dark:text-gray-300 mb-6">
                {blog.content}
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                For more technical details and implementation examples, check out our documentation or contact our research team.
              </p>
            </div>

            <div className="mt-8 pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-center">
              <button
                onClick={onClose}
                className="w-full sm:w-auto px-4 py-2 bg-[#0e4f63] hover:bg-[#093947] dark:bg-gray-700 dark:hover:bg-gray-600 text-white font-medium rounded-md shadow-sm transition-colors"
              >
                Close Article
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default BlogDetail;