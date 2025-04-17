/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class', // 👈 Enables dark mode
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}', // scans all source files
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
