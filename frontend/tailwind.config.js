/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};

module.exports = {
  darkMode: 'class', // 👈 IMPORTANT
  content: [ './src/**/*.{js,jsx,ts,tsx}', ],
  theme: {
    extend: {},
  },
  plugins: [],
};

