/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans:    ['DM Sans', 'sans-serif'],
        display: ['Bebas Neue', 'sans-serif'],
      },
      colors: {
        green: {
          DEFAULT: '#b8f53d',
          dark:    '#8ec22a',
        },
      },
    },
  },
  plugins: [],
}
