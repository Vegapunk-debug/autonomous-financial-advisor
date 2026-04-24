module.exports = {
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        bg: { primary: '#000000', secondary: '#050505', card: 'rgba(15, 15, 15, 0.4)', hover: '#0a0a0a', border: '#111111' },
        accent: { red: '#c0392b', redLight: '#e74c3c', redGlow: 'rgba(192,57,43,0.15)' },
        txt: { primary: '#ffffff', secondary: '#a1a1a1', muted: '#636366' }
      }
    },
  },
  plugins: [],
}
