import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      // two pages: the briefing (index) and the live-model window (predict)
      input: {
        main: path.resolve(__dirname, 'index.html'),
        predict: path.resolve(__dirname, 'predict.html'),
      },
    },
  },
})
