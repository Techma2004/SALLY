import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://localhost:8000',
      '/memory': 'http://localhost:8000',
      '/daily': 'http://localhost:8000',
      '/stats': 'http://localhost:8000'
    }
  }
})
