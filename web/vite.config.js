import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API calls to the SALLY backend (default port 5678).
const backend = process.env.SALLY_BACKEND || 'http://127.0.0.1:5678'
const api = ['/chat', '/conversations', '/memory', '/tools', '/help', '/settings', '/health', '/system']

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(api.map((path) => [path, backend]))
  }
})
