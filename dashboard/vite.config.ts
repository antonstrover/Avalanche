/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.AVALANCHE_API_PORT ?? '8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        ws: true,
      },
      '/health': `http://127.0.0.1:${apiPort}`,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    // The browser tests need a real browser. Playwright runs them.
    exclude: ['tests/e2e/**', '**/node_modules/**', '**/dist/**'],
  },
})
