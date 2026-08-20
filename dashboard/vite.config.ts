/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    // The browser tests need a real browser. Playwright runs them.
    exclude: ['tests/e2e/**', '**/node_modules/**', '**/dist/**'],
  },
})
