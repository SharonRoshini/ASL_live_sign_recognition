import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react()
  ],
  resolve: {
    dedupe: ['react', 'react-dom']
  },
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'esbuild', // Use esbuild (default, faster, no extra dependency)
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'mediapipe-vendor': ['@mediapipe/hands']
        },
        // Preserve class names for MediaPipe to prevent minification issues
        format: 'es',
        generatedCode: {
          constBindings: false
        }
      },
      // Ensure MediaPipe is handled correctly
      external: []
    },
    // Increase chunk size warning limit for MediaPipe
    chunkSizeWarningLimit: 1000,
    // CommonJS options for MediaPipe and React compatibility
    commonjsOptions: {
      include: [/@mediapipe/, /react/, /react-dom/],
      transformMixedEsModules: true,
      requireReturnsDefault: 'auto'
    }
  },
  // Optimize dependencies for MediaPipe and React
  optimizeDeps: {
    include: ['react', 'react-dom', '@mediapipe/hands'],
    exclude: [],
    // Force pre-bundling to ensure React is properly resolved
    force: true
  }
})

