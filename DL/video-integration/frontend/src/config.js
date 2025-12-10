// DL/video-integration/frontend/src/config.js

// Decide which backend URL to use
const getApiBase = () => {
  // Check for environment variable first (set in Render dashboard)
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }

  // Local development
  if (import.meta.env.DEV) {
    return 'http://localhost:5001';
  }

  // Production (Render) - default backend URL
  return 'https://asl-live-sign-recognition.onrender.com';
};

export const API_BASE = getApiBase();

// Log in dev or if explicitly enabled
if (import.meta.env.DEV || import.meta.env.VITE_DEBUG) {
  console.log('[CONFIG] API Base URL:', API_BASE);
}
