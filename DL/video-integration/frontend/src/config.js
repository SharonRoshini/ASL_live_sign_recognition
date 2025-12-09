// DL/video-integration/frontend/src/config.js

// Decide which backend URL to use
const getApiBase = () => {
  // Local development
  if (import.meta.env.DEV) {
    return 'http://localhost:5001';
  }

  // Production (Render) - always use deployed backend
  return 'https://asl-live-sign-recognition.onrender.com';
};

export const API_BASE = getApiBase();

// Optional: log in dev only
if (import.meta.env.DEV) {
  console.log('[CONFIG] API Base URL:', API_BASE);
}
