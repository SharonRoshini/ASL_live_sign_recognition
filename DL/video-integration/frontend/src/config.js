/**
 * API Configuration
 * Uses environment variables for API base URL
 * Falls back to localhost for local development
 */

// Vite uses import.meta.env for environment variables
// Variables must be prefixed with VITE_ to be exposed to the client
const getApiBase = () => {
  // Check for environment variable (set in Render or .env file)
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Fallback to localhost for local development
  if (import.meta.env.DEV) {
    return 'http://localhost:5001';
  }
  
  // Production fallback (if no env var set, use relative path)
  // This assumes backend is on same domain with /api prefix
  return '/api';
};

export const API_BASE = getApiBase();

// Log API base in development for debugging
if (import.meta.env.DEV) {
  console.log('[CONFIG] API Base URL:', API_BASE);
}
