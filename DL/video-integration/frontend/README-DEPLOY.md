# Quick Deploy Guide - Render.com

## Environment Variables Setup

Create a `.env` file in `video-integration/frontend/` (or set in Render dashboard):

```
VITE_API_BASE_URL=https://your-backend-url.onrender.com
```

For local development, leave it empty to use `http://localhost:5001`

## Render Deployment Steps

1. Go to https://render.com → New Static Site
2. Connect your GitHub repo
3. Settings:
   - **Root Directory**: `video-integration/frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
4. Add Environment Variable:
   - Key: `VITE_API_BASE_URL`
   - Value: Your backend URL (or `http://localhost:5001` for now)
5. Deploy!

Your frontend will be live at: `https://your-app-name.onrender.com`
