# Frontend Deployment Guide - Render.com

This guide explains how to deploy the Gesture2Globe frontend to Render.com (Free Tier).

## Prerequisites

1. GitHub account with your repository
2. Render.com account (free signup)
3. Backend URL (once backend is deployed)

## Deployment Steps

### Option 1: Using Render Dashboard (Recommended)

1. **Sign up/Login to Render**
   - Go to https://render.com
   - Sign up or log in with GitHub

2. **Create New Static Site**
   - Click "New +" → "Static Site"
   - Connect your GitHub repository
   - Select the repository containing this project

3. **Configure Build Settings**
   - **Name**: `gesture2globe-frontend` (or any name you prefer)
   - **Branch**: `main` (or your main branch)
   - **Root Directory**: `video-integration/frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`

4. **Set Environment Variables**
   - Click "Environment" tab
   - Add environment variable:
     - **Key**: `VITE_API_BASE_URL`
     - **Value**: Your backend URL (e.g., `https://your-backend.onrender.com`)
     - **Note**: For now, you can use `http://localhost:5001` for testing, then update after backend deployment

5. **Deploy**
   - Click "Create Static Site"
   - Render will build and deploy your frontend
   - Wait for build to complete (usually 2-5 minutes)

6. **Get Your Frontend URL**
   - Once deployed, you'll get a URL like: `https://gesture2globe-frontend.onrender.com`
   - This is your live frontend URL!

### Option 2: Using render.yaml (Infrastructure as Code)

1. **Ensure render.yaml exists** in your repository root (already created)

2. **Create Static Site via Dashboard**
   - Go to Render Dashboard
   - Click "New +" → "Static Site"
   - Connect repository
   - Render will automatically detect `render.yaml`

3. **Update Environment Variables**
   - After backend is deployed, update `VITE_API_BASE_URL` in Render dashboard
   - Or update it in `render.yaml` and redeploy

## Post-Deployment

### Update Backend URL

Once your backend is deployed:

1. Go to Render Dashboard → Your Static Site → Environment
2. Update `VITE_API_BASE_URL` to your backend URL
3. Click "Save Changes"
4. Render will automatically rebuild and redeploy

### Testing

1. Visit your frontend URL
2. Open browser console (F12)
3. Check for `[CONFIG] API Base URL:` log message
4. Verify it shows your backend URL (not localhost)

## Troubleshooting

### Build Fails

- **Error: npm install fails**
  - Check that `package.json` exists in `video-integration/frontend`
  - Verify Node.js version (Render uses Node 18+ by default)

- **Error: Build command fails**
  - Check build logs in Render dashboard
  - Verify `npm run build` works locally first

### API Not Connecting

- **CORS errors**
  - Backend needs to allow your frontend domain
  - Update backend CORS settings to include your Render frontend URL

- **API calls failing**
  - Verify `VITE_API_BASE_URL` is set correctly
  - Check browser console for errors
  - Ensure backend is running and accessible

### Environment Variables Not Working

- Vite requires `VITE_` prefix for environment variables
- Variables must be set in Render dashboard before build
- Rebuild after changing environment variables

## Local Development

For local development:

1. Copy `.env.example` to `.env.local`
2. Leave `VITE_API_BASE_URL` empty to use `http://localhost:5001`
3. Run `npm run dev` as usual

## Free Tier Limitations

- ✅ Unlimited bandwidth
- ✅ Automatic HTTPS
- ✅ Custom domain support
- ✅ Auto-deploy from Git
- ⚠️ Builds may take 2-5 minutes
- ⚠️ First build after inactivity may be slower

## Next Steps

After frontend is deployed:

1. Deploy backend to Render (Web Service)
2. Update `VITE_API_BASE_URL` in frontend environment variables
3. Test the full application
4. (Optional) Set up custom domain

## Support

- Render Docs: https://render.com/docs
- Render Support: https://render.com/support

---

**Your frontend will be live at**: `https://your-app-name.onrender.com`
