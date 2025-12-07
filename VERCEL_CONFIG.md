# Vercel Configuration Guide

## Backend URL Configuration

The frontend uses environment variables to connect to your Docker backend. **No rewrite rules are needed** - the frontend will make direct API calls to your backend.

### Setup Backend URL

1. **Find your backend URL:**
   - If using Docker on a server: `http://your-server-ip:8000` or `https://api.your-domain.com`
   - If using a cloud service: Use the provided endpoint URL
   - If using a domain: `https://api.your-domain.com` or `https://backend.your-domain.com`

2. **Set Environment Variable in Vercel:**
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add a new variable:
     - **Name:** `VITE_API_URL`
     - **Value:** Your backend URL (e.g., `https://api.your-domain.com` or `http://123.456.789.0:8000`)
     - **Environment:** Production, Preview, Development (select all)
   - Click "Save"

3. **Redeploy:**
   - After adding the environment variable, trigger a new deployment
   - The frontend will be rebuilt with the correct backend URL

### Example Environment Variable Values

**Docker on DigitalOcean/AWS with domain:**
```
VITE_API_URL=https://api.yourdomain.com
```

**Docker on server with IP:**
```
VITE_API_URL=http://123.456.789.0:8000
```

**Docker Hub deployed backend (Railway, Render, etc.):**
```
VITE_API_URL=https://your-docker-backend.railway.app
```

**Important Notes:**
- Use `http://` for non-HTTPS backends (development/testing)
- Use `https://` for production backends
- Don't include trailing slashes
- The frontend will append `/api/` to this URL automatically

### Testing

After setting the environment variable and redeploying:

1. **Test the frontend:**
   - Visit your Vercel deployment URL
   - Open browser console (F12)
   - Try uploading an image
   - Check for network errors

2. **Test the backend directly:**
   ```bash
   curl https://your-backend-url.com/health
   ```

3. **Check environment variable:**
   - The frontend code uses: `import.meta.env.VITE_API_URL`
   - You can verify it's set correctly in the browser console

### Troubleshooting

**Network Error:**
- Verify `VITE_API_URL` is set in Vercel environment variables
- Make sure you redeployed after adding the variable
- Check that your backend is accessible from the internet
- Verify CORS is enabled on your backend (should allow your Vercel domain)

**CORS Errors:**
- Update your backend CORS settings to allow your Vercel domain
- In `backend/main.py`, update `allow_origins` to include your Vercel URL
