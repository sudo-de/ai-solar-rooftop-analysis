# Vercel Configuration Guide

## Backend URL Configuration

The `vercel.json` file contains a rewrite rule that proxies API requests to your Docker backend.

### Update Backend URL

1. **Find your backend URL:**
   - If using Docker on a server: `http://your-server-ip:8000` or `https://api.your-domain.com`
   - If using a cloud service: Use the provided endpoint URL
   - If using a domain: `https://api.your-domain.com` or `https://backend.your-domain.com`

2. **Update vercel.json:**
   ```json
   {
     "rewrites": [
       {
         "source": "/api/(.*)",
         "destination": "https://your-actual-backend-url.com/api/$1"
       }
     ]
   }
   ```

3. **Environment Variables (Optional):**
   - In Vercel Dashboard → Settings → Environment Variables
   - Add `VITE_API_URL` with your backend URL
   - This will be used during build time

### Example Configurations

**Docker on DigitalOcean/AWS:**
```json
"destination": "https://api.yourdomain.com/api/$1"
```

**Docker on server with IP:**
```json
"destination": "http://123.456.789.0:8000/api/$1"
```

**Docker Hub deployed backend:**
```json
"destination": "https://your-docker-backend.railway.app/api/$1"
```

### Testing

After updating, test the API connection:
```bash
curl https://your-vercel-app.vercel.app/api/health
```

This should proxy to your backend and return the health check response.

