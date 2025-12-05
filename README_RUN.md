# 🚀 How to Run the Application

## Quick Start (All Services)

Run everything with one command:
```bash
./start_all.sh
```

This will start:
- ✅ Virtual environment (Python)
- ✅ Backend (FastAPI on port 8000)
- ✅ Frontend (React/Vite on port 5173)

## Manual Start (Separate Terminals)

### Option 1: Using Scripts

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
```

### Option 2: Manual Commands

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to backend
cd backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Start server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

## Access Points

Once running:
- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## First Time Setup

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 2. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
```

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill process on port 5173 (frontend)
lsof -ti:5173 | xargs kill -9
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Frontend Dependencies Issues
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Backend Not Starting
- Check if Python 3.8+ is installed: `python3 --version`
- Verify virtual environment is activated
- Check if all dependencies are installed: `pip list`

### Frontend Not Starting
- Check if Node.js 18+ is installed: `node --version`
- Verify npm is installed: `npm --version`
- Clear cache: `npm cache clean --force`

## Development Workflow

1. **Start Backend** (Terminal 1)
   ```bash
   ./start_backend.sh
   ```

2. **Start Frontend** (Terminal 2)
   ```bash
   ./start_frontend.sh
   ```

3. **Make Changes**
   - Frontend: Changes auto-reload (Vite HMR)
   - Backend: Changes auto-reload (Uvicorn --reload)

4. **View Logs**
   - Backend: Check terminal output or `backend.log`
   - Frontend: Check terminal output or `frontend.log`

## Production Build

### Frontend
```bash
cd frontend
npm run build
# Output in dist/
```

### Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Environment Variables

### Frontend (.env in frontend/)
```
VITE_API_URL=http://localhost:8000
```

### Backend
No environment variables needed for basic setup.

