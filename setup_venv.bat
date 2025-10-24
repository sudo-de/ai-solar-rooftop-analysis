@echo off
REM Virtual Environment Setup Script for Solar Rooftop Analysis (Windows)

echo 🚀 Setting up Solar Rooftop Analysis Environment...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install core dependencies
echo 📚 Installing core dependencies...
pip install -r requirements.txt

REM Install development dependencies
echo 🛠️  Installing development dependencies...
pip install pytest black flake8 mypy

REM Create necessary directories
echo 📁 Creating project directories...
if not exist logs mkdir logs
if not exist outputs mkdir outputs
if not exist temp mkdir temp
if not exist samples mkdir samples
if not exist screenshots mkdir screenshots

REM Set up environment variables
echo 🔐 Setting up environment variables...
if not exist .env (
    echo # API Keys (replace with your actual keys) > .env
    echo OPENWEATHER_API_KEY=your_openweather_key_here >> .env
    echo OPENROUTER_API_KEY=your_openrouter_key_here >> .env
    echo SENTINEL_API_KEY=your_sentinel_key_here >> .env
    echo LANDSAT_API_KEY=your_landsat_key_here >> .env
    echo. >> .env
    echo # Database >> .env
    echo DATABASE_URL=sqlite:///solar_analysis.db >> .env
    echo. >> .env
    echo # Server Configuration >> .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=8000 >> .env
    echo DEBUG=True >> .env
    echo. >> .env
    echo # Advanced Features >> .env
    echo ENABLE_FEDERATED_LEARNING=True >> .env
    echo ENABLE_EDGE_AI=True >> .env
    echo ENABLE_BLOCKCHAIN=True >> .env
    echo ENABLE_AR_VISUALIZATION=True >> .env
    echo ✅ Created .env file with default configuration
) else (
    echo ⚠️  .env file already exists
)

echo ✅ Virtual environment setup complete!
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate
