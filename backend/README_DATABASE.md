# Database Setup

This backend uses SQLite database to store analysis results.

## Database Schema

### AnalysisResult Table
Stores individual rooftop analysis results with:
- Roof analysis data (area, orientation, surface type, suitability)
- Energy predictions
- ROI estimations
- Accuracy metrics
- Recommendations
- Timestamps

### AnalysisSession Table
Stores analysis sessions (multiple files analyzed together):
- Session ID
- Total files
- Status (pending, processing, completed, failed)
- Timestamps

## Database Location

The SQLite database file is created at: `./solar_analysis.db` (in the backend directory)

## API Endpoints

### POST `/api/analyze`
Analyzes rooftop images and saves results to database. Returns:
- Analysis results
- Session ID
- Saved results with database IDs

### GET `/api/analyses`
Get all analysis results (with pagination):
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records to return (default: 100)

### GET `/api/analyses/{analysis_id}`
Get a specific analysis result by ID

### GET `/api/sessions`
Get all analysis sessions (with pagination)

### DELETE `/api/analyses/{analysis_id}`
Delete an analysis result by ID

## Initialization

The database is automatically initialized when the FastAPI server starts. Tables are created if they don't exist.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

The database will be created automatically on first run.

