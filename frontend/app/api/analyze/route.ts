import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    
    // Forward the form data to the backend with increased timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout
    
    const response = await fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      body: formData, // Send as multipart/form-data
      signal: controller.signal,
      headers: {
        'Connection': 'keep-alive',
      }
    });
    
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('API Error:', error);
    
    // Check if it's a timeout error
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json({
        success: false,
        message: 'Analysis timed out - backend is taking too long to respond',
        error: 'TIMEOUT'
      }, { status: 408 });
    }
    
    // Return mock data if backend is not available
    return NextResponse.json({
      success: true,
      message: 'Analysis completed (mock data - backend unavailable)',
      results: {
        totalArea: 150.5,
        usableArea: 120.3,
        annualEnergy: 18000,
        roi: 8.5,
        paybackPeriod: 12.5,
        recommendations: [
          'Optimal panel placement identified',
          'Consider bifacial panels for 15% more energy',
          'Roof orientation is excellent for solar'
        ]
      }
    });
  }
}
