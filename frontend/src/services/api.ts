import axios from 'axios'

// Use relative URL when in production (Docker), or absolute URL for development
const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

export const analyzeRooftop = async (files: File[]) => {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 minutes (300 seconds) - increased for NextGen AI processing
    onUploadProgress: (progressEvent) => {
      // Optional: can be used for upload progress tracking
      if (progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        console.log(`Upload progress: ${percentCompleted}%`)
      }
    },
  })

  return response.data
}

