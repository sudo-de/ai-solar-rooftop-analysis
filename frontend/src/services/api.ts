import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const analyzeRooftop = async (files: File[], segmentationMethod?: string) => {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })
  
  // Add segmentation method if provided
  if (segmentationMethod) {
    formData.append('segmentation_method', segmentationMethod)
  }

  const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 60 seconds
  })

  return response.data
}

