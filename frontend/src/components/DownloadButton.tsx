interface DownloadButtonProps {
  data: any
  format: 'json' | 'csv' | 'pdf'
  filename?: string
}

const DownloadButton = ({ data, format, filename }: DownloadButtonProps) => {
  const handleDownload = () => {
    let content: string
    let mimeType: string
    let extension: string

    switch (format) {
      case 'json':
        content = JSON.stringify(data, null, 2)
        mimeType = 'application/json'
        extension = 'json'
        break
      case 'csv':
        // Simple CSV conversion
        const headers = Object.keys(data)
        const values = Object.values(data)
        content = headers.join(',') + '\n' + values.join(',')
        mimeType = 'text/csv'
        extension = 'csv'
        break
      case 'pdf':
        alert('PDF export coming soon!')
        return
      default:
        return
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `solar-analysis.${extension}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const icons = {
    json: '📄',
    csv: '📊',
    pdf: '📑'
  }

  const labels = {
    json: 'JSON',
    csv: 'CSV',
    pdf: 'PDF'
  }

  const colors = {
    json: 'from-blue-500 to-blue-600',
    csv: 'from-green-500 to-green-600',
    pdf: 'from-red-500 to-red-600'
  }

  return (
    <button
      onClick={handleDownload}
      className={`flex items-center space-x-2 px-5 py-2.5 bg-gradient-to-r ${colors[format]} text-white rounded-xl transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95 font-semibold text-sm`}
    >
      <span className="text-lg">{icons[format]}</span>
      <span>Download {labels[format]}</span>
    </button>
  )
}

export default DownloadButton
