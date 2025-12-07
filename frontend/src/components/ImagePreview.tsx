import { useState, useEffect } from 'react'

interface ImagePreviewProps {
  files: File[]
  onRemove: (index: number) => void
}

const ImagePreview = ({ files, onRemove }: ImagePreviewProps) => {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [urls, setUrls] = useState<string[]>([])

  useEffect(() => {
    const objectUrls = files.map(file => URL.createObjectURL(file))
    
    // Use setTimeout to avoid synchronous setState in effect
    setTimeout(() => {
      setUrls(objectUrls)
    }, 0)

    return () => {
      objectUrls.forEach(url => URL.revokeObjectURL(url))
    }
  }, [files])

  if (files.length === 0) return null

  return (
    <>
      <div className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-white flex items-center">
            <span className="mr-2">🖼️</span>
            Image Preview
            <span className="ml-3 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
              {files.length}
            </span>
          </h3>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {files.map((file, index) => (
            <div
              key={index}
              className="relative group cursor-pointer transform hover:scale-105 transition-all duration-200"
              onClick={() => setSelectedIndex(index)}
            >
              <div className="aspect-square rounded-xl overflow-hidden border-2 border-gray-800 group-hover:border-blue-500 transition-all duration-200 shadow-md group-hover:shadow-xl bg-gradient-to-br from-gray-900 to-gray-800">
                <img
                  src={urls[index]}
                  alt={file.name}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove(index)
                }}
                className="absolute top-2 right-2 w-7 h-7 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-all duration-200 flex items-center justify-center hover:bg-red-600 hover:scale-110 shadow-lg"
                aria-label="Remove image"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/60 to-transparent text-white text-xs p-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <p className="truncate font-medium">{file.name}</p>
                <p className="text-gray-300 text-[10px]">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <div className="absolute top-2 left-2 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold opacity-0 group-hover:opacity-100 transition-opacity">
                {index + 1}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Full Screen Modal */}
      {selectedIndex !== null && (
        <div
          className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setSelectedIndex(null)}
        >
          <div className="relative max-w-5xl max-h-full">
            <img
              src={urls[selectedIndex]}
              alt={files[selectedIndex].name}
              className="max-w-full max-h-[90vh] rounded-2xl shadow-2xl"
            />
            <button
              onClick={() => setSelectedIndex(null)}
              className="absolute top-4 right-4 w-12 h-12 bg-white/20 backdrop-blur-md text-white rounded-full hover:bg-white/30 transition-all duration-200 flex items-center justify-center text-2xl font-bold shadow-lg hover:scale-110"
            >
              ×
            </button>
            <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-md text-white p-5 rounded-2xl">
              <p className="font-semibold text-lg mb-1">{files[selectedIndex].name}</p>
              <div className="flex items-center space-x-4 text-sm text-gray-300">
                <span>{(files[selectedIndex].size / 1024 / 1024).toFixed(2)} MB</span>
                <span>•</span>
                <span>Image {selectedIndex + 1} of {files.length}</span>
              </div>
            </div>
            {files.length > 1 && (
              <>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedIndex(prev => prev !== null && prev > 0 ? prev - 1 : files.length - 1)
                  }}
                  className="absolute left-4 top-1/2 -translate-y-1/2 w-12 h-12 bg-white/20 backdrop-blur-md text-white rounded-full hover:bg-white/30 transition-all duration-200 flex items-center justify-center shadow-lg"
                >
                  ←
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedIndex(prev => prev !== null && prev < files.length - 1 ? prev + 1 : 0)
                  }}
                  className="absolute right-4 top-1/2 -translate-y-1/2 w-12 h-12 bg-white/20 backdrop-blur-md text-white rounded-full hover:bg-white/30 transition-all duration-200 flex items-center justify-center shadow-lg"
                >
                  →
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default ImagePreview
