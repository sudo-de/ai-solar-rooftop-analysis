'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, XMarkIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface AnalysisFormProps {
  onAnalyze: (data: any) => void;
  isAnalyzing: boolean;
}

export function AnalysisForm({ onAnalyze, isAnalyzing }: AnalysisFormProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [city, setCity] = useState('New Delhi');
  const [panelType, setPanelType] = useState('monocrystalline');

  const cities = [
    'New Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Hyderabad', 
    'Ahmedabad', 'Jaipur', 'Kolkata', 'Pune', 'Gurugram'
  ];

  const panelTypes = [
    { value: 'monocrystalline', label: 'Monocrystalline', efficiency: '22%' },
    { value: 'bifacial', label: 'Bifacial', efficiency: '24%' },
    { value: 'perovskite', label: 'Perovskite', efficiency: '26%' }
  ];

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validFiles = acceptedFiles.filter(file => {
      const isValidType = ['image/jpeg', 'image/jpg', 'image/png'].includes(file.type);
      const isValidSize = file.size <= 10 * 1024 * 1024; // 10MB
      
      if (!isValidType) {
        toast.error(`${file.name}: Invalid file type. Use PNG, JPG, or JPEG.`);
        return false;
      }
      if (!isValidSize) {
        toast.error(`${file.name}: File too large. Maximum 10MB.`);
        return false;
      }
      return true;
    });

    setFiles(prev => [...prev, ...validFiles].slice(0, 5)); // Max 5 files
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: true
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (files.length === 0) {
      toast.error('Please upload at least one image');
      return;
    }

    // Create FormData with actual files
    const formData = new FormData();
    
    // Add files to FormData
    files.forEach(file => {
      formData.append('files', file);
    });
    
    // Add other form data
    formData.append('cities', JSON.stringify([city]));
    formData.append('panel_types', JSON.stringify([panelType]));

    // Call the analysis function with FormData
    onAnalyze(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* File Upload */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Rooftop Images
        </label>
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-lg font-medium text-gray-900 mb-2">
            {isDragActive ? 'Drop images here' : 'Drag & drop images here'}
          </p>
          <p className="text-sm text-gray-500">
            or click to select files (PNG, JPG, JPEG, max 10MB each)
          </p>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-4 space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                    <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* City Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select City
        </label>
        <select
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="input-field"
        >
          {cities.map(cityName => (
            <option key={cityName} value={cityName}>
              {cityName}
            </option>
          ))}
        </select>
      </div>

      {/* Panel Type Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Panel Type
        </label>
        <div className="grid grid-cols-1 gap-3">
          {panelTypes.map(panel => (
            <label
              key={panel.value}
              className={`relative flex items-center p-4 border rounded-lg cursor-pointer transition-colors ${
                panelType === panel.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input
                type="radio"
                name="panelType"
                value={panel.value}
                checked={panelType === panel.value}
                onChange={(e) => setPanelType(e.target.value)}
                className="sr-only"
              />
              <div className="flex items-center justify-between w-full">
                <div>
                  <p className="font-medium text-gray-900">{panel.label}</p>
                  <p className="text-sm text-gray-500">Efficiency: {panel.efficiency}</p>
                </div>
                <div className={`w-4 h-4 rounded-full border-2 ${
                  panelType === panel.value
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-gray-300'
                }`}>
                  {panelType === panel.value && (
                    <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5" />
                  )}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isAnalyzing || files.length === 0}
        className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
      >
        {isAnalyzing ? (
          <>
            <div className="loading-spinner" />
            <span>Analyzing...</span>
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Start Analysis</span>
          </>
        )}
      </button>
    </form>
  );
}
