interface ComparisonChartProps {
  data: {
    label: string
    value: number
    color: string
  }[]
}

const ComparisonChart = ({ data }: ComparisonChartProps) => {
  const maxValue = Math.max(...data.map(d => d.value))

  return (
    <div className="space-y-4">
      {data.map((item, index) => (
        <div key={index} className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="font-medium text-gray-700">{item.label}</span>
            <span className="font-semibold text-gray-900">{item.value.toLocaleString()}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full ${item.color} rounded-full transition-all duration-1000 ease-out`}
              style={{ width: `${(item.value / maxValue) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export default ComparisonChart

