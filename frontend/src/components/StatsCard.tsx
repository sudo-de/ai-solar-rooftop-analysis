interface StatsCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: string
  gradient: string
  trend?: {
    value: number
    label: string
  }
}

const StatsCard = ({ title, value, subtitle, icon, gradient, trend }: StatsCardProps) => {
  return (
    <div className={`bg-gradient-to-br ${gradient} rounded-xl p-6 border border-opacity-20 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium opacity-90">{title}</span>
        <span className="text-3xl">{icon}</span>
      </div>
      <div className="text-3xl font-bold mb-1">{value}</div>
      {subtitle && (
        <div className="text-xs opacity-75 mb-2">{subtitle}</div>
      )}
      {trend && (
        <div className="flex items-center space-x-1 text-xs">
          <span className={trend.value >= 0 ? 'text-green-200' : 'text-red-200'}>
            {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
          <span className="opacity-75">{trend.label}</span>
        </div>
      )}
    </div>
  )
}

export default StatsCard

