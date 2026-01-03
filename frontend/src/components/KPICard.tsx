import { TrendingUp, TrendingDown } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string;
  unit: string;
  trend: {
    direction: 'up' | 'down';
    value: string;
    label: string;
  };
  icon: React.ComponentType<{ className?: string }>;
  iconBgColor: string;
  iconColor: string;
  warning?: boolean;
}

export function KPICard({ 
  title, 
  value, 
  unit, 
  trend, 
  icon: Icon, 
  iconBgColor, 
  iconColor,
  warning 
}: KPICardProps) {
  const TrendIcon = trend.direction === 'up' ? TrendingUp : TrendingDown;
  const trendColor = warning 
    ? trend.direction === 'up' ? 'text-amber-600' : 'text-green-600'
    : trend.direction === 'up' ? 'text-green-600' : 'text-rose-600';

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className={`${iconBgColor} ${iconColor} p-3 rounded-lg`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className={`flex items-center gap-1 text-sm ${trendColor}`}>
          <TrendIcon className="w-4 h-4" />
          <span>{trend.value}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h3 className="text-sm text-slate-600">{title}</h3>
        <div className="flex items-baseline gap-2">
          <span className="text-slate-900 text-3xl">{value}</span>
          <span className="text-sm text-slate-500">{unit}</span>
        </div>
        <p className="text-xs text-slate-400">{trend.label}</p>
      </div>
    </div>
  );
}
