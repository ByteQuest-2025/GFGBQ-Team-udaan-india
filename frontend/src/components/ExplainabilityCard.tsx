import { Lightbulb, Calendar, Thermometer, TrendingUp, Activity } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';

interface Factor {
  id: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  impact: 'high' | 'medium' | 'low';
}

const fallbackFactors: Factor[] = [
  {
    id: '1',
    icon: Calendar,
    label: 'Weekend + Holiday pattern detected',
    impact: 'high',
  },
  {
    id: '2',
    icon: Thermometer,
    label: 'Temperature drop to 12°C forecasted',
    impact: 'high',
  },
  {
    id: '3',
    icon: TrendingUp,
    label: 'Respiratory cases increasing 23%',
    impact: 'high',
  },
  {
    id: '4',
    icon: Activity,
    label: 'ICU occupancy trending upward',
    impact: 'medium',
  },
];

export function ExplainabilityCard() {
  const { data } = useDashboardData();
  const factors: Factor[] = (data?.explainability?.factors?.length
    ? data.explainability.factors.map((f) => ({
        id: f.id,
        icon: Lightbulb,
        label: f.label,
        impact: f.impact,
      }))
    : fallbackFactors) as Factor[];

  const getImpactColor = (impact: Factor['impact']) => {
    switch (impact) {
      case 'high':
        return 'bg-rose-100 text-rose-700';
      case 'medium':
        return 'bg-amber-100 text-amber-700';
      case 'low':
        return 'bg-slate-100 text-slate-700';
    }
  };

  return (
    <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg border border-cyan-200 p-6">
      <div className="flex items-start gap-3 mb-4">
        <div className="bg-cyan-600 p-2 rounded-lg">
          <Lightbulb className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-slate-900 mb-1">Why This Alert?</h2>
          <p className="text-sm text-slate-600">AI prediction factors</p>
        </div>
      </div>

      <div className="space-y-3">
        {factors.map((factor) => {
          const Icon = factor.icon;
          return (
            <div
              key={factor.id}
              className="bg-white rounded-lg p-3 flex items-start gap-3 hover:shadow-sm transition-shadow"
            >
              <Icon className="w-4 h-4 text-cyan-700 mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-700">{factor.label}</p>
              </div>
              <span className={`text-xs px-2 py-1 rounded-full ${getImpactColor(factor.impact)} whitespace-nowrap`}>
                {factor.impact}
              </span>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-cyan-200">
        <p className="text-xs text-slate-600">
          Model confidence: <span className="text-slate-900">94.2%</span>
        </p>
        <p className="text-xs text-slate-500 mt-1">
          Based on 3 years of historical data
        </p>
      </div>
    </div>
  );
}
