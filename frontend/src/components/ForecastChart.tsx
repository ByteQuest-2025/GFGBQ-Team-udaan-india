import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useDashboardData } from '../lib/dashboardData';

const fallbackData = [
  { day: 'Mon', predicted: 210, actual: 205 },
  { day: 'Tue', predicted: 225, actual: 230 },
  { day: 'Wed', predicted: 240, actual: 238 },
  { day: 'Thu', predicted: 235, actual: 232 },
  { day: 'Fri', predicted: 260, actual: 255 },
  { day: 'Sat', predicted: 248, actual: null },
  { day: 'Sun', predicted: 265, actual: null },
];

export function ForecastChart() {
  const { data } = useDashboardData();
  const chartData = data?.forecast7d?.length ? data.forecast7d : fallbackData;
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6">
      <div className="mb-6">
        <h2 className="text-slate-900 mb-1">7-Day Admission Forecast</h2>
        <p className="text-sm text-slate-500">Predicted vs actual patient admissions</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis 
            dataKey="day" 
            stroke="#64748b"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="#64748b"
            style={{ fontSize: '12px' }}
            label={{ value: 'Admissions', angle: -90, position: 'insideLeft', style: { fontSize: '12px', fill: '#64748b' } }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              fontSize: '12px'
            }}
          />
          <Legend 
            wrapperStyle={{ fontSize: '12px' }}
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#0891b2" 
            strokeWidth={2}
            dot={{ fill: '#0891b2', r: 4 }}
            name="Predicted"
          />
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#059669" 
            strokeWidth={2}
            dot={{ fill: '#059669', r: 4 }}
            name="Actual"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
