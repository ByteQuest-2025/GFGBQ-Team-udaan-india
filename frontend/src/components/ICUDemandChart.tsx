import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useDashboardData } from '../lib/dashboardData';

const fallbackData = [
  { time: '00:00', demand: 82, capacity: 96 },
  { time: '04:00', demand: 79, capacity: 96 },
  { time: '08:00', demand: 81, capacity: 96 },
  { time: '12:00', demand: 84, capacity: 96 },
  { time: '16:00', demand: 87, capacity: 96 },
  { time: '20:00', demand: 89, capacity: 96 },
  { time: '24:00', demand: 91, capacity: 96 },
];

export function ICUDemandChart() {
  const { data } = useDashboardData();
  const chartData = data?.icuProjection24h?.length ? data.icuProjection24h : fallbackData;
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6">
      <div className="mb-6">
        <h2 className="text-slate-900 mb-1">ICU Demand Projection</h2>
        <p className="text-sm text-slate-500">Next 24-hour occupancy forecast</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis 
            dataKey="time" 
            stroke="#64748b"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="#64748b"
            style={{ fontSize: '12px' }}
            domain={[0, 100]}
            label={{ value: 'Beds', angle: -90, position: 'insideLeft', style: { fontSize: '12px', fill: '#64748b' } }}
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
          <Area
            type="monotone"
            dataKey="demand"
            stroke="#f59e0b"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorDemand)"
            name="Projected Demand"
          />
          <Line 
            type="monotone" 
            dataKey="capacity" 
            stroke="#64748b" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Total Capacity"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
