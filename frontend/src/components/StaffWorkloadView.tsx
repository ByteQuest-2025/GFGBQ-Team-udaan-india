import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDashboardData } from '../lib/dashboardData';

interface ShiftData {
  shift: string;
  department: string;
  required: number;
  assigned: number;
  load: number;
  burnoutRisk: 'low' | 'medium' | 'high';
  priority?: 1 | 2 | 3;
}

const fallbackBurnoutTrendData = [
  { day: 'Mon', index: 7.2 },
  { day: 'Tue', index: 7.4 },
  { day: 'Wed', index: 7.1 },
  { day: 'Thu', index: 6.9 },
  { day: 'Fri', index: 7.3 },
  { day: 'Sat', index: 6.8 },
  { day: 'Sun', index: 6.8 },
];

const shiftData: ShiftData[] = [
  { shift: 'Night (23:00-07:00)', department: 'ICU', required: 8, assigned: 6, load: 9.1, burnoutRisk: 'high', priority: 1 },
  { shift: 'Day (07:00-15:00)', department: 'ICU', required: 12, assigned: 11, load: 8.2, burnoutRisk: 'high', priority: 2 },
  { shift: 'Evening (15:00-23:00)', department: 'Emergency', required: 15, assigned: 14, load: 7.9, burnoutRisk: 'high', priority: 3 },
  { shift: 'Day (07:00-15:00)', department: 'Emergency', required: 18, assigned: 18, load: 7.4, burnoutRisk: 'medium' },
  { shift: 'Evening (15:00-23:00)', department: 'ICU', required: 10, assigned: 10, load: 6.8, burnoutRisk: 'medium' },
  { shift: 'Night (23:00-07:00)', department: 'Emergency', required: 12, assigned: 12, load: 6.2, burnoutRisk: 'low' },
];

export function StaffWorkloadView() {
  const { data } = useDashboardData();

  const getBurnoutColor = (risk: ShiftData['burnoutRisk']) => {
    switch (risk) {
      case 'high':
        return 'bg-rose-100 text-rose-700';
      case 'medium':
        return 'bg-amber-100 text-amber-700';
      case 'low':
        return 'bg-green-100 text-green-700';
    }
  };

  const getLoadColor = (load: number) => {
    if (load >= 8) return 'text-rose-600';
    if (load >= 6.5) return 'text-amber-600';
    return 'text-green-600';
  };

  const understaffedCount = shiftData.filter(s => s.assigned < s.required).length;
  const highRiskCount = shiftData.filter(s => s.burnoutRisk === 'high').length;

  const burnoutTrendData =
    data?.staff?.burnoutTrend7d && data.staff.burnoutTrend7d.length > 0
      ? data.staff.burnoutTrend7d
          .filter((d) => typeof d.day === 'string')
          .map((d) => ({ day: d.day, index: typeof d.index === 'number' ? d.index : null }))
      : fallbackBurnoutTrendData;

  const avgLoadIndex = (() => {
    if (typeof data?.kpis?.staffLoadIndex === 'number') return data.kpis.staffLoadIndex;
    const nums = burnoutTrendData.map((d) => d.index).filter((v): v is number => typeof v === 'number');
    if (nums.length === 0) return null;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
  })();

  return (
    <div className="p-8 space-y-8">
      {/* Header Stats */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h1 className="text-slate-900 mb-6">Staff Workload Analysis</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-600 mb-1">Total Shifts</p>
            <p className="text-3xl text-slate-900">{shiftData.length}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Understaffed</p>
            <p className="text-3xl text-rose-600">{understaffedCount}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">High Burnout Risk</p>
            <p className="text-3xl text-amber-600">{highRiskCount}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Avg Load Index</p>
            <p className="text-3xl text-slate-900">{avgLoadIndex ? avgLoadIndex.toFixed(1) : '—'}</p>
            <p className="text-xs text-slate-500">out of 10</p>
          </div>
        </div>
      </div>

      {/* Burnout Trend */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-slate-900 mb-1">7-Day Burnout Trend</h2>
        <p className="text-sm text-slate-500 mb-6">Average staff load index across all departments</p>
        
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={burnoutTrendData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis 
              dataKey="day" 
              stroke="#64748b"
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="#64748b"
              style={{ fontSize: '12px' }}
              domain={[0, 10]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                fontSize: '12px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="index" 
              stroke="#0891b2" 
              strokeWidth={2}
              dot={{ fill: '#0891b2', r: 4 }}
              name="Load Index"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Critical Alert */}
      <div className="bg-rose-50 border-l-4 border-rose-500 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <div className="bg-rose-100 p-2 rounded-lg">
            <AlertCircle className="w-5 h-5 text-rose-600" />
          </div>
          <div className="flex-1">
            <h3 className="text-slate-900 mb-1">Critical Staffing Issue</h3>
            <p className="text-sm text-slate-700 mb-3">
              Night shift ICU is understaffed by 2 nurses. Load index at 9.1/10 (critical level).
            </p>
            <button className="px-4 py-2 bg-rose-600 text-white rounded-lg text-sm hover:bg-rose-700 transition-colors">
              Contact On-Call Staff
            </button>
          </div>
        </div>
      </div>

      {/* Shift Workload Heatmap */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h2 className="text-slate-900">Shift Load Heatmap</h2>
          <p className="text-sm text-slate-500 mt-1">Current staffing levels and burnout risk assessment</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Shift</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Department</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Required</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Assigned</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Gap</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Load Index</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Burnout Risk</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {shiftData.map((shift, index) => {
                const gap = shift.required - shift.assigned;

                return (
                  <tr key={index} className="hover:bg-slate-50 transition-colors">
                    <td className="py-4 px-6 text-sm">
                      <div className="flex items-center gap-2">
                        {shift.priority && (
                          <span className="px-2 py-0.5 bg-rose-100 text-rose-700 text-xs rounded-full">
                            P{shift.priority}
                          </span>
                        )}
                        <span className="text-slate-900">{shift.shift}</span>
                      </div>
                    </td>
                    <td className="py-4 px-6 text-sm text-slate-700">{shift.department}</td>
                    <td className="py-4 px-6 text-sm text-slate-700">{shift.required}</td>
                    <td className="py-4 px-6 text-sm">
                      <span className={gap > 0 ? 'text-rose-600' : 'text-slate-700'}>
                        {shift.assigned}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-sm">
                      {gap > 0 ? (
                        <span className="text-rose-600 flex items-center gap-1">
                          <TrendingDown className="w-4 h-4" />
                          -{gap}
                        </span>
                      ) : (
                        <span className="text-green-600">OK</span>
                      )}
                    </td>
                    <td className="py-4 px-6 text-sm">
                      <span className={getLoadColor(shift.load)}>
                        {shift.load.toFixed(1)}/10
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`text-xs px-3 py-1 rounded-full ${getBurnoutColor(shift.burnoutRisk)}`}>
                        {shift.burnoutRisk.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      {gap > 0 && (
                        <button className="text-sm text-cyan-700 hover:text-cyan-800 transition-colors">
                          Request staff
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-slate-900 mb-4">Recommended Actions</h2>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-3 bg-cyan-50 rounded-lg">
            <div className="w-6 h-6 bg-cyan-600 text-white rounded-full flex items-center justify-center text-sm flex-shrink-0">
              1
            </div>
            <div className="flex-1">
              <p className="text-sm text-slate-900">Contact on-call nurses for Night ICU shift (immediate)</p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-cyan-50 rounded-lg">
            <div className="w-6 h-6 bg-cyan-600 text-white rounded-full flex items-center justify-center text-sm flex-shrink-0">
              2
            </div>
            <div className="flex-1">
              <p className="text-sm text-slate-900">Schedule additional staff for Evening Emergency (next 4 hours)</p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-cyan-50 rounded-lg">
            <div className="w-6 h-6 bg-cyan-600 text-white rounded-full flex items-center justify-center text-sm flex-shrink-0">
              3
            </div>
            <div className="flex-1">
              <p className="text-sm text-slate-900">Review Day ICU workload distribution to reduce burnout risk</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}