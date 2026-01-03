import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, TrendingUp, Activity, Heart, Baby, Car } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';
import { triggerAction } from '../lib/api';
import { toast } from 'sonner';
import { percentWidthClass } from '../lib/percentWidthClass';

type TimeRange = '24h' | '48h' | '7days';

const baseCaseTypeData = [
  { type: 'Respiratory', predicted: 89, icon: Activity },
  { type: 'Trauma', predicted: 42, icon: Car },
  { type: 'Cardiac', predicted: 67, icon: Heart },
  { type: 'Pediatric', predicted: 34, icon: Baby },
];

function getCaseTypeAccent(type: string) {
  switch (type) {
    case 'Respiratory':
      return { bg: 'bg-cyan-600/10', icon: 'text-cyan-700' };
    case 'Trauma':
      return { bg: 'bg-rose-600/10', icon: 'text-rose-700' };
    case 'Cardiac':
      return { bg: 'bg-amber-500/10', icon: 'text-amber-700' };
    case 'Pediatric':
      return { bg: 'bg-violet-500/10', icon: 'text-violet-700' };
    default:
      return { bg: 'bg-slate-200', icon: 'text-slate-700' };
  }
}

const basePeakHoursData = [
  { hour: '00-04', load: 3 },
  { hour: '04-08', load: 5 },
  { hour: '08-12', load: 8 },
  { hour: '12-16', load: 9 },
  { hour: '16-20', load: 10 },
  { hour: '20-24', load: 7 },
];

const fallbackForecast24h = [
  { time: '00:00', admissions: 8 },
  { time: '04:00', admissions: 6 },
  { time: '08:00', admissions: 12 },
  { time: '12:00', admissions: 15 },
  { time: '16:00', admissions: 18 },
  { time: '20:00', admissions: 14 },
  { time: '24:00', admissions: 10 },
];

const fallbackForecast48h = fallbackForecast24h.map((point) => ({
  ...point,
  admissions: Math.round(point.admissions * 1.6),
}));

const fallbackForecast7d = fallbackForecast24h.map((point) => ({
  ...point,
  admissions: Math.round(point.admissions * 3.2),
}));

export function EmergencyForecastView() {
  const [selectedRange, setSelectedRange] = useState<TimeRange>('24h');
  const { data } = useDashboardData();

  const baseForecast =
    data?.emergencyForecast24h && data.emergencyForecast24h.length > 0
      ? data.emergencyForecast24h
      : fallbackForecast24h;

  const forecastForRange = (() => {
    if (selectedRange === '48h') {
      return baseForecast.map((p) => ({ ...p, admissions: Math.round(p.admissions * 1.6) }));
    }
    if (selectedRange === '7days') {
      return baseForecast.map((p) => ({ ...p, admissions: Math.round(p.admissions * 3.2) }));
    }
    return baseForecast;
  })();

  const caseTypeData = baseCaseTypeData.map((c) => {
    if (selectedRange === '48h') {
      return { ...c, predicted: Math.round(c.predicted * 1.7) };
    }
    if (selectedRange === '7days') {
      return { ...c, predicted: Math.round(c.predicted * 4) };
    }
    return c;
  });

  const peakHoursData = basePeakHoursData.map((slot) => {
    if (selectedRange === '48h') {
      return { ...slot, load: Math.min(10, slot.load + 1) };
    }
    if (selectedRange === '7days') {
      return { ...slot, load: Math.min(10, slot.load + 2) };
    }
    return slot;
  });

  const rangeLabel = selectedRange === '24h' ? 'Next 24 hours' : selectedRange === '48h' ? 'Next 48 hours' : 'Next 7 days';

  return (
    <div className="p-8 space-y-8">
      {/* Header Section */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-slate-900 mb-2">Emergency Department Forecast</h1>
            <p className="text-sm text-slate-600">Predictive load analysis and surge planning</p>
          </div>
          
          <div className="flex gap-2">
            {(['24h', '48h', '7days'] as TimeRange[]).map((range) => (
              <button
                key={range}
                onClick={() => setSelectedRange(range)}
                className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                  selectedRange === range
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {range === '24h' ? '24 Hours' : range === '48h' ? '48 Hours' : '7 Days'}
              </button>
            ))}
          </div>
        </div>

        {/* Surge Probability */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-slate-700">Low Surge Risk</span>
            </div>
            <p className="text-xs text-slate-600">0-4 AM window</p>
          </div>
          
          <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
              <span className="text-sm text-slate-700">Medium Surge Risk</span>
            </div>
            <p className="text-xs text-slate-600">8-12 AM, 4-8 PM windows</p>
          </div>
          
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-rose-500 rounded-full"></div>
              <span className="text-sm text-slate-700">High Surge Risk</span>
            </div>
            <p className="text-xs text-slate-600">12-4 PM peak detected</p>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-8">
          {/* Hourly Load Forecast */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h2 className="text-slate-900 mb-1">Hourly Load Forecast</h2>
            <p className="text-xs text-slate-500 mb-3">{rangeLabel}</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={forecastForRange} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="time" 
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
                <Bar dataKey="admissions" radius={[6, 6, 0, 0]}>
                  {forecastForRange.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.admissions >= 15 ? '#f59e0b' : entry.admissions >= 10 ? '#0891b2' : '#10b981'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Peak Hours Heatmap */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h2 className="text-slate-900 mb-4">Peak Arrival Hours Analysis</h2>
            <div className="space-y-3">
              {peakHoursData.map((slot) => {
                const percentage = (slot.load / 10) * 100;
                const colorClass = slot.load >= 9 ? 'bg-rose-500' : slot.load >= 7 ? 'bg-amber-500' : 'bg-green-500';
                
                return (
                  <div key={slot.hour} className="flex items-center gap-4">
                    <span className="text-sm text-slate-700 w-20">{slot.hour}</span>
                    <div className="flex-1 bg-slate-100 rounded-full h-8 overflow-hidden">
                      <div
                        className={`h-full ${colorClass} transition-all flex items-center justify-end pr-3 ${percentWidthClass(percentage)}`}
                      >
                        <span className="text-xs text-white">{slot.load}/10</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-8">
          {/* Case Type Breakdown */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h2 className="text-slate-900 mb-4">Forecast by Case Type</h2>
            <p className="text-sm text-slate-500 mb-6">{rangeLabel}</p>
            
            <div className="space-y-4">
              {caseTypeData.map((caseType) => {
                const Icon = caseType.icon;
                const accent = getCaseTypeAccent(caseType.type);
                return (
                  <div key={caseType.type} className="p-4 border border-slate-200 rounded-lg hover:shadow-sm transition-shadow">
                    <div className="flex items-center gap-3 mb-3">
                      <div className={`p-2 rounded-lg ${accent.bg}`}>
                        <Icon className={`w-4 h-4 ${accent.icon}`} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm text-slate-900">{caseType.type}</h3>
                        <p className="text-xs text-slate-500">Expected cases</p>
                      </div>
                      <span className="text-2xl text-slate-900">{caseType.predicted}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-slate-600">
                      <TrendingUp className="w-3 h-3 text-amber-600" />
                      <span>+12% vs yesterday</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Preparedness Recommendations */}
          <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg border border-cyan-200 p-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="bg-cyan-600 p-2 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-slate-900 mb-1">Preparedness Actions</h2>
                <p className="text-sm text-slate-600">Recommended interventions</p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="bg-white rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-sm text-slate-900">Increase Triage Staff</h3>
                  <span className="text-xs px-2 py-1 bg-rose-100 text-rose-700 rounded-full">High Priority</span>
                </div>
                <p className="text-xs text-slate-600 mb-3">Add 2 nurses to triage during 12-4 PM peak</p>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await triggerAction({
                        action_type: 'schedule_triage_staff',
                        source: 'emergency_forecast_view',
                        payload: {
                          range: selectedRange,
                          staff_count: 2,
                          window: '12-16',
                        },
                      });
                      toast.success('Action logged and escalation initiated');
                    } catch (e: any) {
                      toast.error(e?.message ?? 'Failed to log action');
                    }
                  }}
                  className="w-full py-2 bg-cyan-600 text-white text-sm rounded-lg hover:bg-cyan-700 transition-colors"
                >
                  Schedule Staff
                </button>
              </div>

              <div className="bg-white rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-sm text-slate-900">Prepare ICU Overflow</h3>
                  <span className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded-full">Medium Priority</span>
                </div>
                <p className="text-xs text-slate-600 mb-3">Ready 4 overflow beds in general ICU</p>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await triggerAction({
                        action_type: 'prepare_icu_overflow',
                        source: 'emergency_forecast_view',
                        payload: {
                          range: selectedRange,
                          overflow_beds: 4,
                        },
                      });
                      toast.success('Action logged and escalation initiated');
                    } catch (e: any) {
                      toast.error(e?.message ?? 'Failed to log action');
                    }
                  }}
                  className="w-full py-2 bg-white border border-slate-300 text-slate-700 text-sm rounded-lg hover:bg-slate-50 transition-colors"
                >
                  Configure Overflow
                </button>
              </div>

              <div className="bg-white rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-sm text-slate-900">Alert Ambulance Services</h3>
                  <span className="text-xs px-2 py-1 bg-slate-100 text-slate-700 rounded-full">Info</span>
                </div>
                <p className="text-xs text-slate-600 mb-3">Notify dispatch of expected delays</p>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await triggerAction({
                        action_type: 'notify_ambulance_services',
                        source: 'emergency_forecast_view',
                        payload: {
                          range: selectedRange,
                          note: 'expected ED delays',
                        },
                      });
                      toast.success('Action logged and escalation initiated');
                    } catch (e: any) {
                      toast.error(e?.message ?? 'Failed to log action');
                    }
                  }}
                  className="w-full py-2 bg-white border border-slate-300 text-slate-700 text-sm rounded-lg hover:bg-slate-50 transition-colors"
                >
                  Send Notification
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
