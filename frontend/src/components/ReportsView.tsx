import { useState } from 'react';
import { FileText, Download, Calendar, Filter, TrendingUp, TrendingDown } from 'lucide-react';

interface Report {
  id: string;
  title: string;
  description: string;
  date: string;
  type: 'weekly' | 'monthly' | 'custom';
}

const availableReports: Report[] = [
  {
    id: '1',
    title: 'Weekly Emergency Summary',
    description: 'Comprehensive overview of emergency department activity, admissions, and wait times',
    date: 'Dec 30 - Jan 5, 2026',
    type: 'weekly',
  },
  {
    id: '2',
    title: 'ICU Utilization Report',
    description: 'Detailed analysis of ICU occupancy, patient flow, and capacity management',
    date: 'December 2025',
    type: 'monthly',
  },
  {
    id: '3',
    title: 'Staff Workload & Burnout Analysis',
    description: 'Staff allocation, load distribution, and burnout risk assessment across departments',
    date: 'Q4 2025',
    type: 'monthly',
  },
  {
    id: '4',
    title: 'Prediction Accuracy Report',
    description: 'Model performance metrics, accuracy trends, and prediction reliability analysis',
    date: 'December 2025',
    type: 'monthly',
  },
];

const trendInsights = [
  {
    metric: 'Emergency Admissions',
    current: '1,842',
    previous: '1,654',
    change: '+11.4%',
    direction: 'up' as const,
    status: 'warning' as const,
  },
  {
    metric: 'ICU Occupancy Rate',
    current: '86.3%',
    previous: '82.1%',
    change: '+5.1%',
    direction: 'up' as const,
    status: 'warning' as const,
  },
  {
    metric: 'Average Wait Time',
    current: '28 min',
    previous: '35 min',
    change: '-20%',
    direction: 'down' as const,
    status: 'success' as const,
  },
  {
    metric: 'Staff Load Index',
    current: '6.8',
    previous: '7.2',
    change: '-5.6%',
    direction: 'down' as const,
    status: 'success' as const,
  },
];

export function ReportsView() {
  const [selectedDepartment, setSelectedDepartment] = useState('all');
  const [selectedSeverity, setSelectedSeverity] = useState('all');

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h1 className="text-slate-900 mb-2">Analytics & Reports</h1>
        <p className="text-sm text-slate-600">Generate compliance-ready reports and analytics insights</p>
      </div>

      {/* Filters Section */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Filter className="w-5 h-5 text-slate-600" />
          <h2 className="text-slate-900">Report Filters</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-slate-700 mb-2">Date Range</label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                defaultValue="Last 30 days"
                className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-700 mb-2">Department</label>
            <select
              value={selectedDepartment}
              onChange={(e) => setSelectedDepartment(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="all">All Departments</option>
              <option value="emergency">Emergency</option>
              <option value="icu">ICU</option>
              <option value="cardiac">Cardiac</option>
              <option value="pediatric">Pediatric</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-slate-700 mb-2">Severity Level</label>
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="all">All Levels</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          <div className="flex items-end">
            <button className="w-full px-4 py-2 bg-cyan-600 text-white rounded-lg text-sm hover:bg-cyan-700 transition-colors">
              Apply Filters
            </button>
          </div>
        </div>
      </div>

      {/* Trend Insights */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-slate-900 mb-6">Monthly Trend Insights</h2>
        <p className="text-sm text-slate-600 mb-6">This month vs last month</p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {trendInsights.map((insight) => {
            const TrendIcon = insight.direction === 'up' ? TrendingUp : TrendingDown;
            const trendColor = insight.status === 'warning' 
              ? 'text-amber-600' 
              : 'text-green-600';
            const bgColor = insight.status === 'warning'
              ? 'bg-amber-50 border-amber-200'
              : 'bg-green-50 border-green-200';

            return (
              <div key={insight.metric} className={`p-4 rounded-lg border ${bgColor}`}>
                <p className="text-sm text-slate-600 mb-2">{insight.metric}</p>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-2xl text-slate-900">{insight.current}</span>
                  <span className="text-xs text-slate-500">from {insight.previous}</span>
                </div>
                <div className={`flex items-center gap-1 text-sm ${trendColor}`}>
                  <TrendIcon className="w-4 h-4" />
                  <span>{insight.change}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Downloadable Reports */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-slate-900 mb-1">Available Reports</h2>
            <p className="text-sm text-slate-500">Hospital audit-ready documentation</p>
          </div>
          <button className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 text-slate-700 rounded-lg text-sm hover:bg-slate-50 transition-colors">
            <Download className="w-4 h-4" />
            Export All
          </button>
        </div>

        <div className="space-y-4">
          {availableReports.map((report) => (
            <div
              key={report.id}
              className="p-5 border border-slate-200 rounded-lg hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex gap-4 flex-1">
                  <div className="p-3 bg-cyan-100 rounded-lg h-fit">
                    <FileText className="w-5 h-5 text-cyan-700" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-slate-900 mb-1">{report.title}</h3>
                    <p className="text-sm text-slate-600 mb-2">{report.description}</p>
                    <div className="flex items-center gap-4">
                      <span className="text-xs text-slate-500">{report.date}</span>
                      <span className="text-xs px-2 py-1 bg-slate-100 text-slate-700 rounded-full">
                        {report.type}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 ml-4">
                  <button className="px-4 py-2 bg-white border border-slate-300 text-slate-700 rounded-lg text-sm hover:bg-slate-50 transition-colors">
                    PDF
                  </button>
                  <button className="px-4 py-2 bg-white border border-slate-300 text-slate-700 rounded-lg text-sm hover:bg-slate-50 transition-colors">
                    CSV
                  </button>
                  <button className="px-4 py-2 bg-cyan-600 text-white rounded-lg text-sm hover:bg-cyan-700 transition-colors">
                    View
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Surge Drivers */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-slate-900 mb-6">Top Surge Drivers (Last 30 Days)</h2>
        
        <div className="space-y-3">
          <div className="flex items-center gap-4 p-4 bg-slate-50 rounded-lg">
            <div className="flex-1">
              <h3 className="text-sm text-slate-900 mb-1">Respiratory Cases Spike</h3>
              <p className="text-xs text-slate-600">Temperature drop correlation: -8°C average</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-slate-900">+142 cases</p>
              <p className="text-xs text-amber-600">+28% impact</p>
            </div>
          </div>

          <div className="flex items-center gap-4 p-4 bg-slate-50 rounded-lg">
            <div className="flex-1">
              <h3 className="text-sm text-slate-900 mb-1">Weekend Admissions</h3>
              <p className="text-xs text-slate-600">Saturday/Sunday pattern detected</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-slate-900">+89 cases</p>
              <p className="text-xs text-amber-600">+18% impact</p>
            </div>
          </div>

          <div className="flex items-center gap-4 p-4 bg-slate-50 rounded-lg">
            <div className="flex-1">
              <h3 className="text-sm text-slate-900 mb-1">Holiday Period Load</h3>
              <p className="text-xs text-slate-600">Year-end seasonal increase</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-slate-900">+67 cases</p>
              <p className="text-xs text-amber-600">+13% impact</p>
            </div>
          </div>
        </div>
      </div>

      {/* Compliance Footer */}
      <div className="bg-slate-100 rounded-lg border border-slate-300 p-4">
        <p className="text-xs text-slate-600 text-center">
          All reports are HIPAA-compliant and audit-ready. Data retention: 7 years. Last audit: December 15, 2025
        </p>
      </div>
    </div>
  );
}
