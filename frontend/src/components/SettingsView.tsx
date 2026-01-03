import { useState } from 'react';
import { Save, Settings as SettingsIcon, Bell, Users, Database, Shield } from 'lucide-react';

export function SettingsView() {
  const [totalBeds, setTotalBeds] = useState(240);
  const [icuBeds, setICUBeds] = useState(96);
  const [icuThreshold, setICUThreshold] = useState(90);
  const [staffLoadThreshold, setStaffLoadThreshold] = useState(8);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [smsNotifications, setSmsNotifications] = useState(true);

  const departments = [
    { name: 'General ICU', beds: 48 },
    { name: 'Cardiac ICU', beds: 24 },
    { name: 'Neuro ICU', beds: 16 },
    { name: 'Pediatric ICU', beds: 8 },
    { name: 'Emergency', beds: 60 },
    { name: 'General Ward', beds: 84 },
  ];

  const users = [
    { name: 'Dr. Sarah Mitchell', role: 'Hospital Administrator', email: 's.mitchell@hospital.com', status: 'active' },
    { name: 'Dr. James Chen', role: 'ICU Head', email: 'j.chen@hospital.com', status: 'active' },
    { name: 'Maria Rodriguez', role: 'Operations Manager', email: 'm.rodriguez@hospital.com', status: 'active' },
    { name: 'Dr. Emily Wright', role: 'Emergency Director', email: 'e.wright@hospital.com', status: 'active' },
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-2">
          <SettingsIcon className="w-6 h-6 text-slate-700" />
          <h1 className="text-slate-900">System Configuration</h1>
        </div>
        <p className="text-sm text-slate-600">Hospital capacity, alerts, and user management</p>
      </div>

      {/* Hospital Configuration */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Database className="w-5 h-5 text-slate-600" />
          <h2 className="text-slate-900">Hospital Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div>
            <label className="block text-sm text-slate-700 mb-2">Total Hospital Beds</label>
            <input
              type="number"
              value={totalBeds}
              onChange={(e) => setTotalBeds(Number(e.target.value))}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />
          </div>

          <div>
            <label className="block text-sm text-slate-700 mb-2">Total ICU Beds</label>
            <input
              type="number"
              value={icuBeds}
              onChange={(e) => setICUBeds(Number(e.target.value))}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />
          </div>
        </div>

        <div className="border-t border-slate-200 pt-6">
          <h3 className="text-sm text-slate-900 mb-4">Department-wise Capacity</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left py-3 px-4 text-sm text-slate-600">Department</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-600">Current Beds</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-600">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {departments.map((dept) => (
                  <tr key={dept.name} className="hover:bg-slate-50">
                    <td className="py-3 px-4 text-sm text-slate-900">{dept.name}</td>
                    <td className="py-3 px-4 text-sm text-slate-700">{dept.beds}</td>
                    <td className="py-3 px-4">
                      <button className="text-sm text-cyan-700 hover:text-cyan-800">
                        Edit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Alert Thresholds */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Bell className="w-5 h-5 text-slate-600" />
          <h2 className="text-slate-900">Alert Thresholds</h2>
        </div>

        <div className="space-y-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-slate-700">ICU Occupancy Alert (%)</label>
              <span className="text-sm text-slate-900">{icuThreshold}%</span>
            </div>
            <input
              type="range"
              min="70"
              max="100"
              value={icuThreshold}
              onChange={(e) => setICUThreshold(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-cyan-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>70%</span>
              <span>85%</span>
              <span>100%</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-slate-700">Staff Load Index Alert</label>
              <span className="text-sm text-slate-900">{staffLoadThreshold}/10</span>
            </div>
            <input
              type="range"
              min="5"
              max="10"
              step="0.5"
              value={staffLoadThreshold}
              onChange={(e) => setStaffLoadThreshold(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-cyan-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>5/10</span>
              <span>7.5/10</span>
              <span>10/10</span>
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-700 mb-2">Surge Sensitivity</label>
            <select className="w-full px-4 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
              <option value="low">Low - Fewer alerts</option>
              <option value="medium" selected>Medium - Balanced</option>
              <option value="high">High - More sensitive</option>
            </select>
            <p className="text-xs text-slate-500 mt-2">Controls how aggressively the system alerts for potential surges</p>
          </div>
        </div>
      </div>

      {/* Notification Preferences */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Bell className="w-5 h-5 text-slate-600" />
          <h2 className="text-slate-900">Notification Preferences</h2>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
            <div>
              <h3 className="text-sm text-slate-900 mb-1">Email Notifications</h3>
              <p className="text-xs text-slate-600">Receive alerts via email</p>
            </div>
            <button
              onClick={() => setEmailNotifications(!emailNotifications)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                emailNotifications ? 'bg-cyan-600' : 'bg-slate-300'
              }`}
            >
              <div
                className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                  emailNotifications ? 'translate-x-6' : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
            <div>
              <h3 className="text-sm text-slate-900 mb-1">SMS Notifications</h3>
              <p className="text-xs text-slate-600">Receive critical alerts via SMS</p>
            </div>
            <button
              onClick={() => setSmsNotifications(!smsNotifications)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                smsNotifications ? 'bg-cyan-600' : 'bg-slate-300'
              }`}
            >
              <div
                className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                  smsNotifications ? 'translate-x-6' : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
            <div>
              <h3 className="text-sm text-slate-900 mb-1">In-App Alerts</h3>
              <p className="text-xs text-slate-600">Show notifications in dashboard</p>
            </div>
            <button className="relative w-12 h-6 rounded-full transition-colors bg-cyan-600">
              <div className="absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform translate-x-6" />
            </button>
          </div>
        </div>
      </div>

      {/* User & Role Management */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-slate-600" />
            <h2 className="text-slate-900">User & Role Management</h2>
          </div>
          <button className="px-4 py-2 bg-cyan-600 text-white rounded-lg text-sm hover:bg-cyan-700 transition-colors">
            Add User
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left py-3 px-4 text-sm text-slate-600">Name</th>
                <th className="text-left py-3 px-4 text-sm text-slate-600">Role</th>
                <th className="text-left py-3 px-4 text-sm text-slate-600">Email</th>
                <th className="text-left py-3 px-4 text-sm text-slate-600">Status</th>
                <th className="text-left py-3 px-4 text-sm text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {users.map((user) => (
                <tr key={user.email} className="hover:bg-slate-50">
                  <td className="py-3 px-4 text-sm text-slate-900">{user.name}</td>
                  <td className="py-3 px-4 text-sm text-slate-700">{user.role}</td>
                  <td className="py-3 px-4 text-sm text-slate-600">{user.email}</td>
                  <td className="py-3 px-4">
                    <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                      {user.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 space-x-3">
                    <button className="text-sm text-cyan-700 hover:text-cyan-800">Edit</button>
                    <button className="text-sm text-slate-600 hover:text-slate-700">Disable</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Security & Compliance */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="w-5 h-5 text-slate-600" />
          <h2 className="text-slate-900">Security & Compliance</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-slate-50 rounded-lg">
            <h3 className="text-sm text-slate-900 mb-2">Last Security Audit</h3>
            <p className="text-xs text-slate-600">December 15, 2025</p>
            <p className="text-xs text-green-600 mt-1">✓ Passed all checks</p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg">
            <h3 className="text-sm text-slate-900 mb-2">HIPAA Compliance</h3>
            <p className="text-xs text-slate-600">Status: Compliant</p>
            <p className="text-xs text-green-600 mt-1">✓ All requirements met</p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg">
            <h3 className="text-sm text-slate-900 mb-2">Data Retention</h3>
            <p className="text-xs text-slate-600">Policy: 7 years</p>
            <p className="text-xs text-slate-500 mt-1">As per hospital policy</p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg">
            <h3 className="text-sm text-slate-900 mb-2">Access Logs</h3>
            <p className="text-xs text-slate-600">Retention: 90 days</p>
            <button className="text-xs text-cyan-700 mt-1 hover:text-cyan-800">View logs</button>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end gap-3">
        <button className="px-6 py-2.5 bg-white border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors">
          Cancel
        </button>
        <button className="flex items-center gap-2 px-6 py-2.5 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors">
          <Save className="w-4 h-4" />
          Save Changes
        </button>
      </div>
    </div>
  );
}
