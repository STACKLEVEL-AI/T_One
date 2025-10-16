import React, { useState } from 'react';
import UserSupportInterface from './components/UserSupportInterface';
import SupportAdminInterface from './components/SupportAdminInterface';

const VTB_ACTIVE = 'bg-gradient-to-r from-[#009FDF] to-[#0A2973] text-white';

const App: React.FC = () => {
  const [view, setView] = useState<'user' | 'admin'>('user');

  return (
    <div>
      <div className="fixed top-4 right-4 z-50 bg-white rounded-xl shadow-lg border border-gray-200 p-1 flex gap-1">
        <button
          onClick={() => setView('user')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            view === 'user' ? VTB_ACTIVE : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Интерфейс пользователя
        </button>
        <button
          onClick={() => setView('admin')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            view === 'admin' ? VTB_ACTIVE : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Панель поддержки
        </button>
      </div>

      {view === 'user' ? <UserSupportInterface /> : <SupportAdminInterface />}
    </div>
  );
};

export default App;
