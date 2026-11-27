import { useEffect, useState } from 'react';
import { Sun, LayoutDashboard, ChartNoAxesCombined, Zap, Menu, Wifi, WifiOff } from 'lucide-react';
import DashboardPage from './pages/DashboardPage.jsx';
import PredictionPage from './pages/PredictionPage.jsx';
import SolarBillPage from './pages/SolarBillPage.jsx';
import { api } from './services/api.js';
import './styles.css';

const navItems = [
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
  { id: 'prediction', label: 'Electricity forecast', icon: ChartNoAxesCombined },
  { id: 'solar', label: 'Solar and bill', icon: Sun },
];

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [households, setHouseholds] = useState([]);
  const [connection, setConnection] = useState('checking');
  useEffect(() => {
    api.health().then(() => setConnection('online')).catch(() => setConnection('offline'));
    api.dashboard().then((data) => setHouseholds(data.data?.households || [])).catch(() => {});
  }, []);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><div className="brand-mark"><Sun size={23} /></div><div><strong>HomeSolar</strong><small>Household energy platform</small></div></div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="side-nav">
          {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={page === id ? 'nav-item active' : 'nav-item'} onClick={() => setPage(id)}><Icon size={18} /><span>{label}</span></button>)}
        </nav>
        <div className="sidebar-bottom"><div className={`connection ${connection}`}><span className="connection-dot" />{connection === 'online' ? 'Express API connected' : connection === 'checking' ? 'Connecting to API…' : 'API offline'}</div><div className="sidebar-footnote">CS698B · HomeSolar project<br />Bareilly household energy data</div></div>
      </aside>
      <main className="main-area">
        <header className="topbar"><div className="topbar-breadcrumb"><span>HomeSolar</span><span>/</span><strong>{navItems.find((item) => item.id === page)?.label || 'Overview'}</strong></div><div className="topbar-status">{connection === 'online' ? <><Wifi size={15} /> API online</> : <><WifiOff size={15} /> {connection}</>}</div></header>
        <div className="page-content">
          {page === 'dashboard' && <DashboardPage />}
          {page === 'prediction' && <PredictionPage households={households} />}
          {page === 'solar' && <SolarBillPage households={households} />}
        </div>
        <footer className="footer-note">HomeSolar estimates are for project demonstration. Verify tariff applicability and use current meter and weather data before making real household decisions.</footer>
      </main>
    </div>
  );
}
