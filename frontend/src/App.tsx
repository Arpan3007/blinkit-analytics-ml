import { useMemo, useState } from 'react';
import {
  BarChart3, BrainCircuit, Database, FileSpreadsheet, Gauge,
  LayoutDashboard, Palette, Package, PanelLeftClose, PanelLeftOpen,
  Settings, ShoppingBag, Users, Upload, Store, Sparkles,
  CheckCircle2, AlertCircle, TrendingUp, Activity,
  Filter, RefreshCw, Eye, Zap, Target, Award, Layers, ArrowUpRight,
  ArrowDownRight, BarChart2, Hash, Type,
  Trash2, Copy, Search, ChevronUp, ChevronDown, Info, XCircle,
  RotateCcw, AlertTriangle
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid,
  ResponsiveContainer, Tooltip, XAxis, YAxis, PieChart as RPieChart,
  Pie, Cell, Area, AreaChart, Legend,
  ScatterChart, Scatter, ZAxis, LineChart, Line
} from 'recharts';
import { uploadCSV, train, predict, predictCustom, getAIInsights, kaggleListFiles, kaggleImportFile, kaggleCleanup, exportDatasetUrl, preprocessDataset } from './lib/api';
import type { Dataset } from './types';

type Page = 'Dashboard' | 'Sales Analysis' | 'Customer Analysis' | 'Product Analysis' | 'Outlet Analysis' | 'Data Preprocessing' | 'ML Studio' | 'Dataset Explorer' | 'Settings';
type Theme = 'sunny' | 'dark' | 'purple' | 'red';

const NAV_ITEMS: [Page, any, string][] = [
  ['Dashboard', LayoutDashboard, 'Overview & KPIs'],
  ['Sales Analysis', BarChart3, 'Revenue trends'],
  ['Customer Analysis', Users, 'Behaviour insights'],
  ['Product Analysis', Package, 'Item performance'],
  ['Outlet Analysis', Store, 'Location metrics'],
  ['Data Preprocessing', Database, 'Clean & transform'],
  ['ML Studio', BrainCircuit, 'Train & compare'],
  ['Dataset Explorer', FileSpreadsheet, 'Raw data view'],
  ['Settings', Settings, 'Appearance'],
];

const PALETTE = ['#f2c200','#7657f5','#e33b46','#20b26b','#f47b20','#1a9bd6','#e91e8c','#00bcd4'];

function fmt(n: number) { return n >= 1e6 ? `${(n/1e6).toFixed(2)}M` : n >= 1e3 ? `${(n/1e3).toFixed(1)}K` : String(Math.round(n)); }
function pct(n: number, t: number) { return t ? `${((n/t)*100).toFixed(1)}%` : '0%'; }

const RELEVANT_KEYWORDS = [
  'sales','revenue','amount','price','item','product','category','outlet',
  'store','fat','weight','rating','visibility','mrp','type','size','location',
  'establishment','identifier','sold','purchase','grocery','blinkit','retail',
  'customer','transaction','quantity','order','sku','stock','inventory','brand'
];

function checkDatasetRelevance(columns: any[]): { relevant: boolean; score: number } {
  const colNames = columns.map((c: any) => c.name.toLowerCase());
  let hits = 0;
  for (const kw of RELEVANT_KEYWORDS) {
    if (colNames.some(c => c.includes(kw))) hits++;
  }
  return { relevant: hits >= 3, score: hits };
}

type DataSourceMeta = {
  sourceType: 'csv' | 'kaggle' | 'quickcommerce';
  sourceName: string;
  sourceUrl?: string;
  datasetName?: string;
  fileName?: string;
  platform?: string;
};

function App() {
  const [page, setPage] = useState<Page>('Dashboard');
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<Theme>('sunny');
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [sourceMeta, setSourceMeta] = useState<DataSourceMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [irrelevantWarning, setIrrelevantWarning] = useState<{ show: boolean; filename: string; score: number } | null>(null);
  const [pendingDataset, setPendingDataset] = useState<Dataset | null>(null);
  const [pendingMeta, setPendingMeta] = useState<DataSourceMeta | null>(null);

  // Shared activator — called by all 3 sources
  const activateDataset = (d: Dataset, meta: DataSourceMeta, skipRelevance = false) => {
    if (!skipRelevance) {
      const { relevant, score } = checkDatasetRelevance(d.columns);
      if (!relevant) {
        setPendingDataset(d); setPendingMeta(meta);
        setIrrelevantWarning({ show: true, filename: d.filename || meta.fileName || '', score });
        return;
      }
    }
    setDataset(d); setSourceMeta(meta); setPage('Dashboard');
  };

  const upload = async (file: File) => {
    setLoading(true); setError('');
    try {
      const d = await uploadCSV(file);
      activateDataset(d, { sourceType: 'csv', sourceName: 'CSV Upload', fileName: file.name });
    }
    catch (e: any) { setError(e.message || 'Upload failed'); }
    finally { setLoading(false); }
  };

  const confirmIrrelevant = () => {
    if (pendingDataset && pendingMeta) { setDataset(pendingDataset); setSourceMeta(pendingMeta); setPage('Dashboard'); }
    setIrrelevantWarning(null); setPendingDataset(null); setPendingMeta(null);
  };

  const dismissIrrelevant = () => { setIrrelevantWarning(null); setPendingDataset(null); setPendingMeta(null); };

  const handleRestart = () => {
    setDataset(null); setSourceMeta(null); setModel(null); setPrediction(null);
    setError(''); setPage('Dashboard');
  };

  return (
    <div className={`app theme-${theme}`}>
      {irrelevantWarning?.show && (
        <div className="modal-overlay">
          <div className="modal-box">
            <div className="modal-icon warn"><AlertTriangle size={32} /></div>
            <h3>Dataset May Be Irrelevant</h3>
            <p><b>"{irrelevantWarning.filename}"</b> doesn't appear to be a sales/retail dataset. This app is designed for <b>Sales, Customer &amp; Product analytics</b>.</p>
            <p className="modal-sub">Only <b>{irrelevantWarning.score}</b> out of expected column keywords were found. Analysis pages may show limited or misleading insights.</p>
            <div className="modal-actions">
              <button className="btn-modal-cancel" onClick={dismissIrrelevant}><XCircle size={15} /> Discard &amp; Re-upload</button>
              <button className="btn-modal-confirm" onClick={confirmIrrelevant}><CheckCircle2 size={15} /> Continue Anyway</button>
            </div>
          </div>
        </div>
      )}
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        <div className="brand">
          <div className="brand-mark">IF</div>
          {!collapsed && <div><b>InsightForge</b><span>Analytics Studio</span></div>}
        </div>
        <button className="collapse-btn" onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
        <nav className="nav">
          {NAV_ITEMS.map(([name, Icon, sub]) => (
            <button key={name} className={`nav-item ${page === name ? 'active' : ''}`} title={name} onClick={() => setPage(name)}>
              <div className="nav-icon"><Icon size={17} /></div>
              {!collapsed && <div className="nav-text"><span>{name}</span><small>{sub}</small></div>}
            </button>
          ))}
        </nav>
        {!collapsed && (
          <div className="sidebar-footer">
            <div className="api-status"><span className="status-dot" />API Connected</div>
            {dataset && <div className="ds-badge"><FileSpreadsheet size={12}/>{dataset.filename}</div>}
          </div>
        )}
      </aside>
      <main className={`main ${collapsed ? 'main-collapsed' : ''}`}>
        <header className="topbar">
          <div className="topbar-left">
            <div className="page-eyebrow">DATA WAREHOUSE &amp; MINING</div>
            <h1 className="page-title">{page}</h1>
          </div>
          <div className="topbar-right">
            {dataset && <div className="ds-chip"><Activity size={13}/>{dataset.summary.rows.toLocaleString()} rows</div>}
            {dataset && sourceMeta && (
              <div className="ds-chip" style={{maxWidth:160,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',fontSize:11}}>
                {sourceMeta.sourceType === 'csv' ? '📁' : sourceMeta.sourceType === 'kaggle' ? '📊' : '🔴'} {sourceMeta.sourceName}
              </div>
            )}
            <label className={`btn-primary ${loading ? 'loading' : ''}`}>
              <Upload size={15}/>{loading ? 'Processing…' : 'Import CSV'}
              <input type="file" accept=".csv" onChange={e => e.target.files?.[0] && upload(e.target.files[0])} />
            </label>
            <button className="btn-icon" onClick={() => setTheme(t => t==='sunny'?'dark':t==='dark'?'purple':t==='purple'?'red':'sunny')} title="Cycle theme">
              <Palette size={17} />
            </button>
          </div>
        </header>
        {error && <div className="alert-error"><AlertCircle size={16}/><span>{error}</span><button onClick={() => setError('')}>✕</button></div>}
        <div className="page-body">
          {page === 'Settings' ? (
            <SettingsPage theme={theme} setTheme={setTheme} dataset={dataset} model={model} onRestart={handleRestart} />
          ) : !dataset ? (
            <Welcome onUpload={upload} onDatasetReady={(d, meta) => activateDataset(d, meta, false)} />
          ) : (
            <>
              {page === 'Dashboard'          && <Dashboard dataset={dataset} sourceMeta={sourceMeta} />}
              {page === 'Sales Analysis'     && <SalesAnalysis dataset={dataset} />}
              {page === 'Customer Analysis'  && <CustomerAnalysis dataset={dataset} />}
              {page === 'Product Analysis'   && <ProductAnalysis dataset={dataset} />}
              {page === 'Outlet Analysis'    && <OutletAnalysis dataset={dataset} />}
              {page === 'Data Preprocessing' && <DataPreprocessing dataset={dataset} setDataset={setDataset} />}
              {page === 'Dataset Explorer'   && <DatasetExplorer dataset={dataset} />}
              {page === 'ML Studio'          && <MLStudio dataset={dataset} setModel={setModel} model={model} setPrediction={setPrediction} prediction={prediction} onUpload={upload} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function Welcome({ onUpload, onDatasetReady }: { onUpload: (f: File) => void; onDatasetReady: (d: any, meta: any) => void }) {
  return (
    <div className="welcome">
      <div className="welcome-glow" />
      <div className="welcome-content">
        <div className="welcome-badge"><Sparkles size={14} /> Smart Analytics</div>
        <h2>Turn any dataset into<br /><span className="gradient-text">actionable intelligence</span></h2>
        <p>Import a CSV or load a dataset directly from Kaggle.</p>
        <div className="datasource-cards">
          <CSVSource onUpload={onUpload} />
          <KaggleSource onDatasetReady={onDatasetReady} />
        </div>
        <div className="feature-grid" style={{marginTop:32}}>
          {[['📊','Sales Intelligence','Revenue & trend analysis'],['👥','Customer Insights','Behaviour segmentation'],['🤖','ML Studio','Real CART/RF algorithms'],['🔬','Data Preprocessing','Clean, encode, transform']].map(([icon, t, d]) => (
            <div className="feature-card" key={t}><div className="feature-icon">{icon}</div><b>{t}</b><span>{d}</span></div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── CSV Source ───────────────────────────────────────────────────────── */
function CSVSource({ onUpload }: { onUpload: (f: File) => void }) {
  return (
    <div className="ds-card">
      <div className="ds-card-header">
        <span className="ds-icon">📁</span>
        <div><b>Import CSV</b><small>Upload an existing dataset</small></div>
      </div>
      <label className="btn-primary full ds-btn">
        <Upload size={15} /> Choose CSV File
        <input type="file" accept=".csv" onChange={e => e.target.files?.[0] && onUpload(e.target.files[0])} />
      </label>
    </div>
  );
}

/* ── Kaggle Source ────────────────────────────────────────────────────── */
function KaggleSource({ onDatasetReady }: { onDatasetReady: (d: any, meta: any) => void }) {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState<'idle'|'loading'|'files'|'importing'>('idle');
  const [files, setFiles] = useState<any[]>([]);
  const [downloadId, setDownloadId] = useState('');
  const [owner, setOwner] = useState('');
  const [slug, setSlug] = useState('');
  const [selected, setSelected] = useState('');
  const [err, setErr] = useState('');

  const listFiles = async () => {
    if (!url.trim()) return;
    setStatus('loading'); setErr('');
    try {
      const res = await kaggleListFiles(url.trim());
      setFiles(res.files); setDownloadId(res.download_id);
      setOwner(res.owner); setSlug(res.slug);
      setSelected(res.files[0]?.filename || '');
      setStatus('files');
    } catch(e: any) { setErr(e.message || 'Failed to fetch Kaggle dataset'); setStatus('idle'); }
  };

  const importFile = async () => {
    if (!selected) return;
    setStatus('importing'); setErr('');
    try {
      const d = await kaggleImportFile(downloadId, selected);
      kaggleCleanup(downloadId).catch(() => {});
      onDatasetReady({
        ...d,
        filename: selected,
        source_type: 'kaggle'
      }, {
        sourceType: 'kaggle',
        sourceName: `${owner}/${slug}`,
        sourceUrl: url,
        datasetName: `${owner}/${slug}`,
        fileName: selected
      });
    } catch(e: any) { setErr(e.message || 'Import failed'); setStatus('files'); }
  };

  return (
    <div className="ds-card">
      <div className="ds-card-header">
        <span className="ds-icon">📊</span>
        <div><b>Kaggle Dataset</b><small>Import via Kaggle URL</small></div>
      </div>
      {status === 'idle' || status === 'loading' ? (
        <>
          <label className="form-label" style={{fontSize:11,fontWeight:700,color:'var(--muted)',textTransform:'uppercase'}}>Kaggle Dataset URL</label>
          <input
            className="form-input"
            placeholder="https://www.kaggle.com/datasets/owner/dataset-name"
            value={url} onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && listFiles()}
          />
          <button className={`btn-primary full ds-btn ${status === 'loading' ? 'loading' : ''}`} onClick={listFiles} disabled={status === 'loading' || !url.trim()}>
            {status === 'loading' ? 'Fetching…' : 'Fetch Dataset Files'}
          </button>
          <div className="ds-status-row">
            <span className="ds-dot disconnected" />
            <span style={{fontSize:11,color:'var(--muted)'}}>Not connected</span>
          </div>
        </>
      ) : (
        <>
          <div style={{fontSize:12,color:'var(--muted)',marginBottom:8}}>📦 <b>{owner}/{slug}</b> — {files.length} file{files.length !== 1 ? 's' : ''} found</div>
          <div className="kaggle-file-list">
            {files.map((f: any) => (
              <label key={f.filename} className={`kaggle-file-item ${selected === f.filename ? 'selected' : ''}`}>
                <input type="radio" name="kf" value={f.filename} checked={selected === f.filename} onChange={() => setSelected(f.filename)} />
                <span className="kf-name">{f.filename}</span>
                <span className="kf-meta">{f.row_count != null ? `${f.row_count.toLocaleString()} rows` : ''} · {f.size_kb}KB</span>
              </label>
            ))}
          </div>
          <div style={{display:'flex',gap:8,marginTop:8}}>
            <button className="btn-secondary" style={{flex:1}} onClick={() => { setStatus('idle'); setFiles([]); setDownloadId(''); }}>
              ← Back
            </button>
            <button className={`btn-primary ${status === 'importing' ? 'loading' : ''}`} style={{flex:2}} onClick={importFile} disabled={status === 'importing' || !selected}>
              {status === 'importing' ? 'Importing…' : 'Import Selected'}
            </button>
          </div>
        </>
      )}
      {err && <div className="ds-error">{err}</div>}
    </div>
  );
}

/* ── Quick Commerce Source ────────────────────────────────────────────── */


function Card({ title, subtitle, children, action }: { title: string; subtitle?: string; children: any; action?: any }) {
  return (
    <section className="card">
      <div className="card-header">
        <div><h3>{title}</h3>{subtitle && <p className="card-sub">{subtitle}</p>}</div>
        {action && <div className="card-action">{action}</div>}
      </div>
      {children}
    </section>
  );
}

function KpiCard({ icon: Icon, label, value, sub, trend, color }: any) {
  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <div className="kpi-icon" style={{ background: `${color}22`, color }}><Icon size={18} /></div>
        {trend !== undefined && (
          <div className={`kpi-trend ${trend >= 0 ? 'up' : 'down'}`}>
            {trend >= 0 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
            {Math.abs(trend)}%
          </div>
        )}
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tt-label">{label}</div>
      {payload.map((p: any, i: number) => (
        <div key={i} className="tt-row"><span style={{ color: p.color }}>●</span>{p.name}: <b>{typeof p.value === 'number' ? p.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : p.value}</b></div>
      ))}
    </div>
  );
}

function NoDataState({ message }: { message: string }) {
  return (
    <div className="empty-state" style={{ minHeight: 120 }}>
      <AlertCircle size={28} />
      <b>No matching column</b>
      <p>{message}</p>
    </div>
  );
}

function InsightTile({ icon, color, title, text }: any) {
  return (
    <div className="insight-tile" style={{ borderColor: `${color}33` }}>
      <div className="it-icon" style={{ background: `${color}22`, color }}>{icon}</div>
      <div className="it-body"><b>{title}</b><p>{text}</p></div>
    </div>
  );
}
/* ══════════════════════════════════════════════════════
   DASHBOARD
══════════════════════════════════════════════════════ */
function Dashboard({ dataset, sourceMeta }: { dataset: Dataset; sourceMeta?: DataSourceMeta | null }) {
  const s = dataset.summary;
  const numDists = dataset.charts?.numeric_distributions || [];
  const cats = dataset.charts?.categorical || [];
  const qualityScore = Math.round(100 - (s.missing_values / Math.max(s.rows * s.columns, 1)) * 100);

  const distData = numDists.slice(0, 6).map((n: any) => ({
    name: n.name.length > 14 ? n.name.slice(0, 14) + '…' : n.name,
    mean: parseFloat(n.mean.toFixed ? n.mean.toFixed(2) : n.mean),
    min: parseFloat(n.min.toFixed ? n.min.toFixed(2) : n.min),
    max: parseFloat(n.max.toFixed ? n.max.toFixed(2) : n.max),
  }));

  const catData = cats[0]?.items || [];

  const sourceLabel = sourceMeta?.sourceType === 'csv' ? '📁 CSV Upload'
    : sourceMeta?.sourceType === 'kaggle' ? `📊 Kaggle — ${sourceMeta.datasetName}`
    : sourceMeta?.sourceType === 'quickcommerce' ? `🔴 Real-Time Quick Commerce (${sourceMeta.platform})`
    : '📁 Dataset';

  const exportFilename = sourceMeta?.sourceType === 'quickcommerce'
    ? `quickcommerce_${sourceMeta.platform || 'data'}.csv`
    : sourceMeta?.sourceType === 'kaggle'
    ? `kaggle_${(sourceMeta.fileName || 'dataset').replace(/[/\\]/g, '_')}`
    : dataset.filename || 'dataset.csv';

  return (
    <div className="content">
      <div className="ds-banner">
        <div className="ds-banner-left">
          <div className="pill active-pill"><span className="pulse-dot" />ACTIVE DATASET</div>
          <h2 className="ds-name">{dataset.filename}</h2>
          <p className="ds-meta">
            {s.rows.toLocaleString()} rows · {s.columns} columns
            {sourceMeta && <> · <span style={{color:'var(--accent)'}}>{sourceLabel}</span></>}
          </p>
        </div>
        <div className="ds-banner-stats" style={{display:'flex',flexDirection:'column',alignItems:'flex-end',gap:10}}>
          <div className="quality-ring">
            <svg viewBox="0 0 60 60"><circle cx="30" cy="30" r="24" fill="none" stroke="var(--line)" strokeWidth="6"/><circle cx="30" cy="30" r="24" fill="none" stroke="var(--accent)" strokeWidth="6" strokeDasharray={`${qualityScore * 1.508} 150.8`} strokeLinecap="round" transform="rotate(-90 30 30)"/></svg>
            <div className="qr-text"><b>{qualityScore}%</b><span>Quality</span></div>
          </div>
          {(sourceMeta?.sourceType === 'kaggle' || sourceMeta?.sourceType === 'quickcommerce') && (
            <a
              href={exportDatasetUrl((dataset as any).dataset_id, exportFilename)}
              download={exportFilename}
              className="btn-primary"
              style={{fontSize:12,padding:'6px 14px',textDecoration:'none',display:'flex',alignItems:'center',gap:6}}
            >
              ⬇ Export CSV
            </a>
          )}
        </div>
      </div>

      <div className="kpi-row">
        <KpiCard icon={ShoppingBag} label="Total Records" value={s.rows.toLocaleString()} sub="All loaded rows" color="#f2c200" />
        <KpiCard icon={Layers} label="Features" value={s.columns} sub={`${s.numeric_columns} numeric`} color="#7657f5" />
        <KpiCard icon={BarChart2} label="Numeric Cols" value={s.numeric_columns} sub={`${s.categorical_columns} categorical`} color="#20b26b" />
        <KpiCard icon={AlertCircle} label="Missing Values" value={s.missing_values.toLocaleString()} sub={pct(s.missing_values, s.rows * s.columns) + ' of cells'} color="#e33b46" />
        <KpiCard icon={Database} label="Memory" value={`${s.memory_kb} KB`} sub="In-memory size" color="#1a9bd6" />
      </div>

      <div className="grid-3-2">
        <Card title="Numeric Feature Ranges" subtitle="Mean · Min · Max across numeric columns">
          {distData.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={distData} barSize={20}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="mean" name="Mean" fill={PALETTE[0]} radius={[4,4,0,0]} />
                  <Bar dataKey="max" name="Max" fill={PALETTE[1]} radius={[4,4,0,0]} opacity={0.6} />
                  <Bar dataKey="min" name="Min" fill={PALETTE[2]} radius={[4,4,0,0]} opacity={0.4} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No numeric columns detected in this dataset." />}
        </Card>
        <Card title="Category Distribution" subtitle={cats[0]?.column || 'First categorical field'}>
          {catData.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={260}>
                <RPieChart>
                  <Pie data={catData} cx="50%" cy="50%" outerRadius={90} innerRadius={45} dataKey="value" nameKey="label" label={({ label, percent }: any) => `${label} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                    {catData.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v: any) => [v.toLocaleString(), 'Count']} />
                </RPieChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No categorical columns detected." />}
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Data Quality Snapshot" subtitle="Automatic integrity check">
          <div className="quality-table">
            {[
              ['Total rows', s.rows.toLocaleString(), 'good'],
              ['Columns', s.columns, 'good'],
              ['Missing cells', s.missing_values.toLocaleString(), s.missing_values > 0 ? 'warn' : 'good'],
              ['Duplicates removed', s.duplicates, s.duplicates > 0 ? 'warn' : 'good'],
              ['Numeric columns', s.numeric_columns, 'good'],
              ['Categorical columns', s.categorical_columns, 'good'],
            ].map(([k, v, cls]) => (
              <div className="qt-row" key={String(k)}>
                <span>{k}</span>
                <div className={`qt-val ${cls}`}>{v}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Column Schema" subtitle="All detected columns">
          <div className="schema-scroll">
            {dataset.columns.map((c: any) => (
              <div className="schema-row" key={c.name}>
                <div className={`dtype-badge ${c.dtype.includes('float') ? 'num' : 'cat'}`}>
                  {c.dtype.includes('float') ? <Hash size={10}/> : <Type size={10}/>}
                  {c.dtype.includes('float') ? 'NUM' : 'CAT'}
                </div>
                <div className="schema-info">
                  <b>{c.name}</b>
                  <span>{c.unique} unique · {c.missing} missing</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Recommended Workflow" subtitle="Follow these steps for best results">
        <div className="workflow-steps">
          {[
            { n: '01', t: 'Explore Data', d: 'Review schema, distributions, and missing values in Dataset Explorer.', icon: Eye },
            { n: '02', t: 'Preprocess', d: 'Handle missing values, encode categories, and remove outliers.', icon: Filter },
            { n: '03', t: 'Analyse', d: 'Dive into Sales, Customer, Product, and Outlet analysis pages.', icon: BarChart3 },
            { n: '04', t: 'Train Models', d: 'Select a target column and compare real ML algorithms in ML Studio.', icon: BrainCircuit },
          ].map(({ n, t, d, icon: Icon }) => (
            <div className="wf-step" key={n}>
              <div className="wf-num">{n}</div>
              <div className="wf-icon"><Icon size={18} /></div>
              <b>{t}</b>
              <p>{d}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   SALES ANALYSIS
══════════════════════════════════════════════════════ */
function SalesAnalysis({ dataset }: { dataset: Dataset }) {
  const nums = dataset.charts?.numeric_distributions || [];
  const cats = dataset.charts?.categorical || [];

  const revenueCol = nums.find((n: any) => /sales|revenue|amount|total|value/i.test(n.name))
    || nums.find((n: any) => /price|mrp|cost/i.test(n.name));
  const qtyCol = nums.find((n: any) => /qty|quantity|count|units|sold/i.test(n.name));

  const catCol = cats.find((c: any) => /item|product|category|type/i.test(c.column));
  const catCol2 = cats.find((c: any) => c !== catCol && /outlet|store|location|size|tier/i.test(c.column));

  if (!catCol && !revenueCol) return (
    <div className="content">
      <div className="empty-state large"><TrendingUp size={48} color="#e33b46"/><br/><b>Irrelevant Dataset for Sales Analysis</b><p>No revenue/sales or category attributes detected in this dataset.</p></div>
    </div>
  );

  const trendData = (catCol?.items || []).slice(0, 8).map((item: any) => ({
    name: item.label?.length > 14 ? item.label.slice(0,14) + '…' : item.label,
    transactions: item.value,
    estRevenue: Math.round(item.value * (revenueCol?.mean || 0)),
  }));

  const totalTransactions = trendData.reduce((a: number, d: any) => a + d.transactions, 0);
  const totalEstRevenue = trendData.reduce((a: number, d: any) => a + d.estRevenue, 0);
  const topCat = trendData[0];

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge sales-badge"><TrendingUp size={13}/> Sales Intelligence</div>
        <h2>Sales Performance Analysis</h2>
        <p>Revenue trends, category breakdowns, and volume distribution derived from <b>{catCol?.column || 'dataset'}</b>.</p>
      </div>

      <div className="kpi-row">
        <KpiCard icon={TrendingUp} label="Est. Total Revenue" value={`₹${fmt(totalEstRevenue)}`} sub={`Count × ₹${revenueCol?.mean?.toFixed(2)} avg`} color="#20b26b" />
        <KpiCard icon={ShoppingBag} label="Total Transactions" value={totalTransactions.toLocaleString()} sub={`Across ${trendData.length} categories`} color="#f2c200" />
        <KpiCard icon={Award} label="Avg Sale Value" value={revenueCol ? `₹${revenueCol.mean?.toFixed(2)}` : 'N/A'} sub={revenueCol ? `Col: ${revenueCol.name}` : 'No revenue column found'} color="#7657f5" />
        <KpiCard icon={Target} label="Top Category" value={topCat?.name || 'N/A'} sub={topCat ? `${topCat.transactions.toLocaleString()} records` : ''} color="#e33b46" />
      </div>

      {trendData.length === 0 && (
        <Card title="No Categorical Data Found" subtitle="">
          <NoDataState message="This dataset has no categorical columns to group sales by. Try a retail/sales dataset with columns like Item_Type, Outlet_Type, or Category." />
        </Card>
      )}

      {trendData.length > 0 && (
        <div className="grid-2">
          <Card title={`Estimated Revenue by ${catCol?.column || 'Category'}`} subtitle={`Grouped from: ${revenueCol?.name || 'N/A'} (mean × count)`}>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trendData} barSize={32}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => `₹${fmt(v)}`} />
                  <Tooltip content={<CustomTooltip />} formatter={(v: any) => [`₹${fmt(Number(v))}`, 'Est. Revenue']} />
                  <Bar dataKey="estRevenue" name="Est. Revenue (₹)" fill="var(--accent)" radius={[6,6,0,0]}>
                    {trendData.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title={`Transaction Volume by ${catCol?.column || 'Category'}`} subtitle="Record count per category — actual data">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                  <Tooltip content={<CustomTooltip />} />
                  <defs>
                    <linearGradient id="salesGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="var(--accent)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <Area dataKey="transactions" name="Transactions" stroke="var(--accent)" fill="url(#salesGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      )}

      <div className="grid-2">
        <Card title="Numeric Feature Summary" subtitle="Statistical overview of all numeric columns">
          {nums.length > 0 ? (
            <div className="stats-table">
              <div className="st-head"><span>Column</span><span>Mean</span><span>Min</span><span>Max</span></div>
              {nums.map((n: any) => (
                <div className="st-row" key={n.name}>
                  <span className="st-name">{n.name}</span>
                  <b>{typeof n.mean === 'number' ? n.mean.toFixed(2) : n.mean}</b>
                  <span className="muted">{typeof n.min === 'number' ? n.min.toFixed(2) : n.min}</span>
                  <span className="muted">{typeof n.max === 'number' ? n.max.toFixed(2) : n.max}</span>
                </div>
              ))}
            </div>
          ) : <NoDataState message="No numeric columns in this dataset." />}
        </Card>

        <Card title={`Secondary Breakdown: ${catCol2?.column || 'N/A'}`} subtitle="Distribution of the second categorical dimension">
          {catCol2?.items?.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={240}>
                <RPieChart>
                  <Pie data={catCol2.items.slice(0, 8)} cx="50%" cy="50%" outerRadius={85} innerRadius={40} dataKey="value" nameKey="label">
                    {catCol2.items.slice(0, 8).map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v: any) => [v.toLocaleString(), 'Count']} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                </RPieChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="Only one categorical column detected." />}
        </Card>
      </div>

      <Card title="Sales Insights" subtitle="Automated observations from your data">
        <div className="insight-cards">
          <InsightTile icon="💡" color="#f2c200" title="Revenue Driver" text={revenueCol ? `"${revenueCol.name}" has mean ₹${revenueCol.mean?.toFixed(2)} with range ₹${revenueCol.min?.toFixed(2)}–₹${revenueCol.max?.toFixed(2)}. Consider targeting high-value segments.` : 'No revenue/price column detected. Columns with "sales", "revenue", "price", or "amount" would enable revenue analysis.'} />
          <InsightTile icon="📦" color="#7657f5" title="Category Mix" text={catCol ? `${cats.length} categorical dimensions detected. "${catCol.column}" has ${catCol.items?.length || 0} distinct values — ideal for segmentation and targeted marketing.` : 'No categorical columns detected for group analysis.'} />
          <InsightTile icon="🚀" color="#20b26b" title="Top Performer" text={topCat ? `Top category "${topCat.name}" drives ${pct(topCat.transactions, totalTransactions)} of all transactions (${topCat.transactions.toLocaleString()} records). Prioritise this segment for maximum impact.` : 'Upload a dataset with categorical columns to see performance leaders.'} />
        </div>
      </Card>
    </div>
  );
}


/* ══════════════════════════════════════════════════════
   CUSTOMER ANALYSIS
══════════════════════════════════════════════════════ */
function CustomerAnalysis({ dataset }: { dataset: Dataset }) {
  const nums = dataset.charts?.numeric_distributions || [];
  const cats = dataset.charts?.categorical || [];

  const incomeCol = nums.find((n: any) => /income|salary|earning|spend|amount|sales/i.test(n.name));
  const ratingCol = nums.find((n: any) => /rating|score|review|satisfaction/i.test(n.name));
  const locCol = cats.find((c: any) => /loc|city|region|area|zone|location|outlet/i.test(c.column));
  const typeCol = cats.find((c: any) => /type|tier|category|segment|customer/i.test(c.column) && c !== locCol);
  const genderCol = cats.find((c: any) => /gender|sex/i.test(c.column));

  if (!locCol && !typeCol) return (
    <div className="content">
      <div className="empty-state large"><Users size={48} color="#e33b46"/><br/><b>Irrelevant Dataset for Customer Analysis</b><p>No customer segment or location attributes detected in this dataset.</p></div>
    </div>
  );

  const locationDist = (locCol?.items || []).slice(0, 8);
  const totalRecords = locationDist.reduce((a: number, d: any) => a + d.value, 0);

  const typeDist = (typeCol?.items || []).slice(0, 6);
  const totalType = typeDist.reduce((a: number, d: any) => a + d.value, 0);

  const segmentData = typeDist.map((item: any, i: number) => ({
    name: item.label?.length > 12 ? item.label.slice(0,12)+'…' : item.label,
    value: item.value,
    fill: PALETTE[i % PALETTE.length],
  }));

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge customer-badge"><Users size={13}/> Customer Intelligence</div>
        <h2>Customer Behaviour Analysis</h2>
        <p>Segment customers by <b>{typeCol?.column || 'type'}</b>, and understand regional patterns from <b>{locCol?.column || 'location'}</b>.</p>
      </div>

      <div className="kpi-row">
        <KpiCard icon={Users} label="Total Records" value={dataset.summary.rows.toLocaleString()} sub="All rows in dataset" color="#7657f5" />
        <KpiCard icon={Store} label="Locations / Outlets" value={(locCol?.items?.length || 0).toLocaleString()} sub={locCol ? `Col: ${locCol.column}` : 'No location column'} color="#1a9bd6" />
        <KpiCard icon={Award} label="Customer Types" value={typeDist.length} sub={typeCol ? `Col: ${typeCol.column}` : 'No type column'} color="#20b26b" />
        <KpiCard icon={Activity} label={ratingCol ? 'Avg Rating' : (incomeCol ? 'Avg Value' : 'Numeric Cols')} value={ratingCol ? ratingCol.mean?.toFixed(2) : (incomeCol ? `₹${incomeCol.mean?.toFixed(2)}` : nums.length)} sub={ratingCol ? `Col: ${ratingCol.name}` : (incomeCol ? `Col: ${incomeCol.name}` : 'numeric features')} color="#f2c200" />
      </div>

      <div className="grid-2">
        <Card title={`${typeCol?.column || 'Customer'} Distribution`} subtitle="Actual record counts per type — no simulation">
          {segmentData.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <RPieChart>
                  <Pie data={segmentData} cx="50%" cy="50%" outerRadius={100} innerRadius={50} dataKey="value" nameKey="name" label={({ name, percent }: any) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                    {segmentData.map((s: any, i: number) => <Cell key={i} fill={s.fill} />)}
                  </Pie>
                  <Tooltip formatter={(v: any) => [v.toLocaleString(), 'Records']} />
                  <Legend iconType="circle" iconSize={9} wrapperStyle={{ fontSize: 10 }} />
                </RPieChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No categorical type/segment column found. Columns with 'type', 'tier', or 'segment' would populate this chart." />}
        </Card>

        <Card title={`Records by ${locCol?.column || 'Location'}`} subtitle="Actual count distribution per location">
          {locationDist.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={locationDist} layout="vertical" barSize={16}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--line)" />
                  <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                  <YAxis dataKey="label" type="category" tick={{ fontSize: 10, fill: 'var(--muted)' }} width={90} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Records" fill="var(--accent)" radius={[0,6,6,0]}>
                    {locationDist.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No location/outlet column found. Columns with 'location', 'city', 'region', or 'outlet' would populate this chart." />}
        </Card>
      </div>

      <div className="grid-2">
        {genderCol ? (
          <Card title={`${genderCol.column} Breakdown`} subtitle="Gender or demographic split">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={genderCol.items?.slice(0, 6) || []} barSize={36}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Count" fill="var(--accent2)" radius={[6,6,0,0]}>
                    {(genderCol.items || []).slice(0,6).map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        ) : (
          <Card title={`${typeCol?.column || 'Customer Type'} Bar Chart`} subtitle="Actual records per customer category">
            {typeDist.length > 0 ? (
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={typeDist} barSize={28}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" name="Records" fill="var(--accent2)" radius={[6,6,0,0]}>
                      {typeDist.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : <NoDataState message="No customer type column found." />}
          </Card>
        )}

        <Card title="Numeric Feature Summary" subtitle="Value ranges for spend/income/rating metrics">
          {nums.length > 0 ? (
            <div className="stats-table">
              <div className="st-head"><span>Metric</span><span>Mean</span><span>Min</span><span>Max</span></div>
              {nums.slice(0, 6).map((n: any) => (
                <div className="st-row" key={n.name}>
                  <span className="st-name">{n.name}</span>
                  <b>{typeof n.mean === 'number' ? n.mean.toFixed(2) : n.mean}</b>
                  <span className="muted">{typeof n.min === 'number' ? n.min.toFixed(2) : n.min}</span>
                  <span className="muted">{typeof n.max === 'number' ? n.max.toFixed(2) : n.max}</span>
                </div>
              ))}
            </div>
          ) : <NoDataState message="No numeric columns." />}
        </Card>
      </div>

      <Card title="Customer Insights" subtitle="Data-driven observations">
        <div className="insight-cards">
          <InsightTile icon="🎯" color="#7657f5" title="Top Location" text={locationDist[0] ? `"${locationDist[0].label}" has the most records (${locationDist[0].value.toLocaleString()}) — ${pct(locationDist[0].value, totalRecords)} of all records. Focus resources here.` : 'No location data found. Add a location/outlet column to enable regional analysis.'} />
          <InsightTile icon="⭐" color="#f2c200" title="Customer Composition" text={typeDist[0] ? `"${typeDist[0].label}" is the largest customer segment with ${typeDist[0].value.toLocaleString()} records (${pct(typeDist[0].value, totalType)} of ${typeCol?.column || 'total'}).` : 'No customer type column detected.'} />
          <InsightTile icon="📍" color="#20b26b" title="Rating / Spend" text={ratingCol ? `"${ratingCol.name}" ranges from ${ratingCol.min?.toFixed(2)} to ${ratingCol.max?.toFixed(2)} with avg ${ratingCol.mean?.toFixed(2)}.` : (incomeCol ? `"${incomeCol.name}" ranges ₹${incomeCol.min?.toFixed(2)}–₹${incomeCol.max?.toFixed(2)} with avg ₹${incomeCol.mean?.toFixed(2)}.` : 'No rating/spend column found.')} />
        </div>
      </Card>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   PRODUCT ANALYSIS
══════════════════════════════════════════════════════ */
function ProductAnalysis({ dataset }: { dataset: Dataset }) {
  const nums = dataset.charts?.numeric_distributions || [];
  const cats = dataset.charts?.categorical || [];

  const priceCol = nums.find((n: any) => /price|mrp|cost|amount|sales/i.test(n.name));
  const ratingCol = nums.find((n: any) => /rating|score|review/i.test(n.name));
  const visibilityCol = nums.find((n: any) => /visib|traffic|exposure/i.test(n.name));
  const itemCol = cats.find((c: any) => /item|product|category/i.test(c.column));
  const fatCol = cats.find((c: any) => /fat|content|weight|size|tier/i.test(c.column));

  if (!itemCol) return (
    <div className="content">
      <div className="empty-state large"><Package size={48} color="#e33b46"/><br/><b>Irrelevant Dataset for Product Analysis</b><p>No product/item categorical attributes detected in this dataset.</p></div>
    </div>
  );

  const products = (itemCol?.items || []).slice(0, 10);
  const totalProducts = products.reduce((a: number, d: any) => a + d.value, 0);

  const productPerf = products.map((p: any) => ({
    name: p.label?.length > 16 ? p.label.slice(0,16)+'…' : p.label,
    sales: p.value,
    estRevenue: Math.round(p.value * (priceCol?.mean || 0)),
    rating: ratingCol ? parseFloat(ratingCol.mean?.toFixed(2)) : null,
  }));

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge product-badge"><Package size={13}/> Product Intelligence</div>
        <h2>Product Performance Analysis</h2>
        <p>Item-level sales volumes, pricing analysis, and category performance from <b>{itemCol?.column || 'dataset'}</b>.</p>
      </div>

      <div className="kpi-row">
        <KpiCard icon={Package} label="Product Categories" value={products.length} sub={`${totalProducts.toLocaleString()} total records`} color="#f47b20" />
        <KpiCard icon={TrendingUp} label="Avg Unit Price" value={priceCol ? `₹${priceCol.mean?.toFixed(2)}` : 'N/A'} sub={priceCol ? `Col: ${priceCol.name} | Range ₹${priceCol.min?.toFixed(0)}–₹${priceCol.max?.toFixed(0)}` : 'No price column found'} color="#f2c200" />
        <KpiCard icon={Award} label="Top Product" value={products[0]?.label?.slice(0,14) || 'N/A'} sub={products[0] ? `${products[0].value.toLocaleString()} records` : ''} color="#20b26b" />
        <KpiCard icon={BarChart2} label="Avg Rating" value={ratingCol ? ratingCol.mean?.toFixed(2) : 'N/A'} sub={ratingCol ? `Col: ${ratingCol.name}` : 'No rating column found'} color="#7657f5" />
      </div>

      {products.length > 0 ? (
        <Card title={`Product Rankings — ${itemCol?.column || 'Category'}`} subtitle="Actual transaction counts and estimated revenue per product">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={productPerf} barSize={priceCol ? 20 : 30}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                {priceCol && <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => `₹${fmt(v)}`} />}
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar yAxisId="left" dataKey="sales" name="Records" fill="var(--accent)" radius={[6,6,0,0]}>
                  {productPerf.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                </Bar>
                {priceCol && <Bar yAxisId="right" dataKey="estRevenue" name="Est. Revenue (₹)" fill="var(--accent2)" radius={[6,6,0,0]} opacity={0.6} />}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      ) : (
        <Card title="No Product Column Found" subtitle="">
          <NoDataState message='No column with "item", "product", or "category" found. Upload a dataset with product/item categories.' />
        </Card>
      )}

      <div className="grid-2">
        <Card title={`${itemCol?.column || 'Category'} Mix`} subtitle="Distribution of product categories">
          {products.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={240}>
                <RPieChart>
                  <Pie data={products} cx="50%" cy="50%" outerRadius={85} innerRadius={0} dataKey="value" nameKey="label">
                    {products.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v: any) => [v.toLocaleString(), 'Records']} />
                  <Legend iconType="square" iconSize={9} wrapperStyle={{ fontSize: 10 }} />
                </RPieChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No product/item column found." />}
        </Card>

        <Card title={fatCol ? `${fatCol.column} Distribution` : 'Secondary Category'} subtitle={fatCol ? `Breakdown of ${fatCol.column} values` : 'No secondary categorical column found'}>
          {fatCol?.items?.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={fatCol.items.slice(0, 6)} barSize={32}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Count" fill="var(--accent2)" radius={[6,6,0,0]}>
                    {fatCol.items.slice(0,6).map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No secondary categorical column (fat content, weight, size) found." />}
        </Card>
      </div>

      {nums.length > 0 && (
        <Card title="Pricing & Numeric Overview" subtitle="All numeric column stats — sorted by mean">
          <div className="stats-table">
            <div className="st-head"><span>Feature</span><span>Mean</span><span>Min</span><span>Max</span></div>
            {[...nums].sort((a: any, b: any) => b.mean - a.mean).map((n: any) => (
              <div className="st-row" key={n.name}>
                <span className="st-name">{n.name}</span>
                <b>{typeof n.mean === 'number' ? n.mean.toFixed(2) : n.mean}</b>
                <span className="muted">{typeof n.min === 'number' ? n.min.toFixed(2) : n.min}</span>
                <span className="muted">{typeof n.max === 'number' ? n.max.toFixed(2) : n.max}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Product Insights" subtitle="Data-driven product recommendations">
        <div className="insight-cards">
          <InsightTile icon="🏆" color="#f47b20" title="Top Performer" text={products[0] ? `"${products[0].label}" leads with ${products[0].value.toLocaleString()} records — ${pct(products[0].value, totalProducts)} of all transactions. Ensure consistent stock.` : 'No product column detected.'} />
          <InsightTile icon="💰" color="#f2c200" title="Pricing Insight" text={priceCol ? `"${priceCol.name}" ranges ₹${priceCol.min?.toFixed(2)}–₹${priceCol.max?.toFixed(2)} with avg ₹${priceCol.mean?.toFixed(2)}. Bundle mid-range items to lift average order value.` : 'No price/MRP column found.'} />
          <InsightTile icon="📊" color="#7657f5" title="Category Diversity" text={`${products.length} product categories tracked. Tail categories (bottom 50%) represent ${pct(products.slice(Math.ceil(products.length/2)).reduce((a: number, p: any) => a + p.value, 0), totalProducts)} of volume — evaluate SKU rationalisation.`} />
        </div>
      </Card>
    </div>
  );
}


/* ══════════════════════════════════════════════════════
   OUTLET ANALYSIS
══════════════════════════════════════════════════════ */
function OutletAnalysis({ dataset }: { dataset: Dataset }) {
  const nums = dataset.charts?.numeric_distributions || [];
  const cats = dataset.charts?.categorical || [];

  const outletCol = cats.find((c: any) => /outlet|store|location|city|region|branch/i.test(c.column));
  const sizeCol = cats.find((c: any) => /size|tier|type|level/i.test(c.column) && c !== outletCol);
  const salesCol = nums.find((n: any) => /sales|revenue|amount|total/i.test(n.name))
    || nums.find((n: any) => /price|mrp/i.test(n.name));
  const estabCol = nums.find((n: any) => /year|estab|established|age/i.test(n.name));

  if (!outletCol) return (
    <div className="content">
      <div className="empty-state large"><Store size={48} color="#e33b46"/><br/><b>Irrelevant Dataset for Outlet Analysis</b><p>No location/outlet attributes detected in this dataset.</p></div>
    </div>
  );

  const outlets = (outletCol?.items || []).slice(0, 8);
  const totalOutlets = outlets.length;
  const totalRecs = outlets.reduce((a: number, o: any) => a + o.value, 0);

  const outletData = outlets.map((o: any) => ({
    name: o.label?.length > 14 ? o.label.slice(0,14)+'…' : o.label,
    transactions: o.value,
    estAvgSales: salesCol ? Math.round(o.value * salesCol.mean) : o.value,
    share: totalRecs > 0 ? parseFloat(((o.value / totalRecs) * 100).toFixed(1)) : 0,
  }));

  const sizeData = (sizeCol || cats[0])?.items?.slice(0, 6) || [];

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge outlet-badge"><Store size={13}/> Outlet Intelligence</div>
        <h2>Outlet &amp; Location Analysis</h2>
        <p>Compare performance across <b>{outletCol?.column || 'location'}</b> locations and size tiers from <b>{sizeCol?.column || 'category'}</b>.</p>
      </div>

      <div className="kpi-row">
        <KpiCard icon={Store} label="Outlet Locations" value={totalOutlets} sub={outletCol ? `Col: ${outletCol.column}` : 'No outlet column'} color="#1a9bd6" />
        <KpiCard icon={TrendingUp} label="Top Outlet" value={outlets[0]?.label?.slice(0,14) || 'N/A'} sub={outlets[0] ? `${outlets[0].value.toLocaleString()} records` : ''} color="#20b26b" />
        <KpiCard icon={BarChart2} label="Avg Sale Value" value={salesCol ? `₹${salesCol.mean?.toFixed(2)}` : 'N/A'} sub={salesCol ? `Col: ${salesCol.name}` : 'No sales column'} color="#f2c200" />
        <KpiCard icon={Activity} label="Size / Tier Categories" value={(sizeCol || cats[0])?.items?.length || 0} sub={sizeCol ? `Col: ${sizeCol.column}` : 'No size column'} color="#7657f5" />
      </div>

      {outletData.length > 0 ? (
        <Card title={`Performance by ${outletCol?.column || 'Outlet'}`} subtitle="Actual record counts and estimated revenue — no simulation">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={outletData} barSize={salesCol ? 20 : 30}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => fmt(v)} />
                {salesCol && <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={v => `₹${fmt(v)}`} />}
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar yAxisId="left" dataKey="transactions" name="Records" fill="var(--accent)" radius={[6,6,0,0]}>
                  {outletData.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                </Bar>
                {salesCol && (
                  <Bar yAxisId="right" dataKey="estAvgSales" name="Est. Revenue (₹)" fill="var(--accent2)" radius={[6,6,0,0]} opacity={0.65} />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      ) : (
        <Card title="No Outlet Column Found" subtitle="">
          <NoDataState message='No column with "outlet", "store", "location", or "city" found. Upload a retail dataset to enable outlet analysis.' />
        </Card>
      )}

      <div className="grid-2">
        <Card title={`${(sizeCol || cats[0])?.column || 'Outlet Size'} Distribution`} subtitle="Actual count per tier">
          {sizeData.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={240}>
                <RPieChart>
                  <Pie data={sizeData} cx="50%" cy="50%" outerRadius={90} innerRadius={45} dataKey="value" nameKey="label" label={({ label, percent }: any) => `${label} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                    {sizeData.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v: any) => [v.toLocaleString(), 'Records']} />
                </RPieChart>
              </ResponsiveContainer>
            </div>
          ) : <NoDataState message="No size/tier categorical column found." />}
        </Card>

        <Card title="Outlet Share Ranking" subtitle="% share of total records per outlet">
          {outletData.length > 0 ? (
            <div className="outlet-rank">
              {outletData.map((o: any, i: number) => (
                <div className="rank-row" key={o.name}>
                  <div className="rank-num" style={{ background: PALETTE[i%PALETTE.length]+'22', color: PALETTE[i%PALETTE.length] }}>{i+1}</div>
                  <div className="rank-info">
                    <b>{o.name}</b>
                    <div className="rank-bar-wrap">
                      <div className="rank-bar" style={{ width: `${o.share}%`, background: PALETTE[i%PALETTE.length] }} />
                    </div>
                  </div>
                  <div className="rank-val">{o.share}% <span className="muted">({o.transactions.toLocaleString()})</span></div>
                </div>
              ))}
            </div>
          ) : <NoDataState message="No outlet data to rank." />}
        </Card>
      </div>

      {estabCol && (
        <Card title={`${estabCol.name} — Establishment Year Stats`} subtitle="Numeric summary of outlet age/year column">
          <div className="quality-table">
            {[
              ['Column', estabCol.name, 'good'],
              ['Earliest', estabCol.min?.toFixed(0), 'good'],
              ['Latest', estabCol.max?.toFixed(0), 'good'],
              ['Mean Year', estabCol.mean?.toFixed(0), 'good'],
            ].map(([k, v, cls]) => (
              <div className="qt-row" key={String(k)}>
                <span>{k}</span>
                <div className={`qt-val ${cls}`}>{v}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Outlet Insights" subtitle="Strategic observations from real data">
        <div className="insight-cards">
          <InsightTile icon="🗺️" color="#1a9bd6" title="Top Location" text={outlets[0] ? `"${outlets[0].label}" is the top-performing outlet with ${outlets[0].value.toLocaleString()} records (${pct(outlets[0].value, totalRecs)} of total). Consider replicating its model.` : 'No outlet/location column found in this dataset.'} />
          <InsightTile icon="🏪" color="#20b26b" title="Size Optimisation" text={sizeData.length > 0 ? `${sizeData.length} outlet size tiers in "${(sizeCol || cats[0])?.column}". Top tier "${sizeData[0]?.label}" has ${sizeData[0]?.value?.toLocaleString()} records.` : 'No outlet size/tier column found.'} />
          <InsightTile icon="📉" color="#e33b46" title="Underperformer Alert" text={outlets.length > 1 ? `Bottom outlet "${outlets[outlets.length-1]?.label}" has ${outlets[outlets.length-1]?.value?.toLocaleString()} records — only ${pct(outlets[outlets.length-1]?.value, totalRecs)} of total. Investigate foot-traffic patterns.` : 'Only one outlet location found.'} />
        </div>
      </Card>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   DATA PREPROCESSING
══════════════════════════════════════════════════════ */
function DataPreprocessing({ dataset, setDataset }: { dataset: Dataset; setDataset: (d: Dataset) => void }) {
  const [activeTab, setActiveTab] = useState<'overview'|'missing'|'encode'|'outliers'|'features'>('overview');
  const [searchCol, setSearchCol] = useState('');
  const [sortBy, setSortBy] = useState<'name'|'missing'|'unique'>('missing');
  const [sortDir, setSortDir] = useState<'asc'|'desc'>('desc');
  const [filterType, setFilterType] = useState<'all'|'numeric'|'categorical'>('all');
  const [applied, setApplied] = useState<string[]>([]);
  const [applyingOp, setApplyingOp] = useState<string | null>(null);
  const [applyError, setApplyError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const cols = dataset.columns;
  const totalMissing = cols.reduce((a, c) => a + c.missing, 0);
  const numCols = cols.filter((c: any) => c.dtype.includes('float'));
  const catCols = cols.filter((c: any) => !c.dtype.includes('float'));

  const filtered = useMemo(() => {
    let c = cols;
    if (filterType === 'numeric') c = numCols;
    if (filterType === 'categorical') c = catCols;
    if (searchCol) c = c.filter((col: any) => col.name.toLowerCase().includes(searchCol.toLowerCase()));
    return [...c].sort((a: any, b: any) => {
      const av = a[sortBy] ?? a.name; const bv = b[sortBy] ?? b.name;
      return sortDir === 'asc' ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });
  }, [cols, filterType, searchCol, sortBy, sortDir]);

  const applyStep = async (label: string) => {
    if (applied.includes(label) || applyingOp) return;
    setApplyingOp(label);
    setApplyError('');
    setSuccessMsg('');
    try {
      const result = await preprocessDataset(dataset.dataset_id, [label]);
      setDataset({
        ...dataset,
        dataset_id: result.dataset_id,
        summary: result.summary,
        columns: result.columns,
        charts: result.charts,
        preview: result.preview,
        target_recommendations: result.target_recommendations,
      });
      setApplied(prev => [...prev, label]);
      setSuccessMsg(`✓ Applied! Dataset updated: ${result.rows_after} rows × ${result.columns_after} columns`);
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (e: any) {
      setApplyError(e.message || 'Operation failed');
      setTimeout(() => setApplyError(''), 5000);
    } finally {
      setApplyingOp(null);
    }
  };

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge preprocess-badge"><Filter size={13}/> Preprocessing Pipeline</div>
        <h2>Data Preprocessing</h2>
        <p>Inspect, clean, and transform your dataset. Each step runs on your actual data and updates all modules.</p>
        {successMsg && <div style={{marginTop:10,padding:'10px 16px',background:'color-mix(in srgb,#20b26b 12%,transparent)',color:'#20b26b',borderRadius:10,fontSize:13,fontWeight:600}}>{successMsg}</div>}
        {applyError && <div style={{marginTop:10,padding:'10px 16px',background:'color-mix(in srgb,#e33b46 12%,transparent)',color:'#e33b46',borderRadius:10,fontSize:13}}>{applyError}</div>}
      </div>

      <div className="kpi-row">
        <KpiCard icon={Database} label="Total Columns" value={cols.length} sub={`${numCols.length} num · ${catCols.length} cat`} color="#1a9bd6" />
        <KpiCard icon={AlertCircle} label="Missing Values" value={totalMissing.toLocaleString()} sub={pct(totalMissing, dataset.summary.rows * cols.length) + ' of all cells'} color={totalMissing > 0 ? '#e33b46' : '#20b26b'} />
        <KpiCard icon={Copy} label="Duplicates" value={dataset.summary.duplicates} sub="Already removed on upload" color="#20b26b" />
        <KpiCard icon={CheckCircle2} label="Steps Applied" value={applied.length} sub="Preprocessing steps" color="#7657f5" />
      </div>

      <div className="tab-bar">
        {(['overview','missing','encode','outliers','features'] as const).map(t => (
          <button key={t} className={`tab-btn ${activeTab===t?'active':''}`} onClick={() => setActiveTab(t)}>
            {t === 'overview' && <Layers size={13}/>}
            {t === 'missing' && <AlertCircle size={13}/>}
            {t === 'encode' && <RefreshCw size={13}/>}
            {t === 'outliers' && <Activity size={13}/>}
            {t === 'features' && <Target size={13}/>}
            {t.charAt(0).toUpperCase()+t.slice(1)} Values
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <Card title="Column Inspector" subtitle="Browse, filter and sort all dataset columns">
          <div className="inspector-toolbar">
            <div className="search-box">
              <Search size={14}/><input placeholder="Search columns…" value={searchCol} onChange={e => setSearchCol(e.target.value)} />
            </div>
            <div className="filter-btns">
              {(['all','numeric','categorical'] as const).map(f => (
                <button key={f} className={`filt-btn ${filterType===f?'active':''}`} onClick={() => setFilterType(f)}>{f}</button>
              ))}
            </div>
            <div className="sort-controls">
              Sort:
              {(['name','missing','unique'] as const).map(s => (
                <button key={s} className={`sort-btn ${sortBy===s?'active':''}`} onClick={() => { if(sortBy===s) setSortDir(d => d==='asc'?'desc':'asc'); else { setSortBy(s); setSortDir('desc'); } }}>
                  {s}{sortBy===s && (sortDir==='asc'?<ChevronUp size={11}/>:<ChevronDown size={11}/>)}
                </button>
              ))}
            </div>
          </div>
          <div className="col-grid">
            {filtered.map((c: any) => {
              const isNum = c.dtype.includes('float');
              const missingPct = dataset.summary.rows > 0 ? (c.missing / dataset.summary.rows * 100).toFixed(1) : '0';
              return (
                <div className="col-card" key={c.name}>
                  <div className="col-card-top">
                    <div className={`col-type-badge ${isNum ? 'num' : 'cat'}`}>{isNum ? <Hash size={11}/> : <Type size={11}/>}{isNum ? 'Numeric' : 'Categorical'}</div>
                    <div className={`missing-chip ${c.missing > 0 ? 'has-missing' : 'clean'}`}>{c.missing > 0 ? `${c.missing} missing` : '✓ complete'}</div>
                  </div>
                  <div className="col-name">{c.name}</div>
                  <div className="col-stats">
                    <span><b>{c.unique}</b> unique</span>
                    <span><b>{missingPct}%</b> null</span>
                  </div>
                  {c.missing > 0 && (
                    <div className="missing-bar-wrap">
                      <div className="missing-bar" style={{ width: `${missingPct}%` }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {activeTab === 'missing' && (
        <div className="grid-2">
          <Card title="Missing Value Strategy" subtitle="Choose imputation method per column type">
            <div className="pipeline-steps">
              <PipelineStep num="01" title="Numeric Imputation — Median Fill" desc="Replaces null values in numeric columns with the column median. Robust to outliers vs mean imputation." applied={applied.includes('median_fill')} onApply={() => applyStep('median_fill')} tag="Numeric" loading={applyingOp === 'median_fill'} />
              <PipelineStep num="02" title="Categorical Imputation — Mode Fill" desc="Replaces null values in categorical columns with the most frequent value (mode)." applied={applied.includes('mode_fill')} onApply={() => applyStep('mode_fill')} tag="Categorical" loading={applyingOp === 'mode_fill'} />
              <PipelineStep num="03" title="Forward Fill (Time-ordered)" desc="Propagates the last valid value forward. Best for time-series datasets with sequential records." applied={applied.includes('ffill')} onApply={() => applyStep('ffill')} tag="All" loading={applyingOp === 'ffill'} />
              <PipelineStep num="04" title="Drop Rows with Missing Values" desc="Removes all rows containing any null. Use only if missing rows represent &lt;5% of dataset." applied={applied.includes('drop_missing')} onApply={() => applyStep('drop_missing')} tag="Caution" danger loading={applyingOp === 'drop_missing'} />
            </div>
          </Card>
          <Card title="Missing Value Report" subtitle="Columns with null cells ranked by severity">
            <div className="missing-report">
              {cols.filter((c: any) => c.missing > 0).length === 0
                ? <div className="no-missing"><CheckCircle2 size={32} color="#20b26b"/><b>No missing values!</b><p>Your dataset is complete.</p></div>
                : cols.filter((c: any) => c.missing > 0).sort((a: any, b: any) => b.missing - a.missing).map((c: any) => (
                  <div className="mv-row" key={c.name}>
                    <div className="mv-info">
                      <b>{c.name}</b>
                      <span>{c.missing} missing ({(c.missing/dataset.summary.rows*100).toFixed(1)}%)</span>
                    </div>
                    <div className="mv-bar-wrap">
                      <div className="mv-bar" style={{ width: `${Math.min(100, c.missing/dataset.summary.rows*100)}%` }} />
                    </div>
                  </div>
                ))
              }
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'encode' && (
        <Card title="Encoding Pipeline" subtitle="Transform categorical variables for ML algorithms">
          <div className="pipeline-steps">
            <PipelineStep num="01" title="Label Encoding" desc="Converts each category to an integer (A→0, B→1, C→2). Compact but implies ordinal relationship. Good for tree-based models." applied={applied.includes('label_enc')} onApply={() => applyStep('label_enc')} tag="Categorical" loading={applyingOp === 'label_enc'} />
            <PipelineStep num="02" title="One-Hot Encoding" desc="Creates binary indicator columns for each category value. Eliminates ordinal bias. Used automatically by the backend pipeline." applied={applied.includes('ohe')} onApply={() => applyStep('ohe')} tag="Categorical" loading={applyingOp === 'ohe'} />
            <PipelineStep num="03" title="Min-Max Normalization (0–1 scale)" desc="Scales numeric features to [0,1] range. Essential for gradient-based models like Logistic Regression." applied={applied.includes('minmax')} onApply={() => applyStep('minmax')} tag="Numeric" loading={applyingOp === 'minmax'} />
            <PipelineStep num="04" title="Z-Score Standardization" desc="Transforms features to zero mean and unit variance. Improves convergence for distance-based algorithms." applied={applied.includes('zscore')} onApply={() => applyStep('zscore')} tag="Numeric" loading={applyingOp === 'zscore'} />
          </div>
          <div className="encode-summary">
            <div className="es-row"><span>Categorical columns</span><b>{catCols.length}</b></div>
            <div className="es-row"><span>Numeric columns</span><b>{numCols.length}</b></div>
            <div className="es-row"><span>Estimated encoded features</span><b>{numCols.length + catCols.reduce((a: number, c: any) => a + Math.min(c.unique, 10), 0)}</b></div>
            <div className="es-note"><Info size={12}/> The backend automatically applies Label Encoding + Median/Mode imputation before any model training.</div>
          </div>
        </Card>
      )}

      {activeTab === 'outliers' && (
        <div>
          <Card title="Outlier Detection" subtitle="Identify extreme values using statistical thresholds">
            <div className="pipeline-steps">
              <PipelineStep num="01" title="IQR Method (1.5× fence)" desc="Flags values below Q1−1.5×IQR or above Q3+1.5×IQR as outliers. The most robust method for skewed distributions." applied={applied.includes('iqr')} onApply={() => applyStep('iqr')} tag="Numeric" loading={applyingOp === 'iqr'} />
              <PipelineStep num="02" title="Z-Score (±3σ threshold)" desc="Flags values more than 3 standard deviations from the mean. Works best for approximately normal distributions." applied={applied.includes('zscore_out')} onApply={() => applyStep('zscore_out')} tag="Numeric" loading={applyingOp === 'zscore_out'} />
              <PipelineStep num="03" title="Cap Outliers (Winsorize)" desc="Clips extreme values to the 5th and 95th percentile instead of removing rows. Preserves data volume." applied={applied.includes('winsorize')} onApply={() => applyStep('winsorize')} tag="Numeric" loading={applyingOp === 'winsorize'} />
              <PipelineStep num="04" title="Remove Outlier Rows" desc="Drops all rows containing flagged outliers. Use with caution — may bias the dataset if outliers are informative." applied={applied.includes('drop_outliers')} onApply={() => applyStep('drop_outliers')} tag="Caution" danger loading={applyingOp === 'drop_outliers'} />
            </div>
          </Card>
          <Card title="Numeric Range Overview" subtitle="Identify wide-range columns — candidates for outlier review">
            <div className="stats-table">
              <div className="st-head"><span>Column</span><span>Mean</span><span>Min</span><span>Max</span><span>Range</span></div>
              {(dataset.charts?.numeric_distributions || []).map((n: any) => (
                <div className="st-row" key={n.name}>
                  <span className="st-name">{n.name}</span>
                  <b>{typeof n.mean === 'number' ? n.mean.toFixed(2) : n.mean}</b>
                  <span className="muted">{typeof n.min === 'number' ? n.min.toFixed(2) : n.min}</span>
                  <span className="muted">{typeof n.max === 'number' ? n.max.toFixed(2) : n.max}</span>
                  <span className={`range-badge ${(n.max - n.min) > n.mean * 5 ? 'wide' : 'normal'}`}>{(n.max - n.min).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'features' && (
        <Card title="Feature Engineering" subtitle="Create and select the most impactful features">
          <div className="pipeline-steps">
            <PipelineStep num="01" title="Drop Low-Variance Features" desc="Removes columns where nearly all values are the same. These add noise without predictive signal." applied={applied.includes('low_var')} onApply={() => applyStep('low_var')} tag="Selection" loading={applyingOp === 'low_var'} />
            <PipelineStep num="02" title="Remove Highly Correlated Features" desc="Drops one column from each pair with Pearson correlation > 0.95 to reduce multicollinearity." applied={applied.includes('corr_drop')} onApply={() => applyStep('corr_drop')} tag="Selection" loading={applyingOp === 'corr_drop'} />
            <PipelineStep num="03" title="Log Transform (Skewed Columns)" desc="Applies log(1+x) to heavily right-skewed numeric columns, reducing the effect of large outliers." applied={applied.includes('log_transform')} onApply={() => applyStep('log_transform')} tag="Transform" loading={applyingOp === 'log_transform'} />
            <PipelineStep num="04" title="Interaction Features (A × B)" desc="Creates multiplicative features between numeric column pairs. Can reveal non-linear relationships." applied={applied.includes('interaction')} onApply={() => applyStep('interaction')} tag="Engineering" loading={applyingOp === 'interaction'} />
          </div>
          {dataset.charts?.correlation && (
            <div style={{ marginTop: 20 }}>
              <h4 style={{ marginBottom: 12, fontSize: 13 }}>Correlation Matrix (top numeric features)</h4>
              <div className="corr-matrix">
                {dataset.charts.correlation.columns.map((col: string) => (
                  <div key={col} className="corr-row">
                    <div className="corr-label">{col.slice(0,12)}</div>
                    {dataset.charts.correlation.values[dataset.charts.correlation.columns.indexOf(col)].map((v: number, ci: number) => (
                      <div key={ci} className="corr-cell" style={{ background: v > 0 ? `rgba(32,178,107,${Math.abs(v)})` : `rgba(227,59,70,${Math.abs(v)})`, color: Math.abs(v) > 0.5 ? '#fff' : 'var(--ink)' }} title={`${col} × ${dataset.charts.correlation.columns[ci]}: ${v}`}>
                        {v.toFixed(2)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function PipelineStep({ num, title, desc, applied, onApply, tag, danger, loading }: any) {
  return (
    <div className={`pipeline-step ${applied ? 'done' : ''} ${danger ? 'danger-step' : ''}`}>
      <div className="ps-num">{loading ? <RefreshCw size={14} style={{animation:'spin 1s linear infinite'}} /> : applied ? <CheckCircle2 size={14} /> : num}</div>
      <div className="ps-body">
        <div className="ps-top">
          <b>{title}</b>
          <span className={`ps-tag ${danger ? 'tag-danger' : ''}`}>{tag}</span>
        </div>
        <p>{desc}</p>
      </div>
      <button className={`ps-btn ${applied ? 'applied' : ''}`} onClick={onApply} disabled={applied || loading}>
        {loading ? <>Processing…</> : applied ? <><CheckCircle2 size={12}/>Applied</> : <><Zap size={12}/>Apply</>}
      </button>
    </div>
  );
}


/* ══════════════════════════════════════════════════════
   DATASET EXPLORER
══════════════════════════════════════════════════════ */
function DatasetExplorer({ dataset }: { dataset: Dataset }) {
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const rowsPerPage = 10;
  const cols = dataset.columns;
  const allRows = dataset.preview;
  const filtered = search
    ? allRows.filter((r: any) => Object.values(r).some((v: any) => String(v).toLowerCase().includes(search.toLowerCase())))
    : allRows;
  const pageRows = filtered.slice(page * rowsPerPage, (page+1) * rowsPerPage);

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge explorer-badge"><FileSpreadsheet size={13}/> Data Explorer</div>
        <h2>Dataset Explorer</h2>
        <p>Preview and search through your raw dataset. Showing up to {allRows.length} rows.</p>
      </div>

      <Card title="Data Table" subtitle={`${dataset.filename} — ${cols.length} columns`}
        action={
          <div className="explorer-controls">
            <div className="search-box"><Search size={13}/><input placeholder="Search rows…" value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} /></div>
          </div>
        }>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th className="row-num">#</th>
                {cols.map((c: any) => (
                  <th key={c.name}>
                    <div className="th-inner">
                      <div className={`th-type ${c.dtype.includes('float') ? 'num' : 'cat'}`}>{c.dtype.includes('float') ? <Hash size={9}/> : <Type size={9}/>}</div>
                      {c.name}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((r: any, i: number) => (
                <tr key={i}>
                  <td className="row-num muted">{page * rowsPerPage + i + 1}</td>
                  {cols.map((c: any) => {
                    const v = r[c.name];
                    const isNull = v === null || v === undefined || v === '—';
                    return <td key={c.name} className={isNull ? 'null-cell' : ''}>{isNull ? <span className="null-badge">null</span> : String(v)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-footer">
          <span className="muted">{filtered.length} rows {search ? `(filtered)` : ''}</span>
          <div className="pagination">
            <button disabled={page === 0} onClick={() => setPage(p => p-1)}>←</button>
            <span>Page {page+1} / {Math.max(1, Math.ceil(filtered.length / rowsPerPage))}</span>
            <button disabled={(page+1)*rowsPerPage >= filtered.length} onClick={() => setPage(p => p+1)}>→</button>
          </div>
        </div>
      </Card>

      <div className="grid-2">
        <Card title="Column Summary" subtitle="Per-column statistics">
          <div className="schema-scroll">
            {cols.map((c: any) => (
              <div className="schema-row" key={c.name}>
                <div className={`dtype-badge ${c.dtype.includes('float') ? 'num' : 'cat'}`}>
                  {c.dtype.includes('float') ? <Hash size={10}/> : <Type size={10}/>}
                  {c.dtype.includes('float') ? 'NUM' : 'CAT'}
                </div>
                <div className="schema-info">
                  <b>{c.name}</b>
                  <span>{c.unique} unique · {c.missing} null</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Dataset Stats" subtitle="Quick summary">
          <div className="quality-table">
            {[['Filename', dataset.filename], ['Rows (preview)', allRows.length], ['Columns', cols.length], ['Numeric', dataset.summary.numeric_columns], ['Categorical', dataset.summary.categorical_columns], ['Total missing', dataset.summary.missing_values]].map(([k, v]) => (
              <div className="qt-row" key={String(k)}><span>{k}</span><div className="qt-val good">{v}</div></div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   ML STUDIO
══════════════════════════════════════════════════════ */
function MLStudio({ dataset, setModel, model, setPrediction, prediction, onUpload }: any) {
  const [target, setTarget] = useState(dataset.columns[0]?.name || '');
  const [task, setTask] = useState('auto');
  const [testSize, setTestSize] = useState('0.2');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const [customInputs, setCustomInputs] = useState<Record<string, any>>({});
  const [customResult, setCustomResult] = useState<any>(null);
  const [customBusy, setCustomBusy] = useState(false);
  const [aiInsights, setAiInsights] = useState<any>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiPoweredBy, setAiPoweredBy] = useState('');

  useMemo(() => {
    if (model?.sample_defaults) setCustomInputs(model.sample_defaults);
  }, [model]);

  const trainNow = async () => {
    setBusy(true); setMsg(''); setAiInsights(null);
    try {
      const m = await train({ dataset_id: dataset.dataset_id, target, task, test_size: parseFloat(testSize), random_state: 42 });
      setModel(m); setMsg(`✓ Best model: ${m.best_algorithm}`);
      if (m.sample_defaults) setCustomInputs(m.sample_defaults);
      // Auto-fetch model insights after training
      setAiLoading(true);
      try {
        const ai = await getAIInsights({ model_id: m.model_id, dataset_id: dataset.dataset_id });
        setAiInsights(ai.ai_insights);
        setAiPoweredBy(ai.powered_by);
      } catch { /* insights are optional */ }
      finally { setAiLoading(false); }
    } catch (e: any) { setMsg(`✗ ${e.message}`); }
    finally { setBusy(false); }
  };

  const predNow = async (file: File) => {
    setBusy(true);
    try {
      const d = await uploadCSV(file);
      const p = await predict({ model_id: model.model_id, dataset_id: d.dataset_id });
      setPrediction({ ...p, filename: d.filename });
      setMsg(`✓ Generated ${p.count} predictions from ${d.filename}`);
    } catch (e: any) { setMsg(`✗ ${e.message}`); }
    finally { setBusy(false); }
  };

  const runCustomPredict = async () => {
    if (!model) return;
    setCustomBusy(true);
    try {
      const res = await predictCustom({ model_id: model.model_id, inputs: customInputs });
      setCustomResult(res);
    } catch (e: any) {
      alert(`Custom prediction error: ${e.message}`);
    } finally {
      setCustomBusy(false);
    }
  };

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge ml-badge"><BrainCircuit size={13}/> ML Studio</div>
        <h2>Machine Learning Studio</h2>
        <p>Train real ML models (Decision Tree, Random Forest, Linear/Logistic Regression) using pure-Python algorithms. Compare results and generate predictions.</p>
      </div>

      <div className="kpi-row">
        <KpiCard icon={Database} label="Training Data Rows" value={dataset.summary.rows.toLocaleString()} sub={`${dataset.summary.columns} total columns`} color="#1a9bd6" />
        <KpiCard icon={Target} label="Default Target" value={dataset.target_recommendations?.[0]?.column || target} sub="Recommended Target" color="#e33b46" />
        <KpiCard icon={Hash} label="Numeric Features" value={dataset.summary.numeric_columns} sub="Used for continuous math" color="#f2c200" />
        <KpiCard icon={Type} label="Categorical Features" value={dataset.summary.categorical_columns} sub="One-hot encoded automatically" color="#7657f5" />
      </div>

      <div className="studio-layout">
        {/* Config panel */}
        <Card title="Configure Training" subtitle="Select target and task type">
          <div className="form-group">
            <label className="form-label">Prediction Target Column</label>
            <select className="form-select" value={target} onChange={e => setTarget(e.target.value)}>
              {dataset.columns.map((c: any) => <option key={c.name} value={c.name}>{c.name} ({c.dtype})</option>)}
            </select>
            <p className="form-hint">The column the model will learn to predict.</p>
          </div>
          <div className="form-group">
            <label className="form-label">Task Type</label>
            <select className="form-select" value={task} onChange={e => setTask(e.target.value)}>
              <option value="auto">Auto Detect</option>
              <option value="regression">Regression (numeric target)</option>
              <option value="classification">Classification (category target)</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Test Split: {(parseFloat(testSize)*100).toFixed(0)}%</label>
            <input type="range" min="0.1" max="0.4" step="0.05" value={testSize} onChange={e => setTestSize(e.target.value)} className="form-range" />
            <div className="range-labels"><span>10%</span><span>40%</span></div>
          </div>
          <button className={`btn-primary full ${busy?'loading':''}`} disabled={busy || !target} onClick={trainNow}>
            <BrainCircuit size={16}/>{busy ? 'Training models…' : 'Train & Compare All Models'}
          </button>
          {msg && <div className={`train-msg ${msg.startsWith('✓')?'success':'error'}`}>{msg}</div>}
        </Card>

        {/* Algorithms info */}
        <Card title="Implemented Algorithms" subtitle="All real from-scratch Python implementations">
          <div className="algo-list">
            {[
              { name: 'Decision Tree (CART)', type: 'Both', desc: 'Gini impurity for classification, MSE variance for regression. Recursive binary splitting.' },
              { name: 'Random Forest', type: 'Both', desc: 'Bootstrap bagging + random feature subsets (√n features). Ensemble of 50 trees.' },
              { name: 'Linear Regression', type: 'Regression', desc: 'OLS normal equation with L2 ridge regularisation (β = (XᵀX+λI)⁻¹Xᵀy).' },
              { name: 'Logistic Regression', type: 'Classification', desc: 'Gradient descent sigmoid + one-vs-rest multi-class strategy.' },
            ].map(a => (
              <div className="algo-card" key={a.name}>
                <div className="algo-top"><b>{a.name}</b><span className={`algo-tag ${a.type.toLowerCase().replace(' ','')}`}>{a.type}</span></div>
                <p>{a.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Model results & Insights */}
      {model && (
        <>
          <div className="model-result-banner">
            <div className="mrb-left">
              <CheckCircle2 size={28} color="#20b26b" />
              <div><b>Best Model: {model.best_algorithm}</b><span>Target: {model.target} · Task: {model.task}</span></div>
            </div>
            <div className="mrb-right">
              <div className="mrb-stat"><b>{model.target}</b><span>Target</span></div>
              <div className="mrb-stat"><b>{model.task}</b><span>Task</span></div>
            </div>
          </div>

          <ModelAnalysis model={model} inline />

          {/* Deep Insights — auto-loaded after training */}
          {aiLoading && (
            <div style={{ display:'flex', alignItems:'center', gap:14, padding:'20px 24px', background:'var(--canvas)', borderRadius:16, border:'1px solid var(--line)', marginTop:20 }}>
              <div style={{ width:20, height:20, borderRadius:'50%', border:'3px solid var(--line)', borderTopColor:'var(--accent)', animation:'spin 0.8s linear infinite', flexShrink:0 }} />
              <span style={{ color:'var(--muted)', fontSize:14 }}>✨ Generating deep insights from your model…</span>
            </div>
          )}
          {aiInsights && !aiLoading && (
            <AIInsightsPanel insights={aiInsights} poweredBy={aiPoweredBy} model={model} />
          )}

          {/* Interactive Attribute Scenario Predictor */}
          <Card title="🎯 Attribute Scenario Predictor" subtitle="Adjust any feature values to simulate what the model predicts — test client, product, pricing, or outlet scenarios live">
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 14, marginBottom: 20 }}>
                {(model.feature_columns || []).slice(0, 9).map((col: string) => (
                  <div key={col} style={{ display:'flex', flexDirection:'column', gap:5 }}>
                    <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', textTransform:'uppercase', letterSpacing:'0.04em' }}>{col}</label>
                    <input
                      type="text"
                      style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--line)', background: 'var(--canvas)', color: 'var(--ink)', fontSize:13, outline:'none' }}
                      value={customInputs[col] !== undefined ? customInputs[col] : ''}
                      onChange={e => setCustomInputs({ ...customInputs, [col]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap:'wrap' }}>
                <button className={`btn-primary ${customBusy ? 'loading' : ''}`} onClick={runCustomPredict} disabled={customBusy}>
                  <Zap size={16} /> Simulate Prediction
                </button>
                {customResult && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 22px', background: 'linear-gradient(135deg, rgba(32,178,107,0.12), rgba(32,178,107,0.04))', borderRadius: 12, border: '1.5px solid #20b26b' }}>
                    <Sparkles size={20} color="#20b26b" />
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--muted)', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em' }}>Predicted {model.target}</div>
                      <div style={{ fontSize: 24, fontWeight: 800, color: '#20b26b', lineHeight:1.2 }}>{String(customResult.prediction)}</div>
                      <div style={{ fontSize: 11, color:'var(--muted)', marginTop:2 }}>via {customResult.algorithm}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </>
      )}



    </div>
  );
}

/* ══════════════════════════════════════════════════════
   INSIGHTS PANEL
══════════════════════════════════════════════════════ */
function AIInsightsPanel({ insights, poweredBy, model }: { insights: any; poweredBy: string; model: any }) {
  const [activeTab, setActiveTab] = useState<'sales'|'product'|'client'|'recommendations'>('sales');
  const impColor = (impact: string) => impact==='High' ? '#e33b46' : impact==='Medium' ? '#f47b20' : '#20b26b';
  const verdictColor = (v: string) => v==='Excellent'?'#20b26b':v==='Good'?'#1a9bd6':v==='Fair'?'#f47b20':'#e33b46';

  return (
    <section style={{ marginTop: 24, borderRadius: 20, border: '1px solid var(--line)', overflow: 'hidden', background: 'var(--card)' }}>
      {/* Header */}
      <div style={{ padding: '20px 24px 16px', borderBottom: '1px solid var(--line)', background: 'linear-gradient(135deg, var(--canvas) 0%, var(--card) 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Sparkles size={20} color="var(--accent)" />
            <b style={{ fontSize: 16 }}>Deep Insights</b>
          </div>
          <div style={{ fontSize: 11, color: 'var(--muted)', padding: '4px 10px', borderRadius: 20, border: '1px solid var(--line)', display:'flex', alignItems:'center', gap:5 }}>
            <span style={{ width:6, height:6, borderRadius:'50%', background: poweredBy.includes('gemini') ? '#20b26b' : '#f47b20', display:'inline-block' }} />
            {poweredBy.includes('gemini') ? 'Powered by Gemini' : 'Rule Engine'}
          </div>
        </div>
        {/* Executive Summary */}
        {insights.executive_summary && (
          <p style={{ fontSize: 13, color: 'var(--ink)', marginTop: 12, lineHeight: 1.6, padding: '12px 16px', background: 'color-mix(in srgb, var(--accent) 8%, transparent)', borderRadius: 10, borderLeft: '3px solid var(--accent)' }}>
            {insights.executive_summary}
          </p>
        )}
        {/* Model Quality Badge */}
        {insights.model_quality && (
          <div style={{ marginTop: 12, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ padding: '6px 14px', borderRadius: 20, background: `${verdictColor(insights.model_quality.verdict)}22`, color: verdictColor(insights.model_quality.verdict), fontWeight: 700, fontSize: 12, border: `1px solid ${verdictColor(insights.model_quality.verdict)}44` }}>
              {insights.model_quality.verdict} Model
            </div>
            <div style={{ padding: '6px 14px', borderRadius: 20, background: 'var(--canvas)', color: 'var(--ink)', fontSize: 12, border: '1px solid var(--line)' }}>
              🎯 {insights.model_quality.confidence} confidence
            </div>
            {insights.model_quality.key_strength && (
              <div style={{ padding: '6px 14px', borderRadius: 20, background: 'var(--canvas)', color: 'var(--muted)', fontSize: 12, border: '1px solid var(--line)', maxWidth: 320, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                ✅ {insights.model_quality.key_strength}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--line)', background: 'var(--canvas)' }}>
        {([['sales','📈 Sales','sales_insights'],['product','📦 Product','product_insights'],['client','👥 Clients','client_insights'],['recommendations','🏆 Top Actions','top_recommendations']] as const).map(([tab, label, key]) => (
          <button key={tab} onClick={() => setActiveTab(tab as any)}
            style={{ padding: '12px 20px', fontSize: 12, fontWeight: 600, border: 'none', cursor:'pointer', background:'transparent', color: activeTab===tab ? 'var(--accent)' : 'var(--muted)', borderBottom: activeTab===tab ? '2px solid var(--accent)' : '2px solid transparent', transition:'all 0.2s' }}>
            {label} {Array.isArray(insights[key]) ? `(${insights[key].length})` : ''}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={{ padding: 24 }}>
        {activeTab !== 'recommendations' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {(insights[activeTab === 'sales' ? 'sales_insights' : activeTab === 'product' ? 'product_insights' : 'client_insights'] || []).map((item: any, i: number) => (
              <div key={i} style={{ padding: '16px 20px', borderRadius: 12, border: '1px solid var(--line)', background: 'var(--canvas)', position:'relative', overflow:'hidden' }}>
                <div style={{ position:'absolute', left:0, top:0, bottom:0, width:4, background: impColor(item.impact), borderRadius:'12px 0 0 12px' }} />
                <div style={{ paddingLeft: 8 }}>
                  <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:6 }}>
                    <b style={{ fontSize: 14 }}>{item.title}</b>
                    <span style={{ fontSize: 10, fontWeight: 700, padding: '3px 8px', borderRadius: 12, background: `${impColor(item.impact)}22`, color: impColor(item.impact), textTransform:'uppercase', letterSpacing:'0.05em' }}>
                      {item.impact} Impact
                    </span>
                  </div>
                  <p style={{ fontSize: 13, color: 'var(--ink)', margin: '0 0 10px', lineHeight: 1.55 }}>{item.insight}</p>
                  {item.action && (
                    <div style={{ fontSize: 12, color: 'var(--muted)', display:'flex', alignItems:'center', gap:6, padding:'8px 12px', background:'var(--card)', borderRadius:8 }}>
                      <Zap size={12} color="var(--accent)" />
                      <b>Action:</b> {item.action}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'recommendations' && insights.top_recommendations && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {insights.top_recommendations.map((rec: any, i: number) => (
              <div key={i} style={{ display:'flex', gap:16, padding: '16px 20px', borderRadius: 12, border: '1px solid var(--line)', background: 'var(--canvas)', alignItems:'flex-start' }}>
                <div style={{ width:32, height:32, borderRadius:'50%', background:'var(--accent)', color:'#fff', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:800, fontSize:14, flexShrink:0 }}>
                  {rec.priority}
                </div>
                <div>
                  <b style={{ fontSize: 14 }}>{rec.title}</b>
                  <p style={{ fontSize: 13, color: 'var(--ink)', margin: '6px 0 0', lineHeight: 1.55 }}>{rec.detail}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Predictions Summary */}
        {insights.predictions_summary && (
          <div style={{ marginTop: 20, padding: '16px 20px', borderRadius: 12, background: 'color-mix(in srgb, var(--accent) 6%, transparent)', border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)' }}>
            <b style={{ fontSize: 13, display:'block', marginBottom:6 }}>📊 {insights.predictions_summary.headline}</b>
            <p style={{ fontSize: 12, color: 'var(--muted)', margin: 0, lineHeight: 1.6 }}>{insights.predictions_summary.details}</p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════
   MODEL ANALYSIS
══════════════════════════════════════════════════════ */
function ModelAnalysis({ model, inline }: { model: any; inline?: boolean }) {
  if (!model) return (
    <div className={inline ? '' : 'content'}>
      <div className="empty-state large"><Gauge size={48}/><b>No trained model yet</b><p>Go to ML Studio and train a model first.</p></div>
    </div>
  );

  const isClass = model.task === 'classification';
  const good = model.results.filter((r: any) => !r.error);
  const best = model.best_algorithm;

  const chartData = good.map((r: any) => ({
    name: r.algorithm.replace(' Classifier','').replace(' Regressor',''),
    score: isClass ? parseFloat((r.accuracy * 100).toFixed(1)) : parseFloat((r.r2 * 100).toFixed(1)),
    secondary: isClass ? parseFloat((r.f1 * 100).toFixed(1)) : parseFloat(r.rmse),
  }));

  return (
    <div className={inline ? '' : 'content'}>
      {model.train_size && (
        <div className="model-split-info" style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 13, background: 'var(--canvas)', padding: '12px 16px', borderRadius: '12px', border: '1px solid var(--line)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Database size={14} color="var(--muted)"/> <b>Training Set:</b> {model.train_size.toLocaleString()} rows</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Target size={14} color="var(--muted)"/> <b>Test Set:</b> {model.test_size_count.toLocaleString()} rows</span>
        </div>
      )}

      {model.insights && model.insights.length > 0 && (
        <div className="insights-banner" style={{ marginTop: inline ? 20 : 0 }}>
          <div className="ib-header"><Sparkles size={16} /> <b>Model Insights</b></div>
          <div className="ib-list">
            {model.insights.map((ins: any, i: number) => (
              <div className={`ib-item ${ins.type}`} key={i}>
                <div className="ib-icon">
                  {ins.type === 'success' ? <CheckCircle2 size={16} /> :
                   ins.type === 'error' ? <XCircle size={16} /> :
                   ins.type === 'warn' ? <AlertTriangle size={16} /> :
                   <Info size={16} />}
                </div>
                <div className="ib-content">
                  <b>{ins.title}</b>
                  <p>{ins.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid-2" style={{ marginTop: 20 }}>
        <Card title="Algorithm Comparison" subtitle={`Held-out test performance — ${isClass ? 'Accuracy %' : 'R² score × 100'}`}>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData} barSize={36} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} domain={isClass ? [0, 100] : ['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="score" name={isClass ? 'Accuracy %' : 'R² ×100'} radius={[6,6,0,0]}>
                  {chartData.map((d: any, i: number) => (
                    <Cell key={i} fill={d.name.includes(best.replace(' Classifier','').replace(' Regressor','')) ? '#20b26b' : 'var(--accent)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Detailed Metrics" subtitle="Full scorecard per algorithm">
          <div className="metrics-table">
            <div className="mt-head">
              <span>Algorithm</span>
              <span>{isClass ? 'Accuracy' : 'R²'}</span>
              <span>{isClass ? 'F1' : 'RMSE'}</span>
              <span>{isClass ? 'Precision' : 'MAE'}</span>
            </div>
            {model.results.map((r: any) => (
              <div className={`mt-row ${r.algorithm === best ? 'best' : ''}`} key={r.algorithm}>
                <span className="mt-name">{r.algorithm}{r.algorithm === best && <Award size={11} color="#20b26b" style={{ marginLeft: 4 }} />}</span>
                {r.error
                  ? <span className="mt-error" style={{ gridColumn: '2/-1' }}>{r.error}</span>
                  : <>
                    <b>{isClass ? `${(r.accuracy*100).toFixed(1)}%` : r.r2}</b>
                    <span>{isClass ? (r.f1*100).toFixed(1)+'%' : r.rmse}</span>
                    <span>{isClass ? (r.precision*100).toFixed(1)+'%' : r.mae}</span>
                  </>
                }
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid-2">
        {model.feature_importance?.length > 0 && (
          <Card title="Feature Importance" subtitle={`Top signals used by ${model.best_algorithm}`}>
            <div className="fi-grid" style={{ maxHeight: 300, overflowY: 'auto', paddingRight: 8 }}>
              {model.feature_importance.map((x: any, i: number) => (
                <div className="fi-row" key={x.feature}>
                  <span className="fi-rank">#{i+1}</span>
                  <span className="fi-name">{x.feature}</span>
                  <div className="fi-bar-wrap">
                    <div className="fi-bar" style={{ width: `${Math.max(2, Math.min(100, x.importance * 300))}%`, background: PALETTE[i % PALETTE.length] }} />
                  </div>
                  <b className="fi-val">{x.importance}</b>
                </div>
              ))}
            </div>
          </Card>
        )}

        {isClass && model.confusion_matrix && (
          <Card title="Confusion Matrix" subtitle="Actual vs Predicted Classes (Test Set)">
            <div className="cm-wrap" style={{ overflowX: 'auto', paddingBottom: 10 }}>
              <table className="cm-table">
                <thead>
                  <tr>
                    <th></th>
                    {model.confusion_matrix.classes.map((c: string) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {model.confusion_matrix.classes.map((c: string, i: number) => (
                    <tr key={c}>
                      <th>{c}</th>
                      {model.confusion_matrix.matrix[i].map((v: number, j: number) => (
                        <td key={j} className={i === j ? 'cm-correct' : v > 0 ? 'cm-incorrect' : 'cm-zero'}>
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="cm-legend">Rows: Actual • Columns: Predicted</div>
            </div>
          </Card>
        )}

        {isClass && model.per_class_metrics && (
          <Card title="Per-Class Performance" subtitle="Precision, Recall, and F1-Score per Class">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={model.per_class_metrics} barSize={16} margin={{ top: 10, right: 10, bottom: 20, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="class" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} domain={[0, 1]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 10, bottom: 0 }} />
                  <Bar dataKey="precision" name="Precision" fill="#20b26b" radius={[4,4,0,0]} />
                  <Bar dataKey="recall" name="Recall" fill="#1a9bd6" radius={[4,4,0,0]} />
                  <Bar dataKey="f1" name="F1 Score" fill="#7657f5" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {!isClass && model.actual_vs_predicted && (
          <Card title="Actual vs Predicted" subtitle="Sample of test set predictions">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="actual" type="number" name="Actual" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <YAxis dataKey="predicted" type="number" name="Predicted" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                  <Scatter name="Predictions" data={model.actual_vs_predicted} fill="#20b26b" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {!isClass && model.residuals && (
          <Card title="Residuals Distribution" subtitle="Prediction Errors (Actual - Predicted)">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={model.residuals.map((r: any, i: number) => ({ id: i, error: r.residual }))} barSize={6} margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="id" tick={false} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'transparent' }} />
                  <Bar dataKey="error" name="Residual Error">
                    {model.residuals.map((r: any, i: number) => (
                      <Cell key={i} fill={r.residual > 0 ? '#20b26b' : '#e33b46'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {model.prediction_distribution && (
          <Card title="Prediction Distribution" subtitle="How predictions are distributed">
            <div className="chart-wrap" style={{ display: 'flex' }}>
              <ResponsiveContainer width="60%" height={260}>
                <RPieChart>
                  <Pie data={model.prediction_distribution} dataKey="count" nameKey="label" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2}>
                    {model.prediction_distribution.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </RPieChart>
              </ResponsiveContainer>
              <div className="pie-legend">
                {model.prediction_distribution.slice(0,6).map((d: any, i: number) => (
                  <div className="pl-item" key={d.label}>
                    <span className="pl-dot" style={{ background: PALETTE[i % PALETTE.length] }} />
                    <span className="pl-label">{d.label}</span>
                    <span className="pl-val">{d.pct}%</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   SETTINGS
══════════════════════════════════════════════════════ */
function SettingsPage({ theme, setTheme, dataset, model, onRestart }: {
  theme: Theme; setTheme: (t: Theme) => void;
  dataset: Dataset | null; model: any; onRestart: () => void;
}) {
  const [showConfirm, setShowConfirm] = useState(false);
  const themes: [Theme, string, string, string][] = [
    ['sunny', '☀️ Sunny Yellow', 'White canvas with warm yellow accents', '#f2c200'],
    ['dark', '🌙 Midnight Dark', 'Deep dark with golden highlights', '#f4c20d'],
    ['purple', '💜 Royal Purple', 'Clean white with vivid purple', '#7657f5'],
    ['red', '❤️ Ruby Red', 'Crisp white with bold red accents', '#e33b46'],
  ];

  return (
    <div className="content">
      <div className="section-hero">
        <div className="section-badge settings-badge"><Settings size={13}/> Preferences</div>
        <h2>Settings</h2>
        <p>Customise appearance and manage session data.</p>
      </div>

      <Card title="Appearance — Theme" subtitle="Choose your workspace colour palette">
        <div className="theme-grid">
          {themes.map(([id, name, desc, color]) => (
            <button key={id} className={`theme-tile ${theme === id ? 'selected' : ''}`} onClick={() => setTheme(id)}>
              <div className="theme-swatch" style={{ background: `linear-gradient(135deg, var(--card) 40%, ${color})` }}>
                <div className="swatch-preview">
                  <div style={{ background: color, width: 32, height: 32, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 }}>
                    {name.split(' ')[0]}
                  </div>
                </div>
              </div>
              <div className="theme-info">
                <b>{name.split(' ').slice(1).join(' ')}</b>
                <span>{desc}</span>
              </div>
              {theme === id && <CheckCircle2 size={16} color={color} className="theme-check" />}
            </button>
          ))}
        </div>
      </Card>

      <Card title="Session Management" subtitle="Clear dataset, models, and predictions to start fresh">
        <div className="session-info">
          <div className="session-status-row">
            <div className={`session-dot ${dataset ? 'active' : 'inactive'}`} />
            <span>Dataset: <b>{dataset ? dataset.filename : 'None loaded'}</b></span>
          </div>
          {dataset && (
            <div className="session-detail">
              {dataset.summary.rows.toLocaleString()} rows · {dataset.summary.columns} columns · {dataset.summary.memory_kb} KB
            </div>
          )}
          <div className="session-status-row" style={{ marginTop: 8 }}>
            <div className={`session-dot ${model ? 'active' : 'inactive'}`} />
            <span>Trained Model: <b>{model ? model.best_algorithm : 'None'}</b></span>
          </div>
          {model && (
            <div className="session-detail">
              Target: {model.target} · Task: {model.task}
            </div>
          )}
        </div>

        {!showConfirm ? (
          <button className="btn-danger-outline" onClick={() => setShowConfirm(true)} disabled={!dataset && !model}>
            <RotateCcw size={15} /> Clear All Data &amp; Restart Session
          </button>
        ) : (
          <div className="confirm-box">
            <AlertTriangle size={20} color="#e33b46" />
            <div>
              <b>Are you sure?</b>
              <p>This will clear the loaded dataset, trained model, and all predictions. You cannot undo this.</p>
            </div>
            <div className="confirm-actions">
              <button className="btn-cancel" onClick={() => setShowConfirm(false)}>Cancel</button>
              <button className="btn-danger" onClick={() => { onRestart(); setShowConfirm(false); }}>
                <Trash2 size={14} /> Yes, Clear Everything
              </button>
            </div>
          </div>
        )}
      </Card>

      <Card title="System Architecture" subtitle="Technology stack overview">
        <div className="arch-grid">
          {[
            { label: 'Frontend', tech: 'React 18 + TypeScript', sub: 'Vite dev server on :5173', icon: '⚛️' },
            { label: 'Backend', tech: 'FastAPI (Python 3.13)', sub: 'Uvicorn ASGI on :8000', icon: '🐍' },
            { label: 'ML Engine', tech: 'Pure-Python stdlib', sub: 'CART, RF, LR — no sklearn', icon: '🤖' },
            { label: 'Storage', tech: 'File-based (pickle/CSV)', sub: 'Models saved to /models/', icon: '💾' },
          ].map(a => (
            <div className="arch-card" key={a.label}>
              <div className="arch-icon">{a.icon}</div>
              <div><span className="arch-label">{a.label}</span><b>{a.tech}</b><small>{a.sub}</small></div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="API Endpoints" subtitle="Backend REST API reference">
        <div className="api-table">
          {[
            ['POST', '/api/datasets/upload', 'Upload CSV and get dataset summary'],
            ['POST', '/api/ml/train', 'Train and compare all algorithms'],
            ['POST', '/api/ml/predict', 'Generate predictions on new data'],
            ['GET', '/', 'Health check — returns engine status'],
            ['GET', '/docs', 'Interactive Swagger API documentation'],
          ].map(([method, path, desc]) => (
            <div className="api-row" key={path}>
              <span className={`method-badge ${method.toLowerCase()}`}>{method}</span>
              <code>{path}</code>
              <span className="api-desc">{desc}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

export default App;


