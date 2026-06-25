// Live curation dashboard. See TDD §5, FR8.
// Connects to /ws/metrics (FunnelStats deltas, SSE fallback) and renders the funnel,
// reject reasons, quality-score histogram, throughput-over-time, and KPI cards.
// Backend-agnostic: works against the FastAPI (Python) or Axum (Rust) runtime.
import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Metrics = {
  ingested: number;
  kept: number;
  rejected_by_stage: Record<string, number>;
  rejected_by_reason: Record<string, number>;
  retention_rate: number;
  docs_per_sec: number;
  embed_calls: number;
  chat_calls: number;
  est_cost_usd: number;
  p50_stage_latency_ms: Record<string, number>;
  quality_hist: { bin: number; count: number }[];
};

type Point = { t: number; rate: number };

function useMetrics() {
  const [m, setM] = useState<Metrics | null>(null);
  const [tput, setTput] = useState<Point[]>([]);
  const [connected, setConnected] = useState(false);
  const prev = useRef<{ t: number; ingested: number } | null>(null);
  const start = useRef<number>(Date.now());

  useEffect(() => {
    let ws: WebSocket | null = null;
    let es: EventSource | null = null;
    let usingSse = false;

    const onPayload = (data: Metrics) => {
      setM(data);
      const now = Date.now();
      if (prev.current) {
        const dt = (now - prev.current.t) / 1000;
        if (dt > 0) {
          const rate = Math.max(0, (data.ingested - prev.current.ingested) / dt);
          const t = Math.round((now - start.current) / 1000);
          setTput((h) => [...h.slice(-59), { t, rate }]);
        }
      }
      prev.current = { t: now, ingested: data.ingested };
    };

    const startSse = () => {
      usingSse = true;
      es = new EventSource("/sse/metrics");
      es.onopen = () => setConnected(true);
      es.onmessage = (e) => onPayload(JSON.parse(e.data));
      es.onerror = () => setConnected(false);
    };

    const proto = location.protocol === "https:" ? "wss" : "ws";
    try {
      ws = new WebSocket(`${proto}://${location.host}/ws/metrics`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => onPayload(JSON.parse(e.data));
      ws.onclose = () => {
        setConnected(false);
        if (!usingSse) startSse(); // fall back once the socket drops
      };
      ws.onerror = () => ws?.close();
    } catch {
      startSse();
    }

    return () => {
      ws?.close();
      es?.close();
    };
  }, []);

  return { m, tput, connected };
}

function Kpi({ label, value, green }: { label: string; value: string; green?: boolean }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className={"value" + (green ? " green" : "")}>{value}</div>
    </div>
  );
}

const tooltipStyle = {
  background: "#1b2434",
  border: "1px solid #243049",
  borderRadius: 8,
  color: "#e6edf6",
  fontSize: 12,
};

export default function App() {
  const { m, tput, connected } = useMetrics();

  const funnel = m
    ? (() => {
        const h = m.rejected_by_stage.heuristic || 0;
        const q = m.rejected_by_stage.quality || 0;
        const d = m.rejected_by_stage.dedup || 0;
        return [
          { stage: "ingested", n: m.ingested },
          { stage: "post-heuristic", n: m.ingested - h },
          { stage: "post-quality", n: m.ingested - h - q },
          { stage: "post-dedup", n: m.ingested - h - q - d },
          { stage: "kept", n: m.kept },
        ];
      })()
    : [];

  const reasons = m
    ? Object.entries(m.rejected_by_reason)
        .map(([reason, count]) => ({ reason, count }))
        .sort((a, b) => b.count - a.count)
    : [];

  const hist = m ? m.quality_hist.map((x) => ({ bin: x.bin.toFixed(2), count: x.count })) : [];
  const p50 = m ? Math.max(0, ...Object.values(m.p50_stage_latency_ms)) : 0;

  return (
    <div className="app">
      <div className="topbar">
        <div>
          <h1>Curatio — live curation funnel</h1>
          <div className="sub">real-time training-data curation · heuristics → quality → dedup</div>
        </div>
        <div className="status">
          <span className={"dot" + (connected ? " live" : "")} />
          {connected ? "live" : "disconnected"}
        </div>
      </div>

      <div className="kpis">
        <Kpi label="ingested" value={m ? m.ingested.toLocaleString() : "—"} />
        <Kpi label="kept" value={m ? m.kept.toLocaleString() : "—"} green />
        <Kpi label="retention" value={m ? `${(m.retention_rate * 100).toFixed(0)}%` : "—"} />
        <Kpi label="embed cost" value={m ? `$${m.est_cost_usd.toFixed(4)}` : "—"} />
        <Kpi label="p50 latency" value={m ? `${p50.toFixed(2)} ms` : "—"} />
      </div>

      <div className="grid">
        <div className="panel wide">
          <h2>Funnel retention</h2>
          {m ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={funnel} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#243049" vertical={false} />
                <XAxis dataKey="stage" stroke="#8a99b3" fontSize={12} />
                <YAxis stroke="#8a99b3" fontSize={12} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff08" }} />
                <Bar dataKey="n" radius={[4, 4, 0, 0]}>
                  {funnel.map((_, i) => (
                    <Cell key={i} fill={i === funnel.length - 1 ? "#34d399" : "#4f8cff"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">waiting for the first metrics frame…</div>
          )}
        </div>

        <div className="panel">
          <h2>Rejections by reason</h2>
          {reasons.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={reasons} layout="vertical" margin={{ top: 4, right: 12, left: 24, bottom: 0 }}>
                <CartesianGrid stroke="#243049" horizontal={false} />
                <XAxis type="number" stroke="#8a99b3" fontSize={12} allowDecimals={false} />
                <YAxis type="category" dataKey="reason" stroke="#8a99b3" fontSize={11} width={90} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff08" }} />
                <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">no rejections yet</div>
          )}
        </div>

        <div className="panel">
          <h2>Quality-score distribution</h2>
          {hist.some((h) => h.count > 0) ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={hist} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#243049" vertical={false} />
                <XAxis dataKey="bin" stroke="#8a99b3" fontSize={10} interval={3} />
                <YAxis stroke="#8a99b3" fontSize={12} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff08" }} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {hist.map((h, i) => (
                    <Cell key={i} fill={parseFloat(h.bin) >= 0.5 ? "#34d399" : "#f87171"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">no scored docs yet (quality model required)</div>
          )}
        </div>

        <div className="panel wide">
          <h2>Throughput (docs/sec)</h2>
          {tput.length > 1 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={tput} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#243049" vertical={false} />
                <XAxis dataKey="t" stroke="#8a99b3" fontSize={12} unit="s" />
                <YAxis stroke="#8a99b3" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="rate" stroke="#4f8cff" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">accumulating throughput samples…</div>
          )}
        </div>
      </div>
    </div>
  );
}
