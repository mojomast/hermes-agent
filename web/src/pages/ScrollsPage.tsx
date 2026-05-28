import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Clock,
  Database,
  FileText,
  FolderOpen,
  GitCompareArrows,
  Lightbulb,
  Play,
  RefreshCw,
  ScrollText,
  Settings2,
  TrendingUp,
  Trophy,
} from "lucide-react";
import { api } from "@/lib/api";
import type { ScrollsArtifactFile, ScrollsConfigDiffEntry, ScrollsConfigInfo, ScrollsExperimentRun, ScrollsHypothesisProjection, ScrollsMetricTrendPoint, ScrollsResearchResponse, ScrollsValidationMatrixCell } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";

function fmtNumber(value: unknown, digits = 4): string {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function fmtTime(value?: string | number | null): string {
  if (!value) return "—";
  const d = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

function fmtBytes(value?: number | null): string {
  if (!Number.isFinite(value ?? NaN)) return "—";
  const n = value ?? 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function getNested(obj: unknown, path: string[]): unknown {
  let cur = obj as Record<string, unknown> | undefined;
  for (const part of path) {
    if (!cur || typeof cur !== "object") return undefined;
    cur = cur[part] as Record<string, unknown> | undefined;
  }
  return cur;
}

function compactJson(value: unknown): string {
  if (value === undefined) return "—";
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function SummaryTile({ icon: Icon, label, value, sub }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{label}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {sub && <p className="mt-1 text-xs text-muted-foreground normal-case">{sub}</p>}
      </CardContent>
    </Card>
  );
}

function RunTable({ runs }: { runs: ScrollsExperimentRun[] }) {
  if (!runs.length) {
    return <p className="py-6 text-sm text-muted-foreground normal-case">No experiments have been logged yet.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-muted-foreground">
            <th className="py-2 pr-4 text-left font-medium">Run</th>
            <th className="px-4 py-2 text-right font-medium">Val loss</th>
            <th className="px-4 py-2 text-right font-medium">F1</th>
            <th className="px-4 py-2 text-right font-medium">LR</th>
            <th className="px-4 py-2 text-right font-medium">Depth</th>
            <th className="py-2 pl-4 text-right font-medium">Time</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="border-b border-border/50 hover:bg-secondary/20">
              <td className="py-2 pr-4 font-mono-ui text-xs normal-case">{run.run_id}</td>
              <td className="px-4 py-2 text-right">{fmtNumber(run.metrics.val_loss ?? run.main_metric)}</td>
              <td className="px-4 py-2 text-right">{fmtNumber(run.metrics.val_f1)}</td>
              <td className="px-4 py-2 text-right">{fmtNumber(getNested(run.config, ["training", "learning_rate"]), 5)}</td>
              <td className="px-4 py-2 text-right">{String(getNested(run.config, ["model", "depth"]) ?? "—")}</td>
              <td className="py-2 pl-4 text-right text-muted-foreground normal-case">{fmtTime(run.timestamp)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ConfigList({ configs, running, onRun }: { configs: ScrollsConfigInfo[]; running: string | null; onRun: (name: string) => void }) {
  if (!configs.length) return <p className="text-sm text-muted-foreground normal-case">No configs found.</p>;
  return (
    <div className="grid gap-2">
      {configs.slice(0, 12).map((cfg) => (
        <div key={cfg.name} className="flex items-center justify-between gap-3 border border-border/60 bg-secondary/10 p-3">
          <div className="min-w-0">
            <div className="truncate font-mono-ui text-xs normal-case">{cfg.name}</div>
            <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-muted-foreground normal-case">
              <span>lr {fmtNumber(getNested(cfg.summary, ["training", "learning_rate"]), 5)}</span>
              <span>depth {String(getNested(cfg.summary, ["model", "depth"]) ?? "—")}</span>
              <span>train {String(getNested(cfg.summary, ["dataset", "train_scroll_id"]) ?? "—")}</span>
              <span>val {String(getNested(cfg.summary, ["dataset", "val_scroll_id"]) ?? "—")}</span>
            </div>
          </div>
          <Button size="sm" variant="outline" disabled={!!running} onClick={() => onRun(cfg.name)}>
            <Play className="h-3 w-3" />
            Run
          </Button>
        </div>
      ))}
    </div>
  );
}

function ArtifactList({ artifacts }: { artifacts: ScrollsArtifactFile[] }) {
  if (!artifacts.length) return <p className="text-sm text-muted-foreground normal-case">No artifacts listed for the latest run.</p>;
  return (
    <div className="grid gap-2">
      {artifacts.map((artifact) => (
        <div key={artifact.path} className="flex items-center justify-between gap-3 border border-border/60 bg-secondary/10 p-2 text-xs normal-case">
          <div className="min-w-0">
            <div className="truncate font-mono-ui">{artifact.name}</div>
            <div className="truncate text-muted-foreground">{artifact.path}</div>
          </div>
          <Badge variant="outline" className="shrink-0 normal-case">{artifact.kind} · {fmtBytes(artifact.size_bytes)}</Badge>
        </div>
      ))}
    </div>
  );
}

function MetricTrend({ points }: { points: ScrollsMetricTrendPoint[] }) {
  if (!points.length) return <p className="py-6 text-sm text-muted-foreground normal-case">No metric trend data yet.</p>;
  const losses = points.map((p) => Number(p.val_loss ?? p.main_metric)).filter(Number.isFinite);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const latest = points[points.length - 1];
  const previous = points.length > 1 ? points[points.length - 2] : null;
  const delta = previous ? Number(latest.main_metric) - Number(previous.main_metric) : null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 text-xs normal-case">
        <Badge variant={delta !== null && delta <= 0 ? "default" : "secondary"}>latest loss {fmtNumber(latest.val_loss ?? latest.main_metric)}</Badge>
        <Badge variant="outline">Δ previous {delta === null ? "—" : fmtNumber(delta)}</Badge>
        <Badge variant="outline">latest F1 {fmtNumber(latest.val_f1)}</Badge>
      </div>
      <div className="flex h-28 items-end gap-1 border border-border/60 bg-black/20 p-3" title="Recent val_loss/main_metric trend; lower is better for current configs">
        {points.map((p) => {
          const value = Number(p.val_loss ?? p.main_metric);
          const normalized = max === min ? 0.5 : (value - min) / (max - min);
          const height = 12 + (1 - normalized) * 84;
          return (
            <div key={p.run_id} className="min-w-2 flex-1 rounded-t bg-primary/70" style={{ height: `${height}%` }} title={`${p.run_id}: loss ${fmtNumber(value)} F1 ${fmtNumber(p.val_f1)}`} />
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground normal-case">Bars are recent runs ordered left→right; taller means lower validation loss.</p>
    </div>
  );
}

function ValidationMatrix({ cells }: { cells: ScrollsValidationMatrixCell[] }) {
  if (!cells.length) return <p className="py-6 text-sm text-muted-foreground normal-case">No train/validation scroll pairs recorded yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-muted-foreground">
            <th className="py-2 pr-4 text-left font-medium">Train → Val</th>
            <th className="px-4 py-2 text-right font-medium">Best loss</th>
            <th className="px-4 py-2 text-right font-medium">Best F1</th>
            <th className="px-4 py-2 text-right font-medium">Runs</th>
            <th className="py-2 pl-4 text-left font-medium">Best run</th>
          </tr>
        </thead>
        <tbody>
          {cells.map((cell) => (
            <tr key={`${cell.train_scroll_id}-${cell.val_scroll_id}`} className="border-b border-border/50 hover:bg-secondary/20">
              <td className="py-2 pr-4 font-mono-ui text-xs normal-case">{cell.train_scroll_id} → {cell.val_scroll_id}</td>
              <td className="px-4 py-2 text-right">{fmtNumber(cell.best_main_metric)}</td>
              <td className="px-4 py-2 text-right">{fmtNumber(cell.best_val_f1)}</td>
              <td className="px-4 py-2 text-right">{cell.run_count}</td>
              <td className="py-2 pl-4 font-mono-ui text-xs normal-case text-muted-foreground">{cell.best_run_id}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LatestRunCard({ run }: { run: ScrollsExperimentRun | null | undefined }) {
  if (!run) return <p className="text-sm text-muted-foreground normal-case">No latest run available.</p>;
  const reason = getNested(run.config, ["autoresearch", "parent_reason"]);
  const reasonText = reason == null ? "" : String(reason);
  return (
    <div className="space-y-3 text-sm normal-case">
      <div className="flex flex-wrap gap-2">
        <Badge className="font-mono-ui normal-case">{run.run_id}</Badge>
        <Badge variant="outline">loss {fmtNumber(run.metrics.val_loss ?? run.main_metric)}</Badge>
        <Badge variant="outline">F1 {fmtNumber(run.metrics.val_f1)}</Badge>
        <Badge variant="outline">{fmtTime(run.timestamp)}</Badge>
      </div>
      {reasonText && <p className="text-muted-foreground">Hypothesis: {reasonText}</p>}
      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
        <span>train {String(getNested(run.config, ["dataset", "train_scroll_id"]) ?? "—")}</span>
        <span>val {String(getNested(run.config, ["dataset", "val_scroll_id"]) ?? "—")}</span>
        <span>lr {fmtNumber(getNested(run.config, ["training", "learning_rate"]), 5)}</span>
        <span>depth {String(getNested(run.config, ["model", "depth"]) ?? "—")}</span>
      </div>
      <ArtifactList artifacts={run.artifacts ?? []} />
    </div>
  );
}

function ConfigDiffCard({ diffs, latest, previous }: { diffs: ScrollsConfigDiffEntry[]; latest?: ScrollsExperimentRun | null; previous?: ScrollsExperimentRun | null }) {
  if (!latest) return <p className="text-sm text-muted-foreground normal-case">No latest run available.</p>;
  if (!previous) return <p className="text-sm text-muted-foreground normal-case">Need at least two runs to show a config diff.</p>;
  if (!diffs.length) return <p className="text-sm text-muted-foreground normal-case">Latest run config matches the previous run.</p>;
  const delta = Number(latest.main_metric) - Number(previous.main_metric);
  return (
    <div className="space-y-3 normal-case">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>Comparing latest <span className="font-mono-ui">{latest.run_id}</span> against previous <span className="font-mono-ui">{previous.run_id}</span>.</span>
        <Badge variant={Number.isFinite(delta) && delta <= 0 ? "default" : "secondary"}>Δ loss {Number.isFinite(delta) ? fmtNumber(delta) : "—"}</Badge>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-muted-foreground">
              <th className="py-2 pr-3 text-left font-medium">Config path</th>
              <th className="px-3 py-2 text-left font-medium">Before</th>
              <th className="py-2 pl-3 text-left font-medium">After</th>
            </tr>
          </thead>
          <tbody>
            {diffs.map((diff) => (
              <tr key={diff.path} className="border-b border-border/50">
                <td className="py-2 pr-3 font-mono-ui text-primary">{diff.path}</td>
                <td className="max-w-48 truncate px-3 py-2 font-mono-ui text-muted-foreground" title={compactJson(diff.before)}>{compactJson(diff.before)}</td>
                <td className="max-w-48 truncate py-2 pl-3 font-mono-ui" title={compactJson(diff.after)}>{compactJson(diff.after)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HypothesisTracker({ hypotheses }: { hypotheses: ScrollsHypothesisProjection[] }) {
  if (!hypotheses.length) return <p className="text-sm text-muted-foreground normal-case">No AutoResearch hypotheses found in recent run configs.</p>;
  return (
    <div className="grid gap-2">
      {hypotheses.slice(0, 6).map((item) => (
        <div key={item.run_id} className="border border-border/60 bg-secondary/10 p-3 text-xs normal-case">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="font-mono-ui normal-case">{item.run_id}</Badge>
            <Badge variant={item.status === "improved" ? "default" : item.status === "regressed" ? "secondary" : "outline"}>{item.status}</Badge>
            <Badge variant="outline">loss {fmtNumber(item.metric)}</Badge>
            <Badge variant="outline">Δ previous {item.metric_delta_vs_previous == null ? "—" : fmtNumber(item.metric_delta_vs_previous)}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">{item.reason || "No hypothesis text recorded."}</p>
          <div className="mt-2 flex flex-wrap gap-1">
            {item.changed_paths.slice(0, 8).map((path) => <Badge key={path} variant="outline" className="font-mono-ui text-[10px] normal-case">{path}</Badge>)}
            {!item.changed_paths.length && <span className="text-muted-foreground">No config changes recorded.</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ScrollsPage() {
  const [data, setData] = useState<ScrollsResearchResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);
  const { toast, showToast } = useToast();

  const load = useCallback(() => {
    setLoading(true);
    api
      .getScrollsResearch()
      .then(setData)
      .catch((e) => showToast(`Failed to load ScrollPrize research: ${e}`, "error"))
      .finally(() => setLoading(false));
  }, [showToast]);

  useEffect(() => {
    load();
  }, [load]);

  const best = data?.experiments.best ?? null;
  const latestRun = data?.experiments.latest ?? data?.experiments.recent?.[0] ?? null;
  const previousRun = data?.experiments.recent?.[1] ?? null;
  const source = String(data?.data_summary.source ?? "unknown");
  const scrollCount = Array.isArray(data?.data_summary.scrolls) ? data?.data_summary.scrolls.length ?? 0 : 0;
  const bestMetric = best ? fmtNumber(best.main_metric) : "—";
  const bestSub = best ? `${best.run_id} · F1 ${fmtNumber(best.metrics.val_f1)}` : "No baseline yet";

  const handleAutoresearch = async () => {
    setRunning("autoresearch");
    try {
      const res = await api.triggerScrollsAutoresearch();
      showToast(`AutoResearch started (pid ${res.pid})`, "success");
      setTimeout(load, 1200);
    } catch (e) {
      showToast(`Could not start AutoResearch: ${e}`, "error");
    } finally {
      setRunning(null);
    }
  };

  const handleRunConfig = async (name: string) => {
    setRunning(name);
    try {
      const res = await api.runScrollsConfig(name);
      showToast(`Experiment started (pid ${res.pid})`, "success");
      setTimeout(load, 1200);
    } catch (e) {
      showToast(`Could not start experiment: ${e}`, "error");
    } finally {
      setRunning(null);
    }
  };

  const scrolls = useMemo(() => {
    const raw = data?.data_summary.scrolls;
    return Array.isArray(raw) ? raw as Array<Record<string, unknown>> : [];
  }, [data]);

  if (loading && !data) {
    return <div className="flex items-center justify-center py-24"><div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" /></div>;
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-bold tracking-wide">
            <ScrollText className="h-5 w-5" />
            ScrollPrize Research
          </h1>
          <p className="mt-1 text-sm text-muted-foreground normal-case">
            Manage the Vesuvius ink-detection AutoResearch loop, configs, data slices, and experiment leaderboard.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={load} disabled={loading}>
            <RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
            Refresh
          </Button>
          <Button onClick={handleAutoresearch} disabled={!!running || data?.lock_active}>
            <Play className="h-4 w-4" />
            {data?.lock_active ? "Running" : "Run AutoResearch"}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SummaryTile icon={Trophy} label="Best val loss" value={bestMetric} sub={bestSub} />
        <SummaryTile icon={Activity} label="Runs" value={String(data?.experiments.count ?? 0)} sub={`${data?.configs.length ?? 0} configs tracked`} />
        <SummaryTile icon={Database} label="Data source" value={source} sub={`${scrollCount} scroll inventories · ${data?.prepared_datasets.length ?? 0} prepared subsets`} />
        <SummaryTile icon={Clock} label="Cron" value={data?.cron.installed ? "Installed" : "Missing"} sub={data?.cron.line ?? "No Vesuvius cron line found"} />
      </div>

      {!data?.exists && (
        <Card className="border-destructive/60">
          <CardContent className="py-4 text-sm text-destructive normal-case">
            AutoResearch project root not found at {data?.project_root}.
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><BarChart3 className="h-4 w-4" /> Experiment leaderboard</CardTitle>
          </CardHeader>
          <CardContent>
            <RunTable runs={data?.experiments.recent ?? []} />
          </CardContent>
        </Card>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base"><FolderOpen className="h-4 w-4" /> Latest run + artifacts</CardTitle>
            </CardHeader>
            <CardContent>
              <LatestRunCard run={latestRun} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base"><Settings2 className="h-4 w-4" /> Configs</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfigList configs={data?.configs ?? []} running={running} onRun={handleRunConfig} />
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><TrendingUp className="h-4 w-4" /> Metric trend</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricTrend points={data?.experiments.metric_trends ?? []} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Database className="h-4 w-4" /> Cross-scroll validation matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <ValidationMatrix cells={data?.experiments.validation_matrix ?? []} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><GitCompareArrows className="h-4 w-4" /> Latest config diff</CardTitle>
          </CardHeader>
          <CardContent>
            <ConfigDiffCard diffs={data?.experiments.config_diffs?.latest_vs_previous ?? []} latest={latestRun} previous={previousRun} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Lightbulb className="h-4 w-4" /> Hypothesis tracker</CardTitle>
          </CardHeader>
          <CardContent>
            <HypothesisTracker hypotheses={data?.experiments.hypotheses ?? []} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Database className="h-4 w-4" /> Scroll inventory</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {scrolls.map((scroll) => (
                <Badge key={String(scroll.scroll_id)} variant="outline" className="normal-case">
                  Scroll {String(scroll.scroll_id)} · {Array.isArray(scroll.segments) ? scroll.segments.length : 0} regions
                </Badge>
              ))}
              {!scrolls.length && <span className="text-sm text-muted-foreground normal-case">No scroll inventory available.</span>}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><FileText className="h-4 w-4" /> AutoResearch log tail</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-72 overflow-auto whitespace-pre-wrap border border-border/60 bg-black/30 p-3 font-mono-ui text-xs normal-case text-muted-foreground">
              {(data?.logs.lines ?? []).join("\n") || "No log output yet."}
            </pre>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
