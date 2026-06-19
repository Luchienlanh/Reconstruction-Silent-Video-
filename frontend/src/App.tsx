import { useEffect, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock3,
  FileVideo,
  Loader2,
  UploadCloud,
} from "lucide-react";

type Timing = {
  total_seconds?: number;
  vtp_seconds?: number;
  decode_seconds?: number;
};

type TranscribeResult = {
  filename?: string;
  transcript: string;
  vtp_text: string;
  model_type: string;
  checkpoint: string;
  device: string;
  visual_frames: number;
  visual_dim: number;
  timing: Timing;
  stats: Record<string, number>;
};

const formatSeconds = (value?: number) =>
  typeof value === "number" ? `${value.toFixed(2)}s` : "n/a";

const formatStat = (value: number) => value.toFixed(4);

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || isRunning) {
      return;
    }

    setIsRunning(true);
    setError("");
    setResult(null);

    const body = new FormData();
    body.append("video", file);

    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        body,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "Transcription failed.");
      }
      setResult(payload as TranscribeResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Transcription failed.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Silent video recognition</p>
          <h1>Lip-to-text demo</h1>
          <p className="subcopy">
            Upload a silent talking-head clip and run the trained l2t_arch checkpoint.
          </p>
        </div>
        <div className="status-strip">
          <span>
            <Activity size={16} />
            VTP visual features
          </span>
          <span>
            <CheckCircle2 size={16} />
            l2t_arch decoder
          </span>
        </div>
      </section>

      <section className="workspace-grid">
        <form className="upload-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <div>
              <p className="section-label">Input video</p>
              <h2>Video source</h2>
            </div>
            <FileVideo size={22} />
          </div>

          <label className={`dropzone ${file ? "has-file" : ""}`}>
            <input accept="video/*" type="file" onChange={handleFileChange} />
            <UploadCloud size={34} />
            <span>{file ? file.name : "Choose a video file"}</span>
            <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "MP4, MOV, AVI, MKV, WEBM"}</small>
          </label>

          <div className="video-frame">
            {previewUrl ? (
              <video controls muted playsInline src={previewUrl} />
            ) : (
              <div className="empty-preview">
                <FileVideo size={42} />
                <span>No video selected</span>
              </div>
            )}
          </div>

          <button className="primary-button" disabled={!file || isRunning} type="submit">
            {isRunning ? <Loader2 className="spin" size={18} /> : <UploadCloud size={18} />}
            {isRunning ? "Running model" : "Transcribe"}
          </button>

          {error ? (
            <div className="error-box">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          ) : null}
        </form>

        <section className="result-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Output text</p>
              <h2>Transcript</h2>
            </div>
            {isRunning ? <Loader2 className="spin" size={22} /> : <Clock3 size={22} />}
          </div>

          <div className={`transcript-box ${result ? "filled" : ""}`}>
            {result ? result.transcript || "No transcript returned." : "Waiting for model output."}
          </div>

          <div className="compare-grid">
            <div>
              <span>VTP hypothesis</span>
              <p>{result?.vtp_text || "n/a"}</p>
            </div>
            <div>
              <span>Model type</span>
              <p>{result?.model_type || "n/a"}</p>
            </div>
          </div>

          <div className="metric-grid">
            <Metric label="Device" value={result?.device || "n/a"} />
            <Metric label="Frames" value={result ? `${result.visual_frames}` : "n/a"} />
            <Metric label="Feature dim" value={result ? `${result.visual_dim}` : "n/a"} />
            <Metric label="Total time" value={formatSeconds(result?.timing.total_seconds)} />
            <Metric label="VTP time" value={formatSeconds(result?.timing.vtp_seconds)} />
            <Metric label="Decode time" value={formatSeconds(result?.timing.decode_seconds)} />
          </div>

          {result && Object.keys(result.stats).length > 0 ? (
            <div className="stats-row">
              {Object.entries(result.stats).map(([key, value]) => (
                <span key={key}>
                  {key}: {formatStat(value)}
                </span>
              ))}
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
