import { useEffect, useRef, useState } from "react";
import { Play, Pause, Loader2 } from "lucide-react";
import { getVideoBlob, type VideoMeta } from "@/lib/video-store";
import * as detectionStore from "@/lib/detection-results";

type BackendFrame = {
  frame: number;
  timestamp_sec: number;
  num_proposals: number;
  num_detections: number;
  active_tracks?: number;
  warning_tracks?: number;
  max_score?: number;
  warning?: boolean;
  proposals?: Array<[number, number, number, number]>;
  detections?: Array<{
    box: [number, number, number, number];
    prob: number;
    quality: number;
    score: number;
    center?: [number, number];
    trackId?: number;
  }>;
  tracks?: Array<{
    track_id: number;
    last_box: [number, number, number, number];
    avg_score: number;
    hits: number;
    missed_frames: number;
    warning?: boolean;
  }>;
};

type BackendOutput = {
  frames?: BackendFrame[];
  frames_processed?: number;
  detections_total?: number;
  videoUrl?: string;
  csvUrl?: string;
  sourceVideoUrl?: string;
};

interface Props {
  video: VideoMeta;
}

export function DetectionPlayer({ video }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detectionRecord, setDetectionRecord] = useState<null | any>(null);
  const [currentStats, setCurrentStats] = useState<BackendFrame | null>(null);

  // Load blob -> object URL
  useEffect(() => {
    // poll detection store for status/result for this video
    setDetectionRecord(detectionStore.getRecord(video.id));
    const iv = setInterval(() => setDetectionRecord(detectionStore.getRecord(video.id)), 1000);
    return () => clearInterval(iv);
  }, [video.id]);

  const backendOutput = (detectionRecord?.result?.output ?? null) as BackendOutput | null;
  const backendFrames = backendOutput?.frames ?? [];
  const detectionReady = backendFrames.length > 0;

  const normalizeFrameStats = (frame: BackendFrame): BackendFrame => {
    const proposals = frame.proposals?.length ?? frame.num_proposals ?? 0;
    const detections = frame.detections?.length ?? frame.num_detections ?? 0;
    const activeTracks =
      frame.active_tracks ?? frame.tracks?.filter((track) => track.missed_frames === 0).length ?? 0;
    const warningTracks =
      frame.warning_tracks ?? frame.tracks?.filter((track) => track.warning).length ?? 0;
    const maxScore =
      frame.max_score ??
      (frame.detections?.length
        ? Math.max(...frame.detections.map((detection) => detection.score))
        : 0);

    return {
      ...frame,
      num_proposals: proposals,
      num_detections: detections,
      active_tracks: activeTracks,
      warning_tracks: warningTracks,
      max_score: maxScore,
    };
  };

  useEffect(() => {
    let revoke: string | null = null;
    setLoading(true);
    if (detectionRecord && detectionRecord.status !== "done") {
      setLoading(true);
      return () => {
        if (revoke) URL.revokeObjectURL(revoke);
      };
    }

    getVideoBlob(video.id).then((blob) => {
      if (!blob) {
        setLoading(false);
        return;
      }
      const u = URL.createObjectURL(blob);
      revoke = u;
      setUrl(u);
      setLoading(false);
    });
    return () => {
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [video.id, detectionRecord?.status]);

  // Render loop: keep time in sync with the playing clip.
  useEffect(() => {
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        const t = v.currentTime;
        setTime(t);
        const currentBackendFrame =
          detectionReady && duration > 0
            ? backendFrames[
                Math.min(
                  backendFrames.length - 1,
                  Math.max(0, Math.round((t / Math.max(duration, 0.001)) * (backendFrames.length - 1))),
                )
              ]
            : null;
        setCurrentStats(currentBackendFrame ? normalizeFrameStats(currentBackendFrame) : null);
        const c = canvasRef.current;
        if (c && currentBackendFrame) {
          const w = c.clientWidth;
          const h = c.clientHeight;
          if (c.width !== w || c.height !== h) {
            c.width = w;
            c.height = h;
          }
          const ctx = c.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, w, h);
            const srcW = v.videoWidth || w;
            const srcH = v.videoHeight || h;
            const sx = w / Math.max(srcW, 1);
            const sy = h / Math.max(srcH, 1);

            for (const det of currentBackendFrame.detections ?? []) {
              const [x1, y1, x2, y2] = det.box;
              const rx = x1 * sx;
              const ry = y1 * sy;
              const rw = (x2 - x1) * sx;
              const rh = (y2 - y1) * sy;
              const cx = rx + rw / 2;
              const cy = ry + rh / 2;
              const radius = Math.max(rw, rh);
              const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.6);
              grad.addColorStop(0, `rgba(255,159,28,${0.35 + det.score * 0.4})`);
              grad.addColorStop(1, "transparent");
              ctx.fillStyle = grad;
              ctx.beginPath();
              ctx.arc(cx, cy, radius * 1.6, 0, Math.PI * 2);
              ctx.fill();
              ctx.lineWidth = 2;
              ctx.strokeStyle = det.score > 0.75 ? "#ff7a18" : "#ff9f1c";
              ctx.strokeRect(rx, ry, rw, rh);
              const label = `p=${det.prob.toFixed(2)} q=${det.quality.toFixed(2)}`;
              ctx.font = "500 11px 'JetBrains Mono', monospace";
              const tw = ctx.measureText(label).width + 10;
              ctx.fillStyle = "oklch(0.18 0.03 260 / 0.85)";
              ctx.fillRect(rx, ry - 18, tw, 16);
              ctx.fillStyle = "oklch(0.96 0.01 240)";
              ctx.fillText(label, rx + 5, ry - 6);
            }

            for (const track of currentBackendFrame.tracks ?? []) {
              const [x1, y1, x2, y2] = track.last_box;
              const rx = x1 * sx;
              const ry = y1 * sy;
              const rw = (x2 - x1) * sx;
              const rh = (y2 - y1) * sy;
              const isWarning = track.warning ?? false;
              ctx.lineWidth = isWarning ? 3 : 2;
              ctx.strokeStyle = isWarning ? "#ff3b30" : "#ff9f1c";
              ctx.strokeRect(rx, ry, rw, rh);
              const label = `T${track.track_id} S:${track.avg_score.toFixed(2)} H:${track.hits}`;
              ctx.font = "500 11px 'JetBrains Mono', monospace";
              const tw = ctx.measureText(label).width + 10;
              ctx.fillStyle = "oklch(0.18 0.03 260 / 0.85)";
              ctx.fillRect(rx, ry + rh + 2, tw, 16);
              ctx.fillStyle = "oklch(0.96 0.01 240)";
              ctx.fillText(label, rx + 5, ry + rh + 14);
            }

          }
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [backendOutput, detectionReady, duration]);

  const toggle = async () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      try {
        await v.play();
      } catch {
        // Keep the player silent if the browser rejects playback.
      }
    } else {
      v.pause();
    }
  };

  const seek = (pct: number) => {
    const v = videoRef.current;
    if (!v || !duration) return;
    v.currentTime = pct * duration;
  };

  const fmtT = (s: number) => {
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${r.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-4">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-black shadow-2xl">
        <div className="relative aspect-video w-full">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-card/60 backdrop-blur">
              <Loader2 className="size-6 animate-spin text-primary" />
            </div>
          )}
          {!loading && detectionRecord && detectionRecord.status !== "done" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 text-center">
              <Loader2 className="mb-3 size-6 animate-spin text-primary" />
              <p className="text-sm font-medium text-foreground">Waiting for detection to finish</p>
              <p className="mt-1 max-w-xs text-xs text-muted-foreground">
                The clip will appear here only after the backend has finished the notebook-style
                detection run.
              </p>
            </div>
          )}
          {currentStats && (
            <div className="absolute left-4 top-4 z-10 rounded-xl border border-border/60 bg-black/70 px-3 py-2 font-mono text-[11px] leading-5 text-foreground shadow-lg backdrop-blur-sm">
              <div>Proposals: {currentStats.num_proposals}</div>
              <div>Detections: {currentStats.num_detections}</div>
              <div>Active Tracks: {currentStats.active_tracks ?? 0}</div>
              <div>Warning Tracks: {currentStats.warning_tracks ?? 0}</div>
              <div>Max Score: {(currentStats.max_score ?? 0).toFixed(2)}</div>
              {currentStats.warning && <div className="mt-1 text-red-600 font-bold">APPROACHING VEHICLE WARNING</div>}
            </div>
          )}
          {url && (
            <video
              ref={videoRef}
              src={url}
              className="size-full object-contain"
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              playsInline
            />
          )}
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" />
          {!playing && !loading && detectionReady && (
            <button
              onClick={toggle}
              className="absolute inset-0 flex items-center justify-center bg-black/30 transition-opacity hover:bg-black/40"
            >
              <span className="rounded-full bg-primary/95 p-5 text-primary-foreground shadow-2xl glow-primary">
                <Play className="size-7 fill-current" />
              </span>
            </button>
          )}
        </div>

        {/* Controls */}
        <div className="border-t border-border bg-card/80 p-3 backdrop-blur">
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="flex size-9 items-center justify-center rounded-full bg-primary text-primary-foreground hover:opacity-90"
            >
              {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
            </button>
            <div className="flex-1">
              <div
                className="relative h-2 cursor-pointer rounded-full bg-secondary"
                onClick={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  seek((e.clientX - r.left) / r.width);
                }}
              >
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-primary"
                  style={{ width: duration ? `${(time / duration) * 100}%` : "0%" }}
                />
              </div>
            </div>
            <span className="font-mono text-xs text-muted-foreground">
              {fmtT(time)} / {fmtT(duration)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
