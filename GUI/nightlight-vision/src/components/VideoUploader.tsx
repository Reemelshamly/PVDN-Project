import { useCallback, useRef, useState } from "react";
import { UploadCloud, Loader2 } from "lucide-react";
import { saveVideo, type VideoMeta } from "@/lib/video-store";
import * as detectionStore from "@/lib/detection-results";

export function VideoUploader({ onAdded }: { onAdded: (m: VideoMeta) => void }) {
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || !files.length) return;
      setErr(null);
      setBusy(true);
      try {
        for (const f of Array.from(files)) {
          if (!f.type.startsWith("video/")) {
            setErr(`Skipped ${f.name}: not a video`);
            continue;
          }
          const fd = new FormData();
          fd.append("file", f, f.name);

          const res = await fetch("/api/detect", {
            method: "POST",
            body: fd,
          });

          if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || "Detection backend error");
          }

          const data = await res.json();
          const meta = await saveVideo(f);
          detectionStore.setResult(meta.id, data);
          onAdded(meta);
        }
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setBusy(false);
      }
    },
    [onAdded],
  );

  return (
    <div className="space-y-3">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`relative flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed bg-card/40 px-6 py-12 text-center transition-all ${
          drag ? "border-primary bg-primary/5 glow-primary" : "border-border hover:border-primary/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          multiple
          className="sr-only"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="rounded-full bg-primary/15 p-4 text-primary">
          {busy ? <Loader2 className="size-7 animate-spin" /> : <UploadCloud className="size-7" />}
        </div>
        <div>
          <p className="text-base font-semibold text-foreground">
            {busy ? "Running notebook detection…" : "Drop a night-driving clip"}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            or <span className="text-primary underline-offset-2 hover:underline">browse files</span> · MP4, WebM, MOV
          </p>
        </div>
        <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground/70">
          Detection completes before the clip is added to your library
        </p>
      </label>
      {err && <p className="text-sm text-destructive">{err}</p>}
    </div>
  );
}
