import { Film, Trash2, Clock } from "lucide-react";
import { formatBytes, type VideoMeta } from "@/lib/video-store";

interface Props {
  videos: VideoMeta[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function VideoLibrary({ videos, selectedId, onSelect, onDelete }: Props) {
  if (!videos.length) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card/30 p-8 text-center">
        <Film className="mx-auto size-8 text-muted-foreground/60" />
        <p className="mt-3 text-sm text-muted-foreground">
          Your library is empty. Upload a clip to start a detection run.
        </p>
      </div>
    );
  }
  return (
    <ul className="space-y-2">
      {videos.map((v) => {
        const active = v.id === selectedId;
        return (
          <li key={v.id}>
            <div
              className={`group flex items-center gap-3 rounded-xl border p-3 transition-all ${
                active
                  ? "border-primary bg-primary/10 glow-primary"
                  : "border-border bg-card/40 hover:border-primary/40 hover:bg-card/60"
              }`}
            >
              <button onClick={() => onSelect(v.id)} className="flex flex-1 items-center gap-3 text-left">
                <div
                  className={`flex size-10 shrink-0 items-center justify-center rounded-lg ${
                    active ? "bg-primary text-primary-foreground" : "bg-secondary text-foreground"
                  }`}
                >
                  <Film className="size-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-foreground">{v.name}</p>
                  <p className="mt-0.5 flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                    <span>{formatBytes(v.size)}</span>
                    <span className="opacity-50">·</span>
                    <Clock className="size-3" />
                    <span>{new Date(v.addedAt).toLocaleDateString()}</span>
                  </p>
                </div>
              </button>
              <button
                onClick={() => onDelete(v.id)}
                className="rounded-md p-1.5 text-muted-foreground opacity-0 transition-all hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
                aria-label="Delete video"
              >
                <Trash2 className="size-4" />
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
