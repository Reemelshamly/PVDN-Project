import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Car, Sparkles, Radar, Brain, Layers, Activity, MoveRight } from "lucide-react";
import { VideoUploader } from "@/components/VideoUploader";
import { VideoLibrary } from "@/components/VideoLibrary";
import { DetectionPlayer } from "@/components/DetectionPlayer";
import { deleteVideo, listVideos, type VideoMeta } from "@/lib/video-store";
import * as detectionStore from "@/lib/detection-results";
import nightRoadGif from "../../tumblr_afcce6a09b59f901b386e7982c28d7ba_aa71cd45_500.gif";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    listVideos().then((v) => {
      setVideos(v);
      if (!selectedId) {
        const readyVideo = v.find((video) => detectionStore.getRecord(video.id)?.status === "done");
        if (readyVideo) {
          setSelectedId(readyVideo.id);
        }
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = videos.find((v) => v.id === selectedId) ?? null;

  const handleAdded = (m: VideoMeta) => {
    setVideos((prev) => [m, ...prev]);
    setSelectedId(m.id);
  };

  const handleDelete = async (id: string) => {
    await deleteVideo(id);
    setVideos((prev) => prev.filter((v) => v.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  return (
    <div className="relative min-h-screen">
      {/* ambient backdrop */}
      <div className="pointer-events-none absolute inset-0 grid-roads opacity-30" />
      <div className="pointer-events-none absolute -left-32 top-0 size-[480px] rounded-full bg-primary/15 blur-3xl" />
      <div className="pointer-events-none absolute -right-32 top-40 size-[420px] rounded-full bg-accent/15 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Nav */}
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground glow-primary">
              <Radar className="size-5" />
            </div>
            <div>
              <p className="text-base font-bold tracking-tight text-foreground">PVDN</p>
              <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Provident Vehicle Detection · Night
              </p>
            </div>
          </div>
          <nav className="hidden items-center gap-6 text-sm text-muted-foreground sm:flex">
            <a href="#pipeline" className="hover:text-foreground">Pipeline</a>
            <a href="#library" className="hover:text-foreground">Library</a>
            <a
              href="#library"
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90"
            >
              Run detection
            </a>
          </nav>
        </header>

        {/* Hero */}
        <section className="mt-16 grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
              <Sparkles className="size-3 text-primary" />
              Reflection-based early warning
            </span>
            <h1 className="mt-5 text-5xl font-bold leading-[1.05] tracking-tight text-foreground sm:text-6xl">
              See the car <span className="text-primary text-glow">before</span> you see the car.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground">
              PVDN doesn't wait for headlights or vehicle shapes. It learns the subtle reflection
              patterns approaching vehicles leave on the road at night — combining classical computer
              vision, an EfficientNet patch classifier, and temporal tracking into one warning
              pipeline.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <a
                href="#library"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90 glow-primary"
              >
                Upload a clip <MoveRight className="size-4" />
              </a>
              <a
                href="#pipeline"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card/40 px-5 py-3 text-sm font-medium text-foreground transition hover:border-primary/60"
              >
                How it works
              </a>
            </div>

            <dl className="mt-10 grid grid-cols-3 gap-6 border-t border-border pt-6">
              <Metric k="Dataset" v="PVDN" sub="Kaggle, night driving" />
              <Metric k="Backbone" v="EfficientNet" sub="patch classifier" />
              <Metric k="Pipeline" v="9 stages" sub="blob → CNN → track" />
            </dl>
          </div>

          <div className="relative overflow-hidden rounded-3xl border border-border bg-card shadow-2xl">
            <img
              src={nightRoadGif}
              alt="Night road drive"
              className="block h-auto w-full object-cover"
            />
          </div>
        </section>

        <section id="pipeline" className="mt-24">
          <div className="mt-8 grid gap-3 md:grid-cols-3">
            {STAGES.map((s, i) => (
              <div
                key={s.title}
                className="group relative overflow-hidden rounded-xl border border-border bg-card/50 p-5 transition hover:border-primary/60 hover:bg-card/80"
              >
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Step {String(i + 1).padStart(2, "0")}
                </span>
                <div className="mt-2 flex items-start gap-3">
                  <s.icon className="mt-0.5 size-5 text-primary" />
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">{s.title}</h3>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{s.body}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Library + Player */}
        <section id="library" className="mt-24 scroll-mt-8">
          <div className="flex items-end justify-between">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-widest text-primary">Detection studio</p>
              <h2 className="mt-2 text-3xl font-bold text-foreground sm:text-4xl">Your night-driving library.</h2>
              <p className="mt-2 max-w-xl text-sm text-muted-foreground">
                Upload a clip and the system stores it locally in your browser. Pick any saved video
                to replay the detection overlay — reflection boxes, track IDs, CNN scores, and live
                warnings.
              </p>
            </div>
            <span className="hidden rounded-full border border-border bg-card/60 px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground md:inline-flex">
              {videos.length} clip{videos.length === 1 ? "" : "s"} saved
            </span>
          </div>

          <div className="mt-8 grid gap-6 lg:grid-cols-[340px_1fr]">
            <div className="space-y-5">
              <VideoUploader onAdded={handleAdded} />
              <div>
                <h3 className="mb-3 px-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                  Saved videos
                </h3>
                <VideoLibrary
                  videos={videos}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  onDelete={handleDelete}
                />
              </div>
            </div>
            <div>
              {selected ? (
                <DetectionPlayer key={selected.id} video={selected} />
              ) : (
                <div className="flex h-full min-h-[420px] flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-card/30 p-10 text-center">
                  <Radar className="size-10 text-muted-foreground/40" />
                  <p className="mt-4 text-base font-semibold text-foreground">No clip selected</p>
                  <p className="mt-1 max-w-xs text-sm text-muted-foreground">
                    Upload a night-driving video or pick one from your library to start the
                    detection overlay.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        <footer className="mt-24 border-t border-border pt-8 pb-4 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
          PVDN Project · Reflection-based early warning · Hybrid CV + Deep Learning + Temporal Tracking
        </footer>
      </div>
    </div>
  );
}

function Metric({ k, v, sub }: { k: string; v: string; sub: string }) {
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{k}</p>
      <p className="mt-1 text-lg font-semibold text-foreground">{v}</p>
      <p className="text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

const STAGES = [
  { title: "Video input", body: "OpenCV reads the night-driving clip frame-by-frame.", icon: Car },
  { title: "Preprocessing", body: "Grayscale, contrast enhancement, and noise reduction surface weak reflections.", icon: Sparkles },
  { title: "Blob proposals", body: "Bright connected regions become candidate reflection patches.", icon: Layers },
  { title: "Proposal filtering", body: "Geometry, brightness, and road position prune street lights and glare.", icon: Activity },
  { title: "CNN verification", body: "An EfficientNet patch classifier scores each surviving region.", icon: Brain },
  { title: "Quality scoring", body: "CNN probability fuses with blob quality into a combined score.", icon: Activity },
  { title: "Top selection", body: "Only the strongest, most consistent detections survive per frame.", icon: Sparkles },
  { title: "Temporal tracking", body: "Detections must persist and move like an approaching vehicle.", icon: Radar },
  { title: "Warning decision", body: "A stable, high-quality reflection track triggers the early warning.", icon: Car },
];
