"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const PIN_VH_MULTIPLE = 3.2;
const IMMERSE_OVERFILL = 1.04;
const ENTRY_DELAY = 0.2;
const CARD_START_SCALE_DESKTOP = 0.6;
const CARD_START_SCALE_MOBILE = 0.82;

export type HeroScrubProps = {
  /** Number of frames in the scrubbed image sequence. Use 0 together with `media` to put any element in the card. */
  frameCount: number;
  frameUrl: (index: number) => string;
  titleTop: string;
  titleBottom: string;
  bgClassName?: string;
  accentHex?: string;
  defaultAspect?: number;
  /** Optional content for the card (e.g. a video embed) when no frame sequence is used. */
  media?: ReactNode;
  /** Optional captions shown one after another while the card fills the screen. */
  immersedLines?: string[];
  /** Small text under the titles in the resting state. */
  kicker?: ReactNode;
};

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener?.("change", update);
    return () => mq.removeEventListener?.("change", update);
  }, []);
  return reduced;
}

export function HeroScrub({
  frameCount,
  frameUrl,
  titleTop,
  titleBottom,
  bgClassName = "bg-black",
  accentHex = "#3a9b8a",
  defaultAspect = 16 / 9,
  media,
  immersedLines = [],
  kicker,
}: HeroScrubProps) {
  const usesFrames = frameCount > 0;
  const sectionRef = useRef<HTMLElement>(null);
  const stickyRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imagesRef = useRef<HTMLImageElement[]>([]);
  const lastDrawnRef = useRef<number>(-1);
  const bgRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const titleTopRef = useRef<HTMLHeadingElement>(null);
  const titleBottomRef = useRef<HTMLHeadingElement>(null);
  const kickerRef = useRef<HTMLDivElement>(null);
  const linesRef = useRef<HTMLDivElement>(null);

  const [ready, setReady] = useState(!usesFrames);
  const [framesOk, setFramesOk] = useState(true);
  const [aspect, setAspect] = useState<number>(defaultAspect);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced || !usesFrames) return;
    let cancelled = false;
    let errored = 0;
    const images: HTMLImageElement[] = new Array(frameCount);
    imagesRef.current = images;

    const onFirstReady = (img: HTMLImageElement) => {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (canvas && img.naturalWidth && img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(img, 0, 0);
        lastDrawnRef.current = 0;
        setAspect(img.naturalWidth / img.naturalHeight);
      }
      setReady(true);
    };

    const onErr = () => {
      errored++;
      if (!cancelled && errored >= 5) setFramesOk(false);
    };

    const loadOne = (i: number) => {
      const img = new window.Image();
      img.decoding = "async";
      if (i < 4) (img as HTMLImageElement & { fetchPriority?: string }).fetchPriority = "high";
      img.onerror = onErr;
      if (i === 0) img.onload = () => onFirstReady(img);
      img.src = frameUrl(i);
      images[i] = img;
    };

    const INITIAL = Math.min(20, frameCount);
    for (let i = 0; i < INITIAL; i++) loadOne(i);

    const BATCH = 20;
    let cursor = INITIAL;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const loadNext = () => {
      if (cancelled) return;
      const end = Math.min(frameCount, cursor + BATCH);
      for (let i = cursor; i < end; i++) loadOne(i);
      cursor = end;
      if (cursor < frameCount) timer = setTimeout(loadNext, 80);
    };
    timer = setTimeout(loadNext, 200);

    const fallbackTimer = window.setTimeout(() => {
      if (!cancelled && !images[0]?.complete) setFramesOk(false);
    }, 4500);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      window.clearTimeout(fallbackTimer);
    };
  }, [reduced, frameCount, frameUrl, usesFrames]);

  // Entry animation
  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ delay: ENTRY_DELAY });
      tl.from(bgRef.current, { opacity: 0, duration: 1.4, ease: "power2.out" });
      tl.from(cardRef.current, { opacity: 0, duration: 1.1, ease: "power3.out" }, 0.35);
      tl.from(titleTopRef.current, { opacity: 0, y: 30, duration: 1, ease: "expo.out" }, 0.5);
      tl.from(titleBottomRef.current, { opacity: 0, y: -30, duration: 1, ease: "expo.out" }, 0.62);
      if (kickerRef.current) tl.from(kickerRef.current, { opacity: 0, y: 12, duration: 0.8, ease: "expo.out" }, 0.9);
    }, sectionRef);
    return () => ctx.revert();
  }, [reduced]);

  // Scroll-driven choreography — uses sticky layout instead of ScrollTrigger pin
  useEffect(() => {
    if (reduced || !ready || (usesFrames && !framesOk)) return;
    const section = sectionRef.current;
    if (!section) return;

    const ctx = gsap.context(() => {
      const startScale = () => (window.innerWidth < 768 ? CARD_START_SCALE_MOBILE : CARD_START_SCALE_DESKTOP);

      const immerseScale = () => {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const baseW = Math.min(vw * 0.92, vh * 0.5 * aspect);
        const baseH = Math.min(vh * 0.5, (vw * 0.92) / aspect);
        if (baseW <= 0 || baseH <= 0) return 1.5;
        return Math.max(vw / baseW, vh / baseH) * IMMERSE_OVERFILL;
      };

      const isLoaded = (i: number) => {
        const img = imagesRef.current[i];
        return !!img && img.complete && img.naturalWidth > 0;
      };

      const drawFrame = (index: number) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        let useIdx = index;
        if (!isLoaded(useIdx)) {
          let found = -1;
          for (let d = 1; d < frameCount; d++) {
            if (useIdx - d >= 0 && isLoaded(useIdx - d)) { found = useIdx - d; break; }
            if (useIdx + d < frameCount && isLoaded(useIdx + d)) { found = useIdx + d; break; }
          }
          if (found === -1) return;
          useIdx = found;
        }
        if (lastDrawnRef.current === useIdx) return;
        const img = imagesRef.current[useIdx];
        const ctx2 = canvas.getContext("2d");
        if (!ctx2 || !img) return;
        ctx2.drawImage(img, 0, 0, canvas.width, canvas.height);
        lastDrawnRef.current = useIdx;
      };

      gsap.set(cardRef.current, { scale: startScale(), transformOrigin: "50% 50%" });

      const master = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: "top top",
          end: "bottom bottom",
          scrub: 0.4,
          invalidateOnRefresh: true,
          onUpdate: (self) => {
            if (!usesFrames) return;
            const p = self.progress;
            const mapped = gsap.utils.clamp(0, 1, (p - 0.15) / 0.63);
            const frameIdx = Math.min(frameCount - 1, Math.floor(mapped * frameCount));
            drawFrame(frameIdx);
          },
        },
      });

      master.to(cardRef.current, { scale: 1, ease: "power2.out", duration: 0.15 }, 0);
      master.to(titleTopRef.current, {
        x: () => (window.innerWidth < 768 ? "-70vw" : "-60vw"),
        letterSpacing: "0.02em",
        ease: "power2.inOut",
        duration: 0.15,
      }, 0);
      master.to(titleBottomRef.current, {
        x: () => (window.innerWidth < 768 ? "70vw" : "60vw"),
        letterSpacing: "0.02em",
        ease: "power2.inOut",
        duration: 0.15,
      }, 0);
      if (kickerRef.current) master.to(kickerRef.current, { opacity: 0, duration: 0.08 }, 0);

      master.to(cardRef.current, { scale: immerseScale(), ease: "power2.in", duration: 0.63 }, 0.15);
      master.to(titleTopRef.current, { opacity: 0, ease: "power1.in", duration: 0.22 }, 0.15);
      master.to(titleBottomRef.current, { opacity: 0, ease: "power1.in", duration: 0.22 }, 0.15);

      // Captions over the immersed card, one after another
      const lines = linesRef.current ? Array.from(linesRef.current.children) : [];
      const slot = lines.length ? 0.44 / lines.length : 0;
      lines.forEach((line, i) => {
        const at = 0.32 + i * slot;
        master.fromTo(line, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: slot * 0.35, ease: "power2.out" }, at);
        master.to(line, { opacity: 0, y: -24, duration: slot * 0.3, ease: "power2.in" }, at + slot * 0.7);
      });

      master.to(cardRef.current, { scale: startScale(), ease: "power3.inOut", duration: 0.22 }, 0.78);
      master.to(titleTopRef.current, {
        x: 0, opacity: 1, letterSpacing: "-0.04em", ease: "power2.inOut", duration: 0.22,
      }, 0.78);
      master.to(titleBottomRef.current, {
        x: 0, opacity: 1, letterSpacing: "-0.04em", ease: "power2.inOut", duration: 0.22,
      }, 0.78);

      ScrollTrigger.refresh();
    }, sectionRef);

    return () => ctx.revert();
  }, [ready, framesOk, reduced, aspect, frameCount, usesFrames]);

  // Tall section + inner sticky div = same visual as ScrollTrigger pin, without needing pin
  const tallHeight = reduced ? "100svh" : `${(PIN_VH_MULTIPLE + 1) * 100}vh`;

  return (
    <section
      ref={sectionRef}
      className={`relative w-full overflow-clip text-ink ${bgClassName}`}
      style={{ height: tallHeight }}
      aria-label="Cinematic scroll-scrubbed hero"
    >
      <div
        ref={stickyRef}
        className="sticky top-0 flex h-[100svh] w-full flex-col items-center justify-center overflow-hidden"
      >
        <div ref={bgRef} aria-hidden className="absolute inset-0 z-0" style={{ backgroundColor: accentHex }} />
        {/* engineering-paper grid and a soft edge so the page reads as printed, not as a screen */}
        <div aria-hidden className="grid-bg absolute inset-0 z-0 opacity-70" />
        <div aria-hidden className="absolute inset-0 z-0" style={{
          background: "radial-gradient(ellipse at center, transparent 55%, rgba(22,20,15,0.10) 100%)",
        }} />

        <div className="relative z-10 flex h-full w-full flex-col items-center justify-center gap-3 md:gap-4">
          <h2 ref={titleTopRef} className="display whitespace-nowrap" style={{
            fontSize: "clamp(2.1rem, 5.4vw, 5.4rem)", lineHeight: 0.85, letterSpacing: "-0.04em",
          }}>{titleTop}</h2>

          <div ref={cardRef} className="relative overflow-hidden rounded-[12px] bg-black shadow-[0_30px_80px_rgba(22,20,15,0.35)] ring-1 ring-black/20 will-change-transform md:rounded-[16px]" style={{
            width: `min(92vw, calc(50svh * ${aspect}))`,
            height: `min(50svh, calc(92vw / ${aspect}))`,
            aspectRatio: aspect,
          }}>
            <div aria-hidden className="pointer-events-none absolute inset-0 z-20 shadow-[inset_0_0_120px_rgba(0,0,0,0.45)]" />
            {usesFrames && framesOk && (
              <canvas ref={canvasRef} aria-hidden className="absolute inset-0 h-full w-full object-cover" />
            )}
            {!usesFrames && media}
          </div>

          <h2 ref={titleBottomRef} className="display whitespace-nowrap text-red" style={{
            fontSize: "clamp(2.1rem, 5.4vw, 5.4rem)", lineHeight: 0.85, letterSpacing: "-0.04em",
          }}>{titleBottom}</h2>

          {kicker && <div ref={kickerRef} className="mt-4">{kicker}</div>}
        </div>

        {immersedLines.length > 0 && (
          <div ref={linesRef} aria-hidden className="pointer-events-none absolute inset-0 z-30 grid place-items-center">
            {immersedLines.map((line) => (
              <p key={line} className="display col-start-1 row-start-1 px-6 text-center text-white opacity-0"
                style={{ fontSize: "clamp(2rem, 6vw, 5.5rem)", textShadow: "0 6px 40px rgba(0,0,0,0.7)" }}>
                {line}
              </p>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
