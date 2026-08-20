"use client";

// A scroll indicator shaped like a paperclip maximiser: the rail holds a fixed
// number of clip slots, and scrolling down "produces" clips into them. The head
// of the chain is where you are in the document; the ghosted slots below are
// what is left to convert.

import { useEffect, useRef, useState } from "react";

const PITCH = 17; // px between clip origins — clips overlap enough to interlock
const CLIP_H = 32; // viewBox height of one clip
const CLIP_W = 26;
const MIN_SLOTS = 6;

export function ScrollPaperclips() {
  const railRef = useRef<HTMLDivElement>(null);
  const [slots, setSlots] = useState(0);
  const [made, setMade] = useState(1);
  const [pct, setPct] = useState(0);
  const [active, setActive] = useState(false);

  // Rail capacity is a function of its rendered height, which CSS owns.
  useEffect(() => {
    const rail = railRef.current;
    if (!rail) return;
    const measure = () => {
      const usable = rail.clientHeight - CLIP_H;
      setSlots(Math.max(MIN_SLOTS, Math.floor(usable / PITCH) + 1));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(rail);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!slots) return;
    let frame = 0;
    let idle: ReturnType<typeof setTimeout>;

    const read = () => {
      frame = 0;
      const doc = document.documentElement;
      const scrollable = doc.scrollHeight - doc.clientHeight;
      const progress = scrollable > 4 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
      setPct(Math.round(progress * 100));
      setMade(Math.round(progress * (slots - 1)) + 1);
    };

    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(read);
      setActive(true);
      clearTimeout(idle);
      idle = setTimeout(() => setActive(false), 1100);
    };

    read();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      cancelAnimationFrame(frame);
      clearTimeout(idle);
    };
  }, [slots]);

  const railHeight = slots ? (slots - 1) * PITCH + CLIP_H : 0;
  const headY = (made - 1) * PITCH;
  const complete = pct >= 100;

  return (
    <div
      ref={railRef}
      className={`clip-rail${active ? " is-active" : ""}${complete ? " is-max" : ""}`}
      aria-hidden="true"
    >
      <div className="clip-track" />
      {slots > 0 && (
        <svg
          className="clip-chain"
          width={CLIP_W}
          height={railHeight}
          viewBox={`0 0 ${CLIP_W} ${railHeight}`}
          fill="none"
        >
          <defs>
            {/* One paperclip: outer arm down, bottom U-turn, inner arm up,
                small top U-turn, tail back down through the middle. */}
            <path
              id="clip-glyph"
              d="M6 12 L6 22 A7 7 0 0 0 20 22 L20 11 A4.5 4.5 0 0 0 11 11 L11 25"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </defs>
          {Array.from({ length: slots }, (_, i) => {
            const y = i * PITCH;
            const forged = i < made;
            // Alternating mirror so the stack reads as a chain, not a stamp.
            const flip = i % 2 === 1 ? ` translate(${CLIP_W} 0) scale(-1 1)` : "";
            return (
              <use
                key={i}
                href="#clip-glyph"
                transform={`translate(0 ${y})${flip}`}
                className={
                  forged ? (i === made - 1 ? "clip is-head" : "clip is-forged") : "clip is-slot"
                }
              />
            );
          })}
        </svg>
      )}
      <span className="clip-count" style={{ transform: `translateY(${headY}px)` }}>
        {complete ? "MAX" : String(made).padStart(3, "0")}
      </span>
    </div>
  );
}
