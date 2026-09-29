import { useEffect, useRef } from "react";
import type { CSSProperties } from "react";

/** stage: 0 idle · 1 searching · 2 finding · 3 checking · 4 error · 5 done (evidence marked) */
const WIDTHS = [92, 100, 84, 96, 60, 100, 88, 70, 44];
const HITS = [7, 4, 1, 6, 3];
const CAPTION = ["", "Searching your documents", "Finding supporting passages", "Checking the answer", "Backend unreachable", "Evidence marked"];

type Props = { stage: number; hits?: number; size?: "hero" | "mini" };

export default function EvidenceStack({ stage, hits = 3, size = "hero" }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.matchMedia("(pointer: fine)").matches || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const move = (e: PointerEvent) => {
      const el = ref.current;
      if (!el) return;
      el.style.setProperty("--ry", `${(e.clientX / window.innerWidth - 0.5) * 18}deg`);
      el.style.setProperty("--rx", `${(e.clientY / window.innerHeight - 0.5) * -12}deg`);
    };
    window.addEventListener("pointermove", move, { passive: true });
    return () => window.removeEventListener("pointermove", move);
  }, []);

  const hit = HITS.slice(0, Math.max(1, Math.min(hits, HITS.length)));

  return (
    <div ref={ref} className={`stack stack-${size} stage-${stage}`} aria-hidden="true">
      <div className="stack-scene">
        <i className="ground" />
        {Array.from({ length: 9 }, (_, i) => (
          <div key={i} className={`slot${hit.includes(i) ? " hit" : ""}`} style={{ "--i": i } as CSSProperties}>
            <div className="sheet">
              <div className="face">
                <b className="t" />
                {WIDTHS.map((w, k) => (
                  <i key={k} className={`ln${k === 2 || k === 5 ? " mk" : ""}`} style={{ width: `${w}%` }} />
                ))}
                <span className="ok">✓</span>
              </div>
            </div>
          </div>
        ))}
        <i className="scan" />
      </div>
      {size === "mini" && CAPTION[stage] && <span className="stack-cap">{CAPTION[stage]}</span>}
    </div>
  );
}
