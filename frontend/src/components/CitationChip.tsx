import type { EvidenceSource } from "../types";

export default function CitationChip({ source }: { source: EvidenceSource }) {
  const open = () => {
    const el = document.getElementById(`source-${source.id}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("source-highlight");
    window.setTimeout(() => el.classList.add("source-highlight"), 20);
    window.setTimeout(() => el.classList.remove("source-highlight"), 1800);
  };
  const snippet = source.text.replace(/[#*`]/g, "").replace(/\s+/g, " ").trim().slice(0, 170);

  return (
    <span className="cite">
      <button type="button" className="citation-chip" onClick={open} aria-label={`Open Evidence ${source.id} from ${source.file}`}>
        {source.id}
      </button>
      <span className="cite-pop" role="tooltip">
        <strong>{source.file}</strong>
        <small>
          {source.page !== null ? `Page ${source.page} · ` : ""}
          {source.score > 0 ? `score ${source.score.toFixed(2)}` : source.label}
        </small>
        {snippet && <em>{snippet}{source.text.length > 170 ? "…" : ""}</em>}
      </span>
    </span>
  );
}
