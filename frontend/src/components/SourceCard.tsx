import type { EvidenceSource } from "../types";

type SourceCardProps = {
  source: EvidenceSource;
};

export default function SourceCard({
  source,
}: SourceCardProps) {
  return (
    <article
      id={`source-${source.id}`}
      className={`source-card ${
        source.used
          ? "source-used"
          : "source-not-used"
      }`}
    >
      <div className="source-card-top">
        <div className="source-file">
          <span
            className="source-file-icon"
            aria-hidden="true"
          >
            ◫
          </span>

          <div>
            <strong>{source.file}</strong>

            {source.page !== null && (
              <span>
                Page {source.page}
              </span>
            )}
          </div>
        </div>

        <span
          className={`relevance-label relevance-${source.label
            .toLowerCase()
            .replace(" ", "-")}`}
        >
          {source.label}
        </span>
      </div>

      <div className="source-passage">
        <p>{cleanPassage(source.text)}</p>
      </div>

      <div className="source-card-footer">
        <span className="source-number">
          Evidence {source.id}
        </span>

        <details className="full-passage">
          <summary>Show full passage</summary>

          <div className="full-passage-content">
            {cleanPassage(source.text)}
          </div>
        </details>
      </div>
    </article>
  );
}

function cleanPassage(text: string): string {
  return String(text || "")
    // Remove escaped Markdown markers such as \# and \\#.
    .replace(/\\+(?=[#*_`])/g, "")
    // Remove Markdown heading markers while preserving the text.
    .replace(/(^|\s)#{1,6}\s+/g, "$1")
    // Remove fenced code blocks.
    .replace(/```[\s\S]*?```/g, "")
    // Remove common inline Markdown emphasis/code.
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    // Flatten extracted document whitespace for readable cards.
    .replace(/\r?\n+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}