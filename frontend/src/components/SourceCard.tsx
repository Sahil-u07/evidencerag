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
  let cleaned = String(text || "");

  // Normalize escaped Markdown markers.
  cleaned = cleaned.replace(
    /\\+(?=[#*_`])/g,
    "",
  );

  // Remove duplicated Markdown heading titles:
  // "## BM25 BM25 is..." -> "BM25 is..."
  // "## Dense Retrieval Dense retrieval..." -> "Dense retrieval..."
  cleaned = cleaned.replace(
    /#{1,6}\s+([A-Za-z][A-Za-z0-9 _/-]{1,40})\s+\1\b/gi,
    "$1",
  );

  // Remove remaining Markdown heading markers.
  cleaned = cleaned.replace(
    /(^|\s)#{1,6}\s+/g,
    "$1",
  );

  // Remove fenced code blocks.
  cleaned = cleaned.replace(
    /```[\s\S]*?```/g,
    "",
  );

  // Remove common inline Markdown formatting.
  cleaned = cleaned
    .replace(
      /\*\*(.*?)\*\*/g,
      "$1",
    )
    .replace(
      /\*(.*?)\*/g,
      "$1",
    )
    .replace(
      /`([^`]+)`/g,
      "$1",
    );

  // Remove accidental immediately repeated words/phrases.
  cleaned = cleaned.replace(
    /\b([A-Za-z0-9][A-Za-z0-9_-]{1,30})\s+\1\b/gi,
    "$1",
  );

  // Flatten document whitespace for the source card.
  return cleaned
    .replace(/\r?\n+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}