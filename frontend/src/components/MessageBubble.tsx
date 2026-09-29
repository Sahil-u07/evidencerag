import { useState } from "react";
import type { ChatMessage } from "../types";
import ProgressSteps from "./ProgressSteps";
import CitationChip from "./CitationChip";
import SourceCard from "./SourceCard";
import VerificationBadge from "./VerificationBadge";

type MessageBubbleProps = {
  message: ChatMessage;
  onRetry: (query: string) => void;
  onDemo: () => void;
};

export default function MessageBubble({
  message,
  onRetry,
  onDemo,
}: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <article className="message-row user-row">
        <div className="message-user">
          {message.content}
        </div>
      </article>
    );
  }

  if (message.error) {
    return (
      <article className="message-row assistant-row">
        <div className="assistant-avatar is-error" aria-hidden="true"><span /></div>
        <div className="assistant-content">
          <div className="assistant-name">EvidenceRAG</div>
          {message.stages && (
            <ProgressSteps stages={message.stages} loading={false} failed />
          )}
          <div className="error-card" role="alert">
            <strong>Can't reach the backend</strong>
            <p>{message.content}</p>
            <div className="error-actions">
              <button type="button" className="btn btn-primary" onClick={() => onRetry(message.query || "")}>Retry</button>
              <button type="button" className="btn btn-ghost" onClick={onDemo}>Try demo mode</button>
            </div>
          </div>
        </div>
      </article>
    );
  }

  const sources = message.sources || [];

  const usedSources = sources.filter(
    (source) => source.used,
  );

  const otherSources = sources.filter(
    (source) => !source.used,
  );

  return (
    <article className="message-row assistant-row">
      <div className="assistant-avatar" aria-hidden="true">
        <span />
      </div>

      <div className="assistant-content">
        <div className="assistant-name">
          EvidenceRAG
        </div>

        {message.stages &&
          message.stages.length > 0 && (
            <ProgressSteps
              stages={message.stages}
              loading={Boolean(message.loading)}
            />
          )}

        {message.content && (
          <div className="answer-text" aria-live="polite">
            <AnswerWithCitations
              text={message.content}
              sources={sources}
            />
          </div>
        )}

        {message.loading && !message.content && (
          <div
            className="answer-placeholder"
            aria-live="polite"
          >
            Finding an answer from your documents…
          </div>
        )}

        {message.verification && (
          <VerificationBadge
            verification={message.verification}
            sources={sources}
          />
        )}

        {usedSources.length > 0 && (
          <section className="sources-section">
            <div className="sources-heading">
              <div>
                <h3>Used in this answer</h3>
                <p>
                  These passages support the response.
                </p>
              </div>

              <span>
                {usedSources.length}{" "}
                {usedSources.length === 1
                  ? "source"
                  : "sources"}
              </span>
            </div>

            <div className="source-list">
              {usedSources.map((source) => (
                <SourceCard
                  key={source.id}
                  source={source}
                />
              ))}
            </div>
          </section>
        )}

        {otherSources.length > 0 && (
          <details className="other-sources">
            <summary>
              <span>
                Also retrieved, not used
              </span>

              <span>
                {otherSources.length} passages
              </span>
            </summary>

            <div className="source-list muted-source-list">
              {otherSources.map((source) => (
                <SourceCard
                  key={source.id}
                  source={source}
                />
              ))}
            </div>
          </details>
        )}

        {!message.loading &&
          message.responseTimeMs !== null &&
          message.responseTimeMs !== undefined && (
            <div className="response-meta">
              <span>
                Answered in{" "}
                {formatResponseTime(
                  message.responseTimeMs,
                )}
              </span>

              <Actions text={message.content} />
            </div>
          )}
      </div>
    </article>
  );
}

function AnswerWithCitations({
  text,
  sources,
}: {
  text: string;
  sources: ChatMessage["sources"];
}) {
  const normalizedSources = sources || [];

  const parts = text.split(
    /(\[\s*Evidence\s*\d+\s*\])/gi,
  );

  return (
    <>
      {parts.map((part, index) => {
        const match = part.match(
          /\[\s*Evidence\s*(\d+)\s*\]/i,
        );

        if (!match) {
          return (
            <span key={index}>
              {part}
            </span>
          );
        }

        const evidenceId = Number(match[1]);

        const source = normalizedSources.find(
          (item) => item.id === evidenceId,
        );

        if (!source) {
          return null;
        }

        return (
          <CitationChip
            key={`${evidenceId}-${index}`}
            source={source}
          />
        );
      })}
    </>
  );
}

function formatResponseTime(ms: number) {
  const seconds = ms / 1000;

  if (seconds < 1) {
    return "< 1 s";
  }

  return `${seconds.toFixed(1)} s`;
}

function Actions({ text }: { text: string }) {
  const [vote, setVote] = useState<"up" | "down" | null>(null);
  const [copied, setCopied] = useState(false);
  return (
    <div className="response-feedback">
      <button type="button" onClick={() => { void navigator.clipboard?.writeText(text); setCopied(true); window.setTimeout(() => setCopied(false), 1400); }}>
        {copied ? "Copied" : "Copy"}
      </button>
      <button type="button" aria-pressed={vote === "up"} onClick={() => setVote("up")}>Helpful</button>
      <button type="button" aria-pressed={vote === "down"} onClick={() => setVote("down")}>Not helpful</button>
    </div>
  );
}
