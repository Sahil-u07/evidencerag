import { useEffect, useState } from "react";
import EvidenceStack from "./EvidenceStack";

type EmptyStateProps = {
  onExampleQuestion: (question: string) => void;
  onAddDocument: () => void;
  hasDocuments: boolean;
};

const exampleQuestions = [
  "What is BM25 and how does it work?",
  "Which documents contain information about retrieval?",
  "How does hybrid search improve the results?",
];

export default function EmptyState({
  onExampleQuestion,
  onAddDocument,
  hasDocuments,
}: EmptyStateProps) {
  // Decorative loop that shows what the product does: search, find, check.
  const [stage, setStage] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setStage(5); return; }
    const seq: [number, number][] = [[0, 1400], [1, 1800], [2, 2300], [3, 2500], [5, 2800]];
    let i = 0, t = 0;
    const run = () => { setStage(seq[i][0]); t = window.setTimeout(() => { i = (i + 1) % seq.length; run(); }, seq[i][1]); };
    run();
    return () => window.clearTimeout(t);
  }, []);
  return (
    <section className="empty-state">
      <div className="empty-state-inner">
        <div
          className="empty-mark"
          aria-hidden="true"
        >
          <span />
          <span />
          <span />
        </div>

        <p className="empty-eyebrow">
          Document intelligence
        </p>

        <h1>
          Ask questions.
          <br />
          <span>Get answers you can trace.</span>
        </h1>

        <p className="empty-description">
          EvidenceRAG searches your documents, finds the
          most relevant passages, and checks that the
          answer is supported by evidence.
        </p>

        <EvidenceStack stage={stage} hits={3} />

        <div className="empty-steps">
          <div className={`empty-step${stage === 1 ? " is-on" : ""}`}>
            <span>1</span>
            <div>
              <strong>Ask</strong>
              <p>Ask a question in plain English.</p>
            </div>
          </div>

          <div className={`empty-step${stage === 2 ? " is-on" : ""}`}>
            <span>2</span>
            <div>
              <strong>Find</strong>
              <p>Relevant passages are retrieved.</p>
            </div>
          </div>

          <div className={`empty-step${stage === 3 || stage === 5 ? " is-on" : ""}`}>
            <span>3</span>
            <div>
              <strong>Check</strong>
              <p>Claims are checked against sources.</p>
            </div>
          </div>
        </div>

        {!hasDocuments && (
          <div className="empty-upload-card">
            <div>
              <strong>
                Start with your documents
              </strong>

              <p>
                Add a PDF, TXT, or Markdown file before
                asking your first question.
              </p>
            </div>

            <button
              type="button"
              onClick={onAddDocument}
            >
              Add document
            </button>
          </div>
        )}

        <div className="example-section">
          <div className="example-heading">
            <span>Try an example</span>
            <span>Click to ask</span>
          </div>

          <div className="example-list">
            {exampleQuestions.map((question) => (
              <button
                type="button"
                className="example-question"
                key={question}
                onClick={() =>
                  onExampleQuestion(question)
                }
              >
                <span>{question}</span>
                <span
                  className="example-arrow"
                  aria-hidden="true"
                >
                  →
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}