import Tilt from "./Tilt";

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

        <div className="empty-steps">
          <Tilt className="empty-step">
            <span>1</span>
            <div>
              <strong>Ask</strong>
              <p>Ask a question in plain English.</p>
            </div>
          </Tilt>

          <Tilt className="empty-step">
            <span>2</span>
            <div>
              <strong>Find</strong>
              <p>Relevant passages are retrieved.</p>
            </div>
          </Tilt>

          <Tilt className="empty-step">
            <span>3</span>
            <div>
              <strong>Check</strong>
              <p>Claims are checked against sources.</p>
            </div>
          </Tilt>
        </div>

        {!hasDocuments && (
          <Tilt className="empty-upload-card" max={3}>
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
          </Tilt>
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