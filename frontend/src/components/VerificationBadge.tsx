import type {
  EvidenceSource,
  VerificationState,
} from "../types";

type VerificationBadgeProps = {
  verification: VerificationState;
  sources: EvidenceSource[];
};

export default function VerificationBadge({
  verification,
  sources,
}: VerificationBadgeProps) {
  const isVerified = verification.supported;

  return (
    <details
      className={`verification-card ${
        isVerified
          ? "verification-success"
          : "verification-warning"
      }`}
    >
      <summary>
        <div className="verification-main">
          <span
            className="verification-icon"
            aria-hidden="true"
          >
            {isVerified ? "✓" : "!"}
          </span>

          <div>
            <strong>
              {isVerified
                ? "Checked: statements backed by sources"
                : "Some statements could not be fully verified"}
            </strong>

            <span>
              {verification.supportedClaims} of{" "}
              {verification.total} claims supported
            </span>
          </div>
        </div>

        <span className="verification-expand">
          Details
        </span>
      </summary>

      <div className="verification-details">
        <p>
          {verification.reason ||
            (isVerified
              ? "The answer was checked against the retrieved passages."
              : "Review the source passages before relying on unsupported statements.")}
        </p>

        {sources.length > 0 && (
          <div className="verification-source-map">
            {sources
              .filter((source) => source.used)
              .map((source) => (
                <button
                  type="button"
                  key={source.id}
                  onClick={() => {
                    const element =
                      document.getElementById(
                        `source-${source.id}`,
                      );

                    element?.scrollIntoView({
                      behavior: "smooth",
                      block: "center",
                    });
                  }}
                >
                  Evidence {source.id}
                  <span>
                    {source.file}
                  </span>
                </button>
              ))}
          </div>
        )}
      </div>
    </details>
  );
}