import type { ProgressStage } from "../types";

type ProgressStepsProps = {
  stages: ProgressStage[];
  loading: boolean;
  failed?: boolean;
};

export default function ProgressSteps({
  stages,
  loading,
  failed = false,
}: ProgressStepsProps) {
  if (!stages.length) {
    return null;
  }

  return (
    <div
      className="progress-steps"
      aria-label="Answer progress"
    >
      {stages.map((stage, index) => {
        const firstIncomplete =
          stages.findIndex(
            (item) => !item.complete,
          );

        const isActive =
          loading &&
          !stage.complete &&
          index === firstIncomplete;

        const isFailed =
          failed &&
          !stage.complete &&
          index === firstIncomplete;

        return (
          <div
            className={`progress-step ${
              stage.complete
                ? "is-complete"
                : isFailed
                  ? "is-failed"
                  : isActive
                    ? "is-active"
                    : ""
            }`}
            key={`${stage.i}-${stage.label}`}
          >
            <span
              className="progress-icon"
              aria-hidden="true"
            >
              {stage.complete
                ? "?"
                : isFailed
                  ? "!"
                  : ""}
            </span>

            <span className="progress-label">
              {stage.label}
            </span>

            {stage.ms > 0 && (
              <span className="progress-time">
                {formatStageTime(stage.ms)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatStageTime(ms: number) {
  if (ms < 1000) {
    return "<1 s";
  }

  return `${(ms / 1000).toFixed(1)} s`;
}
