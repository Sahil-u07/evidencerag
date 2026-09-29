export type ProgressStage = {
  i: number;
  ms: number;
  label: string;
  complete: boolean;
};

export type EvidenceSource = {
  id: number;
  file: string;
  page: number | null;
  score: number;
  label:
    | "Best match"
    | "Good match"
    | "Weak match";
  text: string;
  used: boolean;
};

export type VerificationState = {
  supported: boolean;
  total: number;
  supportedClaims: number;
  reason: string;
};

export type DocumentItem = {
  name: string;
  status: string;
  size?: string | number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  query?: string;
  loading?: boolean;
  error?: boolean;
  stages?: ProgressStage[];
  sources?: EvidenceSource[];
  verification?: VerificationState | null;
  responseTimeMs?: number | null;
};

export type QueryResult = {
  answer: string;
  sources: EvidenceSource[];
  verification: VerificationState;
};
