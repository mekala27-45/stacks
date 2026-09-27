export type Book = {
  id: number;
  title: string;
  author: string;
  year: number | null;
  genre: string;
  tags: string[];
  popularity: number;
  rating?: number;
};
export type Reader = { id: number; history: number[] };
export type Similarity = Record<string, { id: number; score: number }[]>;
export type Interval = { mean: number; low: number; high: number };
export type Recommendation = Book & {
  score: number;
  position: number;
  propensity: number;
  explanation: string;
  impression_id: string;
  exploration: boolean;
  source: string;
  affinity: number;
  collaborative: number;
};
export type Exposure = {
  impression_id: string;
  session_id: string;
  item_id: number;
  position: number;
  propensity: number;
  model_version: string;
  arm: string;
  timestamp: string;
  reward: number;
  feedback_timestamp: string | null;
  event?: string;
};
export type Bundle = {
  catalog: Book[];
  readers: Reader[];
  similarity: Similarity;
  evidence: Record<string, unknown>;
};
export const statement =
  "Demonstration recommendations on a public dataset. No real reader identity is present; recommendations are not personalized to a real person.";
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
