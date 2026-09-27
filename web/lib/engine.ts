import type { Book, Recommendation, Similarity } from "./types.ts";

const primaryAuthor = (book: Book) => book.author.split(",")[0].trim().toLowerCase();

export function recommend(
  catalog: Book[],
  similarity: Similarity,
  history: number[],
  genre: string,
  diverse: boolean,
  random: () => number = Math.random,
): Recommendation[] {
  const seen = new Set(history),
    byId = new Map(catalog.map((book) => [book.id, book]));
  const recent = history.slice(-8).reverse();
  const collaborative = new Map<number, number>(),
    affinity = new Map<string, number>();
  recent.forEach((id, index) => {
    const weight = Math.pow(0.78, index);
    (similarity[id] ?? []).forEach((other) =>
      collaborative.set(
        other.id,
        (collaborative.get(other.id) ?? 0) + other.score * weight,
      ),
    );
    const book = byId.get(id);
    if (book)
      affinity.set(book.genre, (affinity.get(book.genre) ?? 0) + weight);
  });
  const maxPop = Math.max(1, ...catalog.map((b) => b.popularity)),
    maxCollaborative = Math.max(0.001, ...collaborative.values()),
    maxAffinity = Math.max(1, ...affinity.values());
  const ranked = catalog
    .filter(
      (b) => !seen.has(b.id) && (genre === "All books" || b.genre === genre),
    )
    .map((book) => {
      const collab = (collaborative.get(book.id) ?? 0) / maxCollaborative,
        tag = (affinity.get(book.genre) ?? 0) / maxAffinity;
      const popularity = Math.log1p(book.popularity) / Math.log1p(maxPop);
      const score = history.length
        ? 0.62 * collab + 0.23 * tag + 0.15 * popularity
        : popularity;
      return { ...book, score, collaborative: collab, affinity: tag };
    })
    .sort((a, b) => b.score - a.score || a.id - b.id);
  const selected: typeof ranked = [];
  const pool = [...ranked];
  const limit = Math.min(10, ranked.length);
  while (selected.length < Math.max(0, limit - 1) && pool.length) {
    let best = -1,
      bestScore = -Infinity;
    pool.forEach((book, index) => {
      if (selected.filter((b) => primaryAuthor(b) === primaryAuthor(book)).length >= 2) return;
      const repetition = selected.length
        ? selected.filter((b) => b.genre === book.genre).length /
          selected.length
        : 0;
      const score = book.score - (diverse ? 0.2 * repetition : 0);
      if (score > bestScore) {
        best = index;
        bestScore = score;
      }
    });
    if (best < 0) break;
    selected.push(pool.splice(best, 1)[0]);
  }
  const explorationPool = pool
    .filter(
      (book) => selected.filter((b) => primaryAuthor(b) === primaryAuthor(book)).length < 2,
    )
    .slice(0, 20);
  const explored = explorationPool.length
    ? explorationPool[
        Math.min(
          explorationPool.length - 1,
          Math.floor(random() * explorationPool.length),
        )
      ]
    : null;
  if (explored) selected.push(explored);
  const lastBook = recent.length ? byId.get(recent[0]) : null;
  return selected.map((book, index) => {
    const exploration = book.id === explored?.id;
    const source = exploration
      ? "Uniform exploration"
      : book.collaborative > 0
        ? "Item similarity + session"
        : "Training popularity";
    const explanation = exploration
      ? "A little outside your usual shelf."
      : book.collaborative > 0 && lastBook
        ? `Connected to ${lastBook.title.split(" (")[0]}.`
        : book.affinity > 0
          ? `More from your ${book.genre.toLowerCase()} shelf.`
          : "A reader favorite in the training collection.";
    return {
      ...book,
      position: index + 1,
      propensity: exploration ? 1 / explorationPool.length : 1,
      explanation,
      source,
      exploration,
      impression_id: "",
    };
  });
}

export function csv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]);
  const escape = (v: unknown) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  return [
    keys.map(escape).join(","),
    ...rows.map((row) => keys.map((k) => escape(row[k])).join(",")),
  ].join("\n");
}
