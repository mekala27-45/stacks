"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  ArrowRight,
  BookOpen,
  Bookmark,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Download,
  FlaskConical,
  Github,
  Layers3,
  Moon,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sun,
  X,
  ExternalLink,
  Play,
  Info,
  Activity,
  Table2,
} from "lucide-react";
import { recommend, csv } from "@/lib/engine";
import { useLiveSession } from "@/lib/live-session";
import { LiveEvidence, LiveMonitoring } from "./live-evidence";
import {
  EvidenceModules,
  ProtocolComparison,
  OpeEvidence,
} from "./evidence-tables";
import {
  basePath,
  statement,
  type Book,
  type Bundle,
  type Exposure,
  type Recommendation,
  type Interval,
} from "@/lib/types";

const navigation = [
  ["shelf", "The shelf"],
  ["evaluation", "Evaluation"],
  ["inflation", "The shortcuts"],
  ["ope", "Off-policy"],
  ["models", "Models"],
  ["explore", "SQL explorer"],
  ["report", "The report"],
];
const tones = [
  "wine",
  "ochre",
  "blue",
  "green",
  "purple",
  "orange",
  "teal",
  "olive",
];
const families = [
  "fiction",
  "fantasy",
  "science fiction",
  "mystery",
  "romance",
  "nonfiction",
  "classics",
  "young adult",
];
const number = (value: unknown, digits = 3) =>
  typeof value === "number"
    ? value !== 0 && Math.abs(value) < 0.001
      ? value.toExponential(2)
      : value.toFixed(digits)
    : "Unavailable";
function interval(value: unknown) {
  if (typeof value === "number") return number(value);
  if (!value || typeof value !== "object") return "Unavailable";
  const v = value as Record<string, number>;
  return `${number(v.mean ?? v.value ?? v.estimate)} [${number(v.low ?? v.lower)}, ${number(v.high ?? v.upper)}]`;
}
function rowsOf(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? (value as Record<string, unknown>[]) : [];
}
function download(name: string, content: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}
function tone(book: Book) {
  let index = families.indexOf(book.genre.toLowerCase());
  if (index < 0)
    index =
      Array.from(book.genre).reduce((sum, c) => sum + c.charCodeAt(0), 0) %
      tones.length;
  return tones[index];
}

export function Cover({
  book,
  small = false,
}: {
  book: Book;
  small?: boolean;
}) {
  return (
    <div className={`book-cover ${tone(book)} ${small ? "small-cover" : ""}`}>
      <div className="cover-top">
        THE STACKS COLLECTION <span>{String(book.id).padStart(4, "0")}</span>
      </div>
      <div className="cover-title">
        {book.title.replace(/\s*\([^)]*\)\s*$/, "")}
      </div>
      <div className="cover-rule" />
      <div className="cover-author">{book.author}</div>
      <div className="cover-bottom">
        <BookOpen size={small ? 13 : 18} strokeWidth={1.2} />
        <span>{book.genre}</span>
      </div>
    </div>
  );
}

export default function Stacks({ page }: { page: string }) {
  const [bundle, setBundle] = useState<Bundle | null>(null),
    [error, setError] = useState(""),
    [dark, setDark] = useState(false);
  const [history, setHistory] = useState<number[]>([]),
    [reader, setReader] = useState("visitor"),
    [genre, setGenre] = useState("All books"),
    [diverse, setDiverse] = useState(true),
    [query, setQuery] = useState("");
  const [shelf, setShelf] = useState<Recommendation[]>([]),
    [selected, setSelected] = useState<Book | null>(null),
    [trace, setTrace] = useState<Recommendation | null>(null),
    [saved, setSaved] = useState<number[]>([]),
    [showSaved, setShowSaved] = useState(false),
    [exposures, setExposures] = useState<Exposure[]>([]),
    [notice, setNotice] = useState(""),
    [ready, setReady] = useState(false);
  const sessionId = useRef(""),
    signature = useRef(""),
    shelfRef = useRef<Recommendation[]>([]);
  const live = useLiveSession(bundle?.catalog, reader, ready);
  useEffect(() => {
    if (live.mode !== "live") return;
    setShelf(live.shelf);
    shelfRef.current = live.shelf;
    if (live.info?.history) setHistory(live.info.history.map(Number));
    if (live.info?.genre) setGenre(live.info.genre);
  }, [live.mode, live.shelf, live.info]);
  useEffect(() => {
    let active = true;
    setDark(localStorage.getItem("stacks-theme") === "dark");
    const state = sessionStorage.getItem("stacks-session");
    if (state) {
      try {
        const s = JSON.parse(state);
        sessionId.current = s.session_id;
        setHistory(s.history ?? []);
        setReader(s.reader ?? "visitor");
        setSaved(s.saved ?? []);
        setExposures(s.exposures ?? []);
        shelfRef.current = s.shelf ?? [];
        setShelf(shelfRef.current);
      } catch {
        sessionStorage.removeItem("stacks-session");
      }
    }
    if (!sessionId.current) sessionId.current = crypto.randomUUID();
    Promise.all(
      ["catalog", "readers", "similarity", "evidence"].map((name) =>
        fetch(`${basePath}/data/${name}.json`).then((r) => {
          if (!r.ok)
            throw new Error(
              "The evidence bundle could not be loaded. Please reload.",
            );
          return r.json();
        }),
      ),
    )
      .then(([catalog, readers, similarity, evidence]) => {
        if (active) {
          setBundle({ catalog, readers, similarity, evidence });
          setReady(true);
        }
      })
      .catch((e) => setError(e.message));
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("stacks-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    if (!bundle || !ready) return;
    if (live.mode === "live") return;
    const nextSignature = JSON.stringify([
      history,
      reader,
      genre,
      diverse,
      live.mode,
    ]);
    if (signature.current === nextSignature) return;
    signature.current = nextSignature;
    if (page !== "shelf") return;
    const recommendations = recommend(
      bundle.catalog,
      bundle.similarity,
      history,
      genre,
      diverse,
    ).map((item) => ({ ...item, impression_id: crypto.randomUUID() }));
    setShelf(recommendations);
    shelfRef.current = recommendations;
    const timestamp = new Date().toISOString();
    setExposures((old) =>
      [
        ...old,
        ...recommendations.map((item) => ({
          impression_id: item.impression_id,
          session_id: sessionId.current,
          item_id: item.id,
          position: item.position,
          propensity: item.propensity,
          model_version: "browser-cosine-blend-v1",
          arm: diverse ? "diverse" : "baseline",
          timestamp,
          reward: 0,
          feedback_timestamp: null,
        })),
      ].slice(-3000),
    );
  }, [bundle, history, reader, genre, diverse, ready, page, live.mode]);
  useEffect(() => {
    if (!ready) return;
    sessionStorage.setItem(
      "stacks-session",
      JSON.stringify({
        session_id: sessionId.current,
        history,
        reader,
        saved,
        exposures,
        shelf,
      }),
    );
  }, [history, reader, saved, exposures, shelf, ready]);
  useEffect(() => {
    if (!notice) return;
    const id = setTimeout(() => setNotice(""), 4000);
    return () => clearTimeout(id);
  }, [notice]);
  const close = () => {
    setSelected(null);
    setTrace(null);
  };
  useEffect(() => {
    if (!selected && !trace) return;
    const previous = document.activeElement as HTMLElement | null;
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      if (e.key === "Tab") {
        const targets = [
          ...document.querySelectorAll<HTMLElement>(
            ".dialog button, .dialog a, .dialog summary",
          ),
        ].filter((el) => !el.hasAttribute("disabled"));
        if (!targets.length) return;
        const first = targets[0],
          last = targets[targets.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handle);
    document.querySelector<HTMLElement>(".dialog .close-button")?.focus();
    return () => {
      document.removeEventListener("keydown", handle);
      previous?.focus();
    };
  }, [selected, trace]);
  async function feedback(book: Book, event: "click" | "save") {
    const item = shelf.find((item) => item.id === book.id);
    if (live.mode === "live") {
      if (!item) {
        if (event === "save") {
          setSaved((old) => (old.includes(book.id) ? old : [...old, book.id]));
          setNotice(
            "Added to your saved shelf. This catalog browse is not a recommendation impression.",
          );
        } else
          setNotice(
            "Choose a recommendation from your current shelf to update the live session.",
          );
        return;
      }
      try {
        await live.feedback(item.impression_id, event);
      } catch (error) {
        setNotice(
          error instanceof Error
            ? error.message
            : "Your feedback could not be saved.",
        );
        return;
      }
    }
    if (item)
      setExposures((rows) =>
        rows.map((row) =>
          row.impression_id === item.impression_id
            ? {
                ...row,
                reward: 1,
                event,
                feedback_timestamp: new Date().toISOString(),
              }
            : row,
        ),
      );
    if (event === "save") {
      setSaved((old) => (old.includes(book.id) ? old : [...old, book.id]));
      setNotice("Added to your saved shelf.");
    } else {
      setHistory((old) =>
        [...old.filter((id) => id !== book.id), book.id].slice(-40),
      );
      setNotice(
        `Your shelf now takes ${book.title.split(" (")[0]} into account.`,
      );
      setSelected(null);
    }
  }
  function switchReader(value: string) {
    if (reader === value) {
      sessionStorage.removeItem("stacks-live-session");
      live.retry();
    }
    setReader(value);
    setHistory(
      value === "visitor"
        ? []
        : (bundle?.readers.find((r) => String(r.id) === value)?.history ?? []),
    );
    setGenre("All books");
    setShowSaved(false);
    setNotice(
      value === "visitor"
        ? "A fresh chapter. Start exploring."
        : `Showing public reader ${value}'s training history.`,
    );
  }
  const selectedRec = selected
    ? shelf.find((item) => item.id === selected.id)
    : null;
  const found = bundle
    ? bundle.catalog.filter(
        (b) =>
          (!showSaved || saved.includes(b.id)) &&
          `${b.title} ${b.author}`.toLowerCase().includes(query.toLowerCase()),
      )
    : [];
  const displayed = query || showSaved ? found.slice(0, 40) : shelf;
  const evidence = bundle?.evidence ?? {};
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="header">
        <div className="header-inner">
          <Link href="/" className="brand" aria-label="stacks home">
            <span className="brand-mark">
              <span />
              <span />
              <span />
            </span>
            stacks<span className="brand-period">.</span>
          </Link>
          <nav aria-label="Main navigation">
            {navigation.map(([id, label]) => (
              <Link
                className={page === id ? "active" : ""}
                key={id}
                href={id === "shelf" ? "/" : `/${id}/`}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="header-actions">
            <a
              href="https://github.com/mekala27-45/stacks"
              aria-label="Source on GitHub"
              target="_blank"
              rel="noreferrer"
            >
              <Github size={18} />
            </a>
            <button
              className="icon-button"
              aria-label={dark ? "Use light theme" : "Use dark theme"}
              onClick={() => setDark(!dark)}
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>
      <div className="edition-bar">
        <span>AN INDEPENDENT RECOMMENDATION BOOKSHOP</span>
        <span>
          <span className="mode-dot" /> PUBLIC DATA ·{" "}
          {live.mode === "live"
            ? "LIVE SESSION"
            : live.mode === "connecting"
              ? "CONNECTING"
              : "STATIC FALLBACK"}
        </span>
      </div>
      <main
        id="main"
        className={page === "shelf" ? "shelf-main" : "research-main"}
      >
        {error ? (
          <div className="empty-state">
            <BookOpen size={32} />
            <h1>We couldn’t open the collection.</h1>
            <p>{error}</p>
            <button
              className="button primary"
              onClick={() => location.reload()}
            >
              Try again
            </button>
          </div>
        ) : !bundle ? (
          <div
            className="loading"
            aria-label="Loading book collection"
            aria-busy="true"
          >
            <div className="skeleton-title" />
            <div className="book-grid">
              {Array.from({ length: 5 }, (_, i) => (
                <div className="skeleton-cover" key={i} />
              ))}
            </div>
          </div>
        ) : page === "shelf" ? (
          <>
            <section className="shelf-intro">
              <div>
                <div className="eyebrow">A NEW CHAPTER, CHOSEN FOR YOU</div>
                <h1>
                  Good books find
                  <br />
                  their <em>people.</em>
                </h1>
                <p>
                  A shelf that learns what you love.
                  <br className="mobile-break" /> And tells you why.
                </p>
              </div>
              <div className="intro-aside">
                <span className="aside-number">01 / THE SHELF</span>
                <p>
                  A familiar favorite.
                  <br />
                  An unexpected connection.
                  <br />
                  <em>Something worth opening.</em>
                </p>
                <Link href="/trace/">
                  See how your shelf is made <ArrowUpRight size={16} />
                </Link>
              </div>
            </section>
            <section className="reader-strip" aria-label="Reading preferences">
              <div className="reader-label">
                <span className="reader-icon">
                  <BookOpen size={21} />
                </span>
                <div>
                  <span className="overline">YOU’RE BROWSING AS</span>
                  <label className="select-wrap">
                    <select
                      aria-label="Select demonstration reader"
                      value={reader}
                      onChange={(e) => switchReader(e.target.value)}
                    >
                      <option value="visitor">A curious visitor</option>
                      {bundle.readers.map((r) => (
                        <option key={r.id} value={r.id}>
                          Public reader {r.id}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={15} />
                  </label>
                </div>
              </div>
              <div className="reader-description">
                {history.length
                  ? `${history.length} books in this demonstration history. Every click opens a new direction.`
                  : "No reading history yet. Open a book to make this shelf your own."}
              </div>
              <button
                className="text-button reset"
                onClick={() => switchReader("visitor")}
              >
                <RotateCcw size={14} /> Start fresh
              </button>
            </section>
            {history.length > 0 && (
              <div className="history-row">
                <span className="overline">ON YOUR READING TRAIL</span>
                {history
                  .slice(-5)
                  .reverse()
                  .map((id) => {
                    const book = bundle.catalog.find((b) => b.id === id);
                    return book ? (
                      <button key={id} onClick={() => setSelected(book)}>
                        {book.title.split(" (")[0]}
                        <ArrowUpRight size={12} />
                      </button>
                    ) : null;
                  })}
              </div>
            )}
            <div className="shelf-heading">
              <div>
                <h2>
                  {showSaved
                    ? "Your saved shelf"
                    : query
                      ? "From the collection"
                      : history.length
                        ? "Your next chapter"
                        : "A place to begin"}
                  <span className="heading-star">✳</span>
                </h2>
                <p>
                  {showSaved
                    ? "Keep a little reading room for later."
                    : query
                      ? `${found.length} books match your search.`
                      : history.length
                        ? "A little familiar. A little unexpected. All connected to your reading."
                        : "Reader favorites, with a little room for discovery."}
                </p>
              </div>
              <div className="shelf-controls">
                <button
                  className={`saved-toggle ${showSaved ? "selected" : ""}`}
                  onClick={() => {
                    setShowSaved(!showSaved);
                    setQuery("");
                  }}
                >
                  <Bookmark size={15} />
                  <span>Saved</span>
                  <b>{saved.length}</b>
                </button>
                <label className="search-field">
                  <Search size={16} />
                  <input
                    aria-label="Search books or authors"
                    placeholder="Find a book or author"
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value);
                      setShowSaved(false);
                    }}
                  />
                  {query && (
                    <button
                      aria-label="Clear search"
                      onClick={() => setQuery("")}
                    >
                      <X size={14} />
                    </button>
                  )}
                </label>
              </div>
            </div>
            {!query && !showSaved && (
              <div className="filter-row">
                <div className="genre-filters" aria-label="Filter by genre">
                  {[
                    "All books",
                    ...new Set(bundle.catalog.map((b) => b.genre)),
                  ].map((g) => (
                    <button
                      key={g}
                      className={genre === g ? "active" : ""}
                      onClick={async () => {
                        if (live.mode === "live") {
                          try {
                            await live.preferences(g);
                          } catch (error) {
                            setNotice(
                              error instanceof Error
                                ? error.message
                                : "Preference could not be saved.",
                            );
                            return;
                          }
                        }
                        setGenre(g);
                      }}
                    >
                      {g}
                    </button>
                  ))}
                </div>
                <label className="diversity-control">
                  <SlidersHorizontal size={14} />
                  <input
                    type="checkbox"
                    hidden={live.mode === "live"}
                    checked={
                      live.mode === "live"
                        ? live.info?.arm === "diverse"
                        : diverse
                    }
                    disabled={live.mode === "live"}
                    onChange={(e) => setDiverse(e.target.checked)}
                  />
                  {live.mode === "live"
                    ? `Assigned ${live.info?.arm ?? "model"} arm`
                    : "Room for discovery"}
                </label>
              </div>
            )}
            {displayed.length ? (
              <div
                className="book-grid"
                data-testid="recommendation-shelf"
                data-revision={
                  live.mode === "live" ? live.info?.revision : "local"
                }
              >
                {displayed.map((book, index) => {
                  const rec = shelf.find((r) => r.id === book.id);
                  return (
                    <article className="book-card" key={book.id}>
                      <button
                        className="cover-button"
                        onClick={() => setSelected(book)}
                        aria-label={`Open ${book.title}`}
                      >
                        <Cover book={book} />
                        <span className="cover-open">
                          <ArrowUpRight size={20} />
                        </span>
                      </button>
                      <div className="book-meta">
                        <span>{book.genre}</span>
                        <button
                          onClick={() => feedback(book, "save")}
                          aria-label={`Save ${book.title}`}
                          className={saved.includes(book.id) ? "is-saved" : ""}
                        >
                          {saved.includes(book.id) ? (
                            <Check size={16} />
                          ) : (
                            <Bookmark size={16} />
                          )}
                        </button>
                      </div>
                      <h3>
                        <button onClick={() => setSelected(book)}>
                          {book.title.replace(/\s*\([^)]*\)\s*$/, "")}
                        </button>
                      </h3>
                      <p className="author">{book.author}</p>
                      <p className="reason">
                        {rec?.exploration ? (
                          <span className="discovery-tag">
                            A LITTLE DISCOVERY
                          </span>
                        ) : null}
                        {rec?.explanation ??
                          `From the ${book.genre.toLowerCase()} collection.`}
                      </p>
                      {rec && (
                        <button
                          className="trace-link"
                          onClick={() => setTrace(rec)}
                        >
                          Why this book <ArrowUpRight size={13} />
                        </button>
                      )}
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state">
                <Bookmark size={30} />
                <h3>
                  {showSaved ? "A shelf with room to grow." : "No books found."}
                </h3>
                <p>
                  {showSaved
                    ? "Save a book using the bookmark beside its genre."
                    : "Try another title or author."}
                </p>
              </div>
            )}
            <section className="shelf-footnote">
              <div className="footnote-icon">
                <Layers3 size={24} strokeWidth={1.3} />
              </div>
              <div>
                <h3>Nothing up our sleeves. Just good recommendations.</h3>
                <p>
                  Follow a book from candidate to shelf. Inspect the evaluation,
                  the tradeoffs, and the numbers behind every choice.
                </p>
              </div>
              <Link className="button outline" href="/evaluation/">
                Read the evidence <ArrowUpRight size={16} />
              </Link>
            </section>
            <div className="local-note">
              <Info size={14} />
              <p>
                {live.mode === "live"
                  ? `Live ${live.info?.model_version} recommendations. Feedback and selection probabilities are saved by the API; shelf revisions arrive over a server stream. ${live.reason}`
                  : live.mode === "connecting"
                    ? "Connecting to the model service. The static collection is available while the session starts."
                    : `Static fallback: the model service could not be reached. This tab uses a separate local session blend. ${live.reason}`}
              </p>
              {live.mode === "fallback" && (
                <button className="text-button" onClick={live.retry}>
                  Reconnect
                </button>
              )}
              <button
                className="text-button"
                disabled={live.mode !== "live" && !exposures.length}
                onClick={async () => {
                  try {
                    download(
                      "stacks-session.csv",
                      csv(
                        live.mode === "live"
                          ? await live.logs()
                          : (exposures as unknown as Record<string, unknown>[]),
                      ),
                      "text/csv",
                    );
                  } catch (error) {
                    setNotice(
                      error instanceof Error ? error.message : "Export failed.",
                    );
                  }
                }}
              >
                <Download size={14} />
                {live.mode === "live"
                  ? "Export persisted impressions"
                  : `Export ${exposures.length} local impressions`}
              </button>
            </div>
          </>
        ) : (
          <>
            {["evaluation", "inflation", "models", "report"].includes(page) && (
              <div className="population-note">
                <b>THIS EVIDENCE RUN</b>{" "}
                {bundle.catalog.length.toLocaleString()} eligible books ·{" "}
                {Number(
                  (bundle.evidence.dataset as Record<string, unknown>)
                    ?.source_rows ?? 0,
                ).toLocaleString()}{" "}
                source ratings ·{" "}
                {Number(
                  (bundle.evidence.dataset as Record<string, unknown>)
                    ?.evaluated_users ?? 0,
                ).toLocaleString()}{" "}
                headline readers. Slice and shortcut cohorts are labeled
                separately. Training uses source order, which is a proxy without
                timestamps.
              </div>
            )}
            <Research
              page={page}
              bundle={bundle}
              exposures={exposures}
              shelf={shelf}
              showTrace={setTrace}
              live={live}
            />
          </>
        )}
      </main>
      <footer>
        <div>
          <Link href="/" className="footer-brand">
            stacks.
          </Link>
          <span>Every recommendation has a story.</span>
        </div>
        <p>{statement}</p>
        <div className="footer-links">
          <a
            href="https://github.com/mekala27-45/stacks"
            target="_blank"
            rel="noreferrer"
          >
            Source & methods <ArrowUpRight size={12} />
          </a>
          <span>GOODBOOKS-10K · CC BY-SA 4.0</span>
        </div>
      </footer>
      {notice && (
        <div className="toast" role="status">
          <Check size={17} />
          {notice}
        </div>
      )}
      {(selected || trace) && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) close();
          }}
        >
          <section
            className={`dialog ${trace ? "trace-dialog" : ""}`}
            role="dialog"
            aria-modal="true"
            aria-labelledby="dialog-title"
          >
            <button
              className="icon-button close-button"
              aria-label="Close dialog"
              onClick={close}
            >
              <X size={21} />
            </button>
            {trace ? (
              <Trace item={trace} />
            ) : (
              selected && (
                <>
                  <div className="detail-grid">
                    <Cover book={selected} />
                    <div className="detail-copy">
                      <span className="eyebrow">
                        {selected.genre} · {selected.year ?? "UNDATED"}
                      </span>
                      <h2 id="dialog-title">{selected.title}</h2>
                      <p className="detail-author">{selected.author}</p>
                      <div className="detail-rule" />
                      <p>
                        {selectedRec?.explanation ??
                          "Explore a connection in the public book collection."}
                      </p>
                      <p className="subtle">
                        Opening this book updates the demonstration session. Its
                        neighbors help shape your next shelf.
                      </p>
                      <button
                        className="button primary"
                        onClick={() => feedback(selected, "click")}
                      >
                        More like this <ArrowRight size={16} />
                      </button>
                      <button
                        className="button outline"
                        onClick={() => feedback(selected, "save")}
                      >
                        <Bookmark size={16} />
                        {saved.includes(selected.id)
                          ? "Saved to your shelf"
                          : "Save for later"}
                      </button>
                      {selectedRec && (
                        <button
                          className="text-button"
                          onClick={() => {
                            setTrace(selectedRec);
                            setSelected(null);
                          }}
                        >
                          Trace this recommendation <ArrowUpRight size={15} />
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="similar-row">
                    <h3>In good company</h3>
                    <div>
                      {(bundle?.similarity[selected.id] ?? [])
                        .slice(0, 3)
                        .map((sim) => {
                          const book = bundle?.catalog.find(
                            (b) => b.id === sim.id,
                          );
                          return book ? (
                            <button
                              key={book.id}
                              onClick={() => setSelected(book)}
                            >
                              <span className={`mini-spine ${tone(book)}`} />
                              <span>
                                {book.title.split(" (")[0]}
                                <small>{book.author}</small>
                              </span>
                              <ChevronRight size={16} />
                            </button>
                          ) : null;
                        })}
                    </div>
                  </div>
                </>
              )
            )}
          </section>
        </div>
      )}
    </>
  );
}

function Trace({ item }: { item: Recommendation }) {
  if (item.trace) {
    const trace = item.trace;
    const part = (name: string) =>
      (trace[name] ?? {}) as Record<string, unknown>;
    const retrieval = part("retrieval"),
      ranking = part("ranking"),
      reranking = part("reranking"),
      exploration = part("exploration"),
      shadow = part("shadow");
    const features = (ranking.features ?? {}) as Record<string, unknown>;
    const rules = (reranking.rules ?? {}) as Record<string, unknown>;
    const summaries: Record<string, [string, string]> = {
      retrieval: [
        "Start with the evaluated catalog",
        `${String(retrieval.candidate_count ?? 200)} leading eligible candidates from frozen model artifacts. Books already in the training or session history are excluded.`,
      ],
      ranking: [
        String(ranking.scoring_mode ?? "Evaluated model scores").replaceAll(
          "-",
          " ",
        ),
        `Model score ${number(item.score)}. ALS ${number(features.als)}; item cosine ${number(features.item_cosine)}; blend ${number(features.blend)}. The stored trace includes the exact weights.`,
      ],
      reranking: [
        "Keep the shelf within its rules",
        `At most ${String(rules.author_cap ?? 2)} books per primary author. Genre: ${String(rules.genre ?? "All books")}. This serving path keeps the evaluated score order; diversity and calibration experiments are reported separately.`,
      ],
      exploration: [
        item.exploration
          ? "Leave room for a randomized choice"
          : "Place the ranked choice",
        item.exploration
          ? `Position ${item.position} is sampled uniformly from ${Array.isArray(exploration.candidate_pool) ? exploration.candidate_pool.length : "the logged"} eligible books. Conditional probability ${number(item.propensity, 4)}. Reward: a click within ${String(exploration.reward_horizon_seconds ?? 60)} seconds.`
          : `Position ${item.position} is deterministic, with conditional probability 1. This position cannot support evaluation of other actions.`,
      ],
      shadow: [
        "Score an alternative without serving it",
        `${String(shadow.model_version ?? "Alternative model")}. Top-set disagreement ${number(shadow.set_disagreement)}. A shadow prediction is not observed reader feedback.`,
      ],
    };
    return (
      <>
        <div className="eyebrow">FROM EVALUATED MODEL TO LIVE SHELF</div>
        <h2 id="dialog-title">Why this book?</h2>
        <p className="trace-book-name">
          {item.title} <span>by {item.author}</span>
        </p>
        <div className="trace-timeline">
          {["retrieval", "ranking", "reranking", "exploration", "shadow"].map(
            (stage, index) => (
              <div className="trace-step" key={stage}>
                <span className="step-number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <span className="overline">{stage}</span>
                  <h3>{summaries[stage][0]}</h3>
                  <p>{summaries[stage][1]}</p>
                  <details>
                    <summary>Inspect stored {stage} trace</summary>
                    <pre className="trace-json">
                      {JSON.stringify(trace[stage] ?? {}, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            ),
          )}
        </div>
        <div className="data-note">
          <Code2 size={17} />
          <p>
            {item.model_version} · position {item.position} · conditional
            propensity {number(item.propensity, 4)}. This trace is returned with
            the persisted impression.
          </p>
        </div>
      </>
    );
  }
  return (
    <>
      <div className="eyebrow">FROM CANDIDATE TO SHELF</div>
      <h2 id="dialog-title">Why this book?</h2>
      <p className="trace-book-name">
        {item.title} <span>by {item.author}</span>
      </p>
      <div className="trace-timeline">
        {[
          [
            "01",
            "Retrieve",
            item.source,
            `Item cosine similarity ${number(item.collaborative)}. Genre affinity ${number(item.affinity)}. Training popularity ${item.popularity}.`,
          ],
          [
            "02",
            "Score",
            "An explicit, inspectable blend",
            `Session blend: 62% item similarity, 23% genre affinity, 15% log popularity. Without history: popularity only. Score ${number(item.score)}.`,
          ],
          [
            "03",
            "Make room",
            "Rules and diversity",
            "Already-read books are removed. No more than two books by one author. The diversity control applies a 0.20 penalty for repeated genres.",
          ],
          [
            "04",
            "Place",
            `Position ${item.position} · ${item.exploration ? "exploration slot" : "deterministic placement"}`,
            item.exploration
              ? `Uniform draw from the eligible pool. Conditional action propensity ${number(item.propensity, 4)}. This is a position-conditional probability, not a slate probability.`
              : "This deterministic position has conditional propensity 1. It provides no support for evaluating alternative actions.",
          ],
        ].map(([i, title, label, body]) => (
          <div className="trace-step" key={i}>
            <span className="step-number">{i}</span>
            <div>
              <span className="overline">{title}</span>
              <h3>{label}</h3>
              <p>{body}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="data-note">
        <Code2 size={17} />
        <div>
          <b>Browser inference · browser-cosine-blend-v1</b>
          <p>
            The shelf’s simple session blend is separate from the offline models
            in the evaluation. Local impressions can be exported from the shelf.
          </p>
        </div>
      </div>
    </>
  );
}

function PageHeading({
  section,
  title,
  children,
}: {
  section: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <span className="eyebrow">THE RESEARCH NOTES / {section}</span>
      <h1>{title}</h1>
      <p>{children}</p>
    </div>
  );
}
function DataTable({
  rows,
  columns,
}: {
  rows: Record<string, unknown>[];
  columns: [string, string][];
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map(([key], j) => (
                <td key={key} className={j === 0 ? "row-label" : "numeric"}>
                  {typeof row[key] === "object"
                    ? interval(row[key])
                    : typeof row[key] === "number"
                      ? number(row[key])
                      : String(row[key] ?? "Unavailable")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && (
        <p className="empty-table">This run has no results for this module.</p>
      )}
    </div>
  );
}
function Pushback({ children }: { children: React.ReactNode }) {
  return (
    <aside className="pushback">
      <span className="eyebrow">A FAIR QUESTION</span>
      <h3>“Wouldn’t a bestsellers list do?”</h3>
      <p>{children}</p>
    </aside>
  );
}

function Research({
  page,
  bundle,
  exposures,
  shelf,
  showTrace,
  live,
}: {
  page: string;
  bundle: Bundle;
  exposures: Exposure[];
  shelf: Recommendation[];
  showTrace: (r: Recommendation) => void;
  live: ReturnType<typeof useLiveSession>;
}) {
  const e = bundle.evidence as Record<string, unknown>;
  const metrics = rowsOf(e.metrics),
    protocol = (e.protocol ?? {}) as Record<string, unknown>,
    dataset = (e.dataset ?? {}) as Record<string, unknown>;
  const inflation = (e.inflation ?? {}) as Record<string, unknown>,
    ope = (e.ope ?? {}) as Record<string, unknown>;
  if (page === "explore") return <Explorer bundle={bundle} />;
  if (page === "trace")
    return (
      <>
        <PageHeading section="THE TRACE" title="Every choice, in the open.">
          Retrieval, scoring, rules, diversity, and exploration. Follow a book
          through the decisions that put it on the shelf.
        </PageHeading>
        {shelf.length ? (
          <div className="trace-selection">
            {shelf.map((item) => (
              <button key={item.id} onClick={() => showTrace(item)}>
                <span className="trace-position">
                  {String(item.position).padStart(2, "0")}
                </span>
                <div>
                  <b>{item.title}</b>
                  <small>
                    {item.source} · propensity {number(item.propensity, 4)}
                  </small>
                </div>
                <ArrowUpRight size={18} />
              </button>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <Layers3 size={30} />
            <h3>Your next shelf has a story.</h3>
            <p>
              Open the shelf first to create a session and trace its
              recommendations.
            </p>
            <Link href="/" className="button primary">
              Browse the shelf <ArrowRight size={16} />
            </Link>
          </div>
        )}
      </>
    );
  if (page === "evaluation")
    return (
      <>
        <PageHeading section="EVALUATION" title="Measure the recommendation.">
          A full-catalog test, a popularity baseline, and uncertainty beside
          every mean. Better recommendations begin with better questions.
        </PageHeading>
        <div className="protocol-banner">
          <FlaskConical size={22} />
          <div>
            <b>Source-order holdout · full-catalog ranking</b>
            <p>
              Goodbooks has no timestamps. Its published row order is an
              ordering proxy, not a verified calendar timeline. Headline results
              apply only to the documented subset.
            </p>
          </div>
          <a
            href="https://github.com/mekala27-45/stacks/blob/main/docs/evaluation.md"
            target="_blank"
            rel="noreferrer"
          >
            Protocol <ArrowUpRight size={15} />
          </a>
        </div>
        <div className="stat-grid">
          <Stat
            label="BOOKS IN THE CATALOG"
            value={bundle.catalog.length.toLocaleString()}
            caption="All eligible books are ranked"
          />
          <Stat
            label="EVALUATION USERS"
            value={String(
              dataset.evaluated_users ??
                dataset.test_users ??
                protocol.evaluation_users ??
                "See manifest",
            )}
            caption="Intervals resample held-out users"
          />
          <Stat
            label="HOLDOUT"
            value="Source order"
            caption="No fabricated dates or timestamps"
          />
          <Stat
            label="BASELINE"
            value="Popularity"
            caption="The comparison we must earn"
          />
        </div>
        <SectionTitle
          title="How the models compare"
          caption="Per-user means with 95% bootstrap intervals. No claim of a statistically corrected winner without a supporting comparison."
        />
        <MetricChart rows={metrics} />
        <DataTable
          rows={metrics.map((row) => ({
            ...row,
            name: row.label ?? row.model,
          }))}
          columns={[
            ["name", "Model"],
            ["ndcg", "NDCG @ 10 · 95% CI"],
            ["recall", "Recall @ 10 · 95% CI"],
            ["recall200", "Retrieval recall @ 200 · 95% CI"],
          ]}
        />
        <EvidenceModules evidence={e} />
        <Pushback>
          The popularity baseline belongs in the results because it may already
          be useful. Accuracy alone does not establish product value. This
          demonstration has no production experiment or evidence of revenue
          impact.
        </Pushback>
      </>
    );
  if (page === "inflation")
    return (
      <>
        <PageHeading
          section="THE SHORTCUTS"
          title="A better score. Or an easier test?"
        >
          Change the evaluation and the same model can tell a different story.
          These comparisons expose the effect of two common shortcuts.
        </PageHeading>
        <div className="protocol-banner">
          <Info size={22} />
          <div>
            <b>Diagnostic comparisons, never headline metrics</b>
            <p>
              The source-order split is the reference. Random splits and sampled
              negatives are shown here to measure sensitivity, not to
              manufacture a gain.
            </p>
          </div>
        </div>
        <SectionTitle
          title="01 / Shuffle the timeline"
          caption="Source-order holdout versus a random split. Both protocols are actually rerun on the same model family."
        />
        <ProtocolComparison rows={rowsOf(inflation.split)} kind="split" />
        <SectionTitle
          title="02 / Shrink the competition"
          caption="Full-catalog ranking versus sampled negatives. An easier candidate set can make a model look better without changing its recommendations."
        />
        <ProtocolComparison rows={rowsOf(inflation.sampled)} kind="sampled" />
        <div className="reading-note">
          <span className="eyebrow">THE PAPER BEHIND THE QUESTION</span>
          <h3>On Sampled Metrics for Item Recommendation</h3>
          <p>
            Sampled ranking metrics need not preserve model ordering. The size
            and direction of the change in this run are empirical results, not
            an assumption.
          </p>
          <a
            href="https://research.google/pubs/on-sampled-metrics-for-item-recommendation/"
            target="_blank"
            rel="noreferrer"
          >
            Read Krichene & Rendle’s paper <ArrowUpRight size={15} />
          </a>
        </div>
      </>
    );
  if (page === "ope")
    return (
      <>
        <PageHeading
          section="OFF-POLICY EVALUATION"
          title="Ask what might have happened."
        >
          Estimate a target policy from logged actions. First check against a
          simulator whose true value is known.
        </PageHeading>
        <div className="method-grid">
          {[
            [
              "IPS",
              "Reweight",
              "Correct each observed reward by its target-to-logging probability ratio.",
            ],
            [
              "SNIPS",
              "Normalize",
              "Normalize importance weights to trade finite-sample bias for stability.",
            ],
            [
              "DM",
              "Predict",
              "Average a reward model under the target policy. Misspecification can bias it.",
            ],
            [
              "DR",
              "Correct",
              "Combine reward predictions with propensity-weighted residuals.",
            ],
          ].map(([id, title, body]) => (
            <div key={id}>
              <span className="method-id">{id}</span>
              <h3>{title}</h3>
              <p>{body}</p>
            </div>
          ))}
        </div>
        <SectionTitle
          title="A simulator that knows the answer"
          caption="Bias, variance, and interval coverage across independent seeded logs. A deliberately misspecified reward model makes the assumptions visible."
        />
        <OpeEvidence ope={ope} />
        <LiveEvidence live={live} />
        <SectionTitle
          title="The static fallback session"
          caption="Local demonstration logs are position-conditional. They are not pooled across visitors and do not establish causal product lift."
        />
        <div className="stat-grid three">
          <Stat
            label="LOCAL IMPRESSIONS"
            value={String(exposures.length)}
            caption="Retained in this tab only"
          />
          <Stat
            label="OBSERVED FEEDBACK"
            value={String(exposures.filter((x) => x.reward > 0).length)}
            caption="Book opens and saves"
          />
          <Stat
            label="OFF-POLICY ESTIMATE"
            value="Not identified"
            caption="A target policy and reward horizon are required"
          />
        </div>
        <div className="data-note">
          <Info size={18} />
          <p>
            Deterministic slots lack support for unseen actions. The final slot
            explores uniformly within a limited candidate pool. An unbiased
            estimate over the whole catalog is not available from those logs.
          </p>
        </div>
        <Pushback>
          A wide interval is information, not a failure to hide. Neither a
          simulator nor a small public bandit sample proves that a new bookstore
          policy will improve real reader outcomes.
        </Pushback>
      </>
    );
  if (page === "models")
    return (
      <>
        <PageHeading
          section="MODELS & OPERATIONS"
          title="The model is only part of it."
        >
          A registry should refuse unsupported claims. See what is implemented,
          what was measured, and where this release stops.
        </PageHeading>
        <div className="model-cards">
          {metrics.map((m, index) => (
            <div className="model-card" key={index}>
              <div>
                <span className="model-index">
                  MODEL / {String(index + 1).padStart(2, "0")}
                </span>
                <span className="status-tag">Evaluated offline</span>
              </div>
              <h2>{String(m.label ?? m.model)}</h2>
              <p>
                {String(
                  m.description ??
                    "Evaluated against held-out interactions in the committed evidence bundle. Training items are excluded from the ranking.",
                )}
              </p>
              <div className="model-result">
                <span>NDCG @ 10</span>
                <b>{interval(m.ndcg)}</b>
              </div>
              <a
                className="text-button"
                href={`${basePath}/downloads/cards/${String(m.model)}.md`}
                download
              >
                Read model card <Download size={14} />
              </a>
            </div>
          ))}
        </div>
        <SectionTitle
          title="Promotion is an evidence decision"
          caption="The backend implements gates for accuracy, retrieval, coverage, tail share, calibration, latency, and schema compatibility. Missing evidence blocks promotion."
        />
        <div className="data-note">
          <Info size={18} />
          <div>
            <b>No production model is promoted by this website.</b>
            <p>
              The live shelf serves the frozen evaluated model artifacts. When
              the API is unavailable, the browser uses its disclosed fallback.
              Offline superiority and production impact are separate claims.
            </p>
          </div>
        </div>
        <CapabilityTable modules={rowsOf(e.modules)} />
        <LiveMonitoring live={live} />
        <SectionTitle
          title="Session monitoring"
          caption="Counts describe this browser tab. Click rates are not comparable until rewards have a defined observation window."
        />
        <DataTable
          rows={Array.from({ length: 10 }, (_, i) => {
            const rows = exposures.filter((x) => x.position === i + 1);
            return {
              position: i + 1,
              impressions: rows.length,
              feedback: rows.filter((x) => x.reward > 0).length,
            };
          })}
          columns={[
            ["position", "Position"],
            ["impressions", "Local impressions"],
            ["feedback", "Feedback events"],
          ]}
        />
        <Pushback>
          There is no measured production drift without production traffic.
          Monitoring endpoints and persistence tests are included in the
          runnable API. The live service persists demonstration observations;
          the local fallback is labeled separately.
        </Pushback>
      </>
    );
  return (
    <>
      <PageHeading section="THE REPORT" title="What the evidence says.">
        A reproducible record of this build. Every measured result comes from
        the committed evidence manifest.
      </PageHeading>
      <div className="report-actions">
        <button className="button primary" onClick={() => window.print()}>
          Print report <ExternalLink size={15} />
        </button>
        <button
          className="button outline"
          onClick={() =>
            download(
              "stacks-evidence.json",
              JSON.stringify(e, null, 2),
              "application/json",
            )
          }
        >
          <Download size={15} />
          Evidence manifest
        </button>
        <a
          className="text-button"
          href="https://github.com/mekala27-45/stacks/blob/main/RESULTS.md"
          target="_blank"
          rel="noreferrer"
        >
          Repository report <ArrowUpRight size={15} />
        </a>
      </div>
      <article className="paper">
        <div className="paper-kicker">
          STACKS / RECOMMENDATION SYSTEMS / REPRODUCIBLE RESEARCH
        </div>
        <h2>Measuring the measurement</h2>
        <p>
          The goal is to make recommendation quality inspectable: compare
          against popularity, separate retrieval from ranking, retain
          uncertainty, and expose the effects of convenient evaluation
          shortcuts.
        </p>
        <h3>Data and protocol</h3>
        <p>
          The catalog comes from Goodbooks-10k, shared under CC BY-SA 4.0.
          Evaluation uses a bounded subset and source-order holdout. The source
          supplies no timestamps; no verified calendar-time claim can be made.
          Evaluation removes each reader’s training books from the candidate
          set.
        </p>
        <DataTable
          rows={metrics.map((row) => ({
            ...row,
            name: row.label ?? row.model,
          }))}
          columns={[
            ["name", "Model"],
            ["ndcg", "NDCG @ 10 · 95% CI"],
            ["recall", "Recall @ 10 · 95% CI"],
          ]}
        />
        <h3>What the experiment does not show</h3>
        <ul>
          {(Array.isArray(e.limitations)
            ? e.limitations
            : [
                "This is a bounded public-data demonstration, not a production experiment.",
              ]
          ).map((x, i) => (
            <li key={i}>{String(x)}</li>
          ))}
        </ul>
        <h3>Serving and accountability</h3>
        <p>
          The public shelf recomputes recommendations in the browser using
          committed item similarities, recency-weighted history, genre affinity,
          and popularity. Each recommendation exposes its trace. Impressions and
          feedback remain in the current browser tab and can be exported. The
          repository also contains a separate runnable API with durable logging.
        </p>
        <h3>Reproduce the work</h3>
        <pre>
          uv sync --frozen{"\n"}uv run python scripts/build_evidence.py{"\n"}uv
          run python scripts/render_reports.py{"\n"}uv run pytest
        </pre>
        <p className="report-statement">{statement}</p>
      </article>
    </>
  );
}

function Stat({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="stat">
      <span className="overline">{label}</span>
      <strong>{value}</strong>
      <p>{caption}</p>
    </div>
  );
}
function SectionTitle({ title, caption }: { title: string; caption: string }) {
  return (
    <div className="section-title">
      <h2>{title}</h2>
      <p>{caption}</p>
    </div>
  );
}
function MetricChart({ rows }: { rows: Record<string, unknown>[] }) {
  const [table, setTable] = useState(false);
  const vals = rows.map((r) => ({
    label: String(r.label ?? r.model),
    value: (r.ndcg ?? {}) as Interval,
  }));
  const max = Math.max(0.01, ...vals.map((r) => r.value.high ?? 0)) * 1.1;
  return (
    <div className="chart-panel">
      <div className="chart-title">
        <b>NDCG @ 10</b>
        <button className="text-button" onClick={() => setTable(!table)}>
          <Table2 size={14} />
          {table ? "Show chart" : "Show values"}
        </button>
      </div>
      {table ? (
        <DataTable
          rows={rows.map((r) => ({ ...r, name: r.label ?? r.model }))}
          columns={[
            ["name", "Model"],
            ["ndcg", "Mean · 95% interval"],
          ]}
        />
      ) : (
        <div className="bar-chart">
          {vals.map((r, index) => (
            <div className="bar-row" key={r.label}>
              <span>{r.label}</span>
              <div className="bar-track">
                <div
                  className={`bar-fill series-${index}`}
                  style={{ width: `${((r.value.mean ?? 0) / max) * 100}%` }}
                  title={interval(r.value)}
                />
                <span
                  className="error-bar"
                  style={{
                    left: `${((r.value.low ?? 0) / max) * 100}%`,
                    width: `${(((r.value.high ?? 0) - (r.value.low ?? 0)) / max) * 100}%`,
                  }}
                />
              </div>
              <b>{number(r.value.mean)}</b>
            </div>
          ))}
          <div className="chart-axis">
            <span>0</span>
            <span>{number(max / 2)}</span>
            <span>{number(max)}</span>
          </div>
        </div>
      )}
      <p className="chart-caption">
        Bars show means; whiskers show user-bootstrap intervals. Read corrected
        pairwise comparisons before claiming a winner.
      </p>
    </div>
  );
}
function CapabilityTable({ modules }: { modules: Record<string, unknown>[] }) {
  return (
    <div className="capabilities">
      {modules.map((m, index) => (
        <div key={index}>
          <span
            className={`capability-dot ${String(m.status).includes("implemented") || String(m.status).includes("complete") ? "implemented" : ""}`}
          />
          <b>{String(m.name)}</b>
          <span className="status-tag">{String(m.status)}</span>
          <p>{String(m.detail ?? m.reason ?? "")}</p>
        </div>
      ))}
    </div>
  );
}

function Explorer({ bundle }: { bundle: Bundle }) {
  const [sql, setSql] = useState(
      "SELECT genre, count(*) AS books,\n       round(avg(popularity), 1) AS mean_training_positives\nFROM catalog\nGROUP BY genre\nORDER BY books DESC;",
    ),
    [rows, setRows] = useState<Record<string, unknown>[]>([]),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [elapsed, setElapsed] = useState("");
  const dbRef = useRef<import("@duckdb/duckdb-wasm").AsyncDuckDB | null>(null);
  const workerRef = useRef<Worker | null>(null);
  useEffect(
    () => () => {
      dbRef.current?.terminate();
      workerRef.current?.terminate();
    },
    [],
  );
  async function run() {
    setBusy(true);
    setError("");
    const start = performance.now();
    try {
      if (!dbRef.current) {
        const duckdb = await import("@duckdb/duckdb-wasm");
        const bundles = duckdb.getJsDelivrBundles();
        const selected = await duckdb.selectBundle(bundles);
        const blob = new Blob(
          [`importScripts(${JSON.stringify(selected.mainWorker!)});`],
          { type: "text/javascript" },
        );
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);
        workerRef.current = worker;
        URL.revokeObjectURL(url);
        const db = new duckdb.AsyncDuckDB(
          new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING),
          worker,
        );
        await db.instantiate(selected.mainModule, selected.pthreadWorker);
        await db.registerFileText(
          "catalog.json",
          JSON.stringify(bundle.catalog),
        );
        await db.registerFileText(
          "metrics.json",
          JSON.stringify(bundle.evidence.metrics ?? []),
        );
        const connection = await db.connect();
        await connection.query(
          "CREATE TABLE catalog AS SELECT * FROM read_json_auto('catalog.json'); CREATE TABLE metrics AS SELECT * FROM read_json_auto('metrics.json');",
        );
        await connection.close();
        dbRef.current = db;
      }
      if (
        !/^\s*(select|with|describe|show|explain)\b/i.test(sql) ||
        /;\s*\S/.test(sql)
      )
        throw new Error(
          "Use a single read-only SELECT, WITH, DESCRIBE, SHOW, or EXPLAIN statement.",
        );
      const connection = await dbRef.current.connect();
      try {
        const result = await connection.query(sql);
        setRows(
          result
            .toArray()
            .slice(0, 500)
            .map((row) =>
              JSON.parse(
                JSON.stringify(row, (_, v) =>
                  typeof v === "bigint" ? Number(v) : v,
                ),
              ),
            ),
        );
        setElapsed(`${((performance.now() - start) / 1000).toFixed(2)}s`);
      } finally {
        await connection.close();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "The query could not run.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeading
        section="SQL EXPLORER"
        title="Your question. The same evidence."
      >
        Run SQL in your browser against the committed book catalog and
        evaluation results. No data leaves this page.
      </PageHeading>
      <div className="explorer-layout">
        <aside>
          <span className="eyebrow">AVAILABLE TABLES</span>
          <h3>
            <Table2 size={17} />
            catalog
          </h3>
          <p>
            id · title · author · year
            <br />
            genre · tags · popularity
          </p>
          <h3>
            <Table2 size={17} />
            metrics
          </h3>
          <p>
            Model results with nested
            <br />
            bootstrap interval objects
          </p>
          <button
            className="text-button"
            onClick={() =>
              setSql(
                "SELECT model, ndcg.mean AS ndcg,\n       ndcg.low AS lower_95, ndcg.high AS upper_95\nFROM metrics\nORDER BY ndcg.mean DESC;",
              )
            }
          >
            Load evaluation query <ArrowRight size={13} />
          </button>
        </aside>
        <div>
          <label className="overline" htmlFor="sql-input">
            QUERY EDITOR
          </label>
          <textarea
            id="sql-input"
            spellCheck={false}
            value={sql}
            onChange={(e) => setSql(e.target.value)}
          />
          <div className="query-actions">
            <span>DuckDB-WASM · browser execution</span>
            <button className="button primary" disabled={busy} onClick={run}>
              <Play size={15} />
              {busy ? "Running…" : "Run query"}
            </button>
          </div>
          {error && (
            <div className="query-error" role="alert">
              {error}
            </div>
          )}
          <div className="query-result">
            <div>
              <b>
                {rows.length} rows {elapsed && `· ${elapsed}`}
              </b>
              <button
                className="text-button"
                disabled={!rows.length}
                onClick={() =>
                  download("stacks-query.csv", csv(rows), "text/csv")
                }
              >
                <Download size={14} />
                CSV
              </button>
            </div>
            {rows.length ? (
              <DataTable
                rows={rows}
                columns={Object.keys(rows[0]).map((k) => [k, k])}
              />
            ) : (
              <p className="subtle">Run a query to open the evidence.</p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
