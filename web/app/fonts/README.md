# Bundled fonts

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

The three WOFF2 files are the exact Latin-subset font bytes emitted by the last successful `next/font/google` build. `next/font/local` now includes them without downloading Google Fonts CSS during a build. The original families, weight ranges, display policy and CSS variables are preserved. Characters outside the bundled Latin subsets use the browser's fallback fonts.

`provenance.json` records each original Next.js asset filename, SHA256, byte size and official source. Font metadata was checked using Next.js's bundled font parser: Literata, Work Sans and Spline Sans Mono, all upright variable fonts. The mono file supports weights 300 through 700; the application retains its existing 400 through 500 range.

Each family is distributed under the SIL Open Font License 1.1. The accompanying family-specific OFL files were downloaded from the official [Google Fonts repository](https://github.com/google/fonts), preserving their copyright notices and complete license text. These font licenses are separate from the application's MIT license.
