import { test } from "node:test";
import assert from "node:assert/strict";
import { recommend, csv } from "../lib/engine.ts";
import type { Book } from "../lib/types.ts";
const catalog: Book[] = Array.from({ length: 35 }, (_, i) => ({
  id: i + 1,
  title: `Book ${i + 1}`,
  author: `Author ${Math.floor(i / 3)}`,
  year: 2000,
  genre: i % 2 ? "Fiction" : "Fantasy",
  tags: [],
  popularity: 35 - i,
}));
test("history is excluded, author cap applied and propensity records the actual exploration pool", () => {
  const result = recommend(
    catalog,
    {},
    [1, 2, 3],
    "All books",
    true,
    () => 0.99,
  );
  assert.equal(result.length, 10);
  assert.ok(result.every((x) => ![1, 2, 3].includes(x.id)));
  for (const book of result)
    assert.ok(result.filter((x) => x.author === book.author).length <= 2);
  assert.equal(new Set(result.map((x) => x.id)).size, 10);
  assert.equal(result.filter((x) => x.exploration).length, 1);
  assert.ok(result.slice(0, 9).every((x) => x.propensity === 1));
  assert.ok(result[9].propensity >= 1 / 20 && result[9].propensity <= 1);
});
test("click context changes scores toward collaborative neighbors", () => {
  const result = recommend(
    catalog,
    { "1": [{ id: 33, score: 1 }] },
    [1],
    "All books",
    true,
    () => 0,
  );
  assert.equal(result[0].id, 33);
  assert.equal(result[0].source, "Item similarity + session");
});
test("empty and exhausted filters return no made-up books", () => {
  assert.deepEqual(recommend(catalog, {}, [], "Absent genre", true), []);
  assert.deepEqual(
    recommend(
      catalog,
      {},
      catalog.map((b) => b.id),
      "All books",
      true,
    ),
    [],
  );
});
test("small pools keep valid positions and do not duplicate the exploration item", () => {
  const result = recommend(
    catalog.slice(0, 2),
    {},
    [],
    "All books",
    false,
    () => 0,
  );
  assert.deepEqual(
    result.map((x) => x.position),
    [1, 2],
  );
  assert.equal(result[1].propensity, 1);
});
test("CSV quotes commas, quotes and newlines", () => {
  assert.equal(
    csv([{ name: 'A,"B"', note: "line\nnext" }]),
    '"name","note"\n"A,""B""","line\nnext"',
  );
});
