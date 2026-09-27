import { test, expect } from "@playwright/test";

test("live shelf resumes persisted session and receives an external revision over SSE", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(page.getByText("PUBLIC DATA · LIVE SESSION")).toBeVisible();
  const shelf = page.getByTestId("recommendation-shelf");
  await expect(shelf.locator(".book-card")).toHaveCount(10);
  const revision = Number(await shelf.getAttribute("data-revision"));
  const token = await page.evaluate(
    () =>
      JSON.parse(sessionStorage.getItem("stacks-live-session")!)
        .token as string,
  );
  const initial = await request.get(
    `http://127.0.0.1:8001/v1/session/${token}`,
  );
  expect(initial.ok()).toBeTruthy();
  const payload = await initial.json();
  expect(payload.model_version).toMatch(/als|blend/);
  const item = payload.items[0];
  const event = await request.post(
    `http://127.0.0.1:8001/v1/session/${token}/event`,
    { data: { impression_id: item.impression_id, event: "click" } },
  );
  expect(event.ok()).toBeTruthy();
  // No page request initiated this event. Only the SSE stream can deliver it.
  await expect(shelf).toHaveAttribute("data-revision", String(revision + 1));
  const logs = await request.get(
    `http://127.0.0.1:8001/v1/session/${token}/logs`,
  );
  expect(logs.ok()).toBeTruthy();
  expect(JSON.stringify(await logs.json())).toContain(item.impression_id);
  await page.reload();
  await expect(page.getByText("PUBLIC DATA · LIVE SESSION")).toBeVisible();
  await expect(shelf).toHaveAttribute("data-revision", String(revision + 1));
});

test("unreachable API retains searchable static shelf and accessible trace", async ({
  page,
}) => {
  await page.route("http://127.0.0.1:8001/**", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("PUBLIC DATA · STATIC FALLBACK")).toBeVisible();
  const shelf = page.getByTestId("recommendation-shelf");
  await expect(shelf.locator(".book-card")).toHaveCount(10);
  await page.getByRole("button", { name: "Why this book" }).first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByLabel("Search books or authors").fill("Tolkien");
  await expect(shelf.locator(".book-card").first()).toContainText("Tolkien");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
