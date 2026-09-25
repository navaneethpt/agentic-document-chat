import { expect, test, Page } from "@playwright/test";

async function upload(page: Page) {
  await page.goto("/");
  const input = page.getByLabel("Choose documents");
  await expect(input).toBeEnabled();
  await input.setInputFiles({ name: "launch.md", mimeType: "text/markdown", buffer: Buffer.from("Project Cedar launches in June.") });
  await page.getByRole("button", { name: "Process documents" }).click();
  await expect(page.getByText("1 searchable passages")).toBeVisible();
}
async function ask(page: Page, question: string) {
  await page.getByRole("textbox", { name: "Ask about your documents" }).fill(question);
  await page.getByRole("button", { name: "Send question" }).click();
}

test("upload, cited answer, historical evidence, refresh and clear", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await upload(page);
  await ask(page, "When does it launch?");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Source 1", exact: true }).click();
  await expect(page.locator(".source-card.selected")).toContainText("Project Cedar launches in June.");
  await ask(page, "What is the annual budget?");
  await expect(page.locator(".message.assistant").last()).toContainText("Please upload");
  await expect(page.locator(".message.assistant").last()).toContainText("annual budget document");
  await page.locator(".message.assistant").first().getByRole("button", { name: /View research/ }).click();
  await expect(page.locator(".right-panel")).toContainText("Answer checked and ready");
  await page.reload();
  await expect(page.locator(".message.assistant")).toHaveCount(2);
  await expect(page.locator(".right-panel")).toContainText("Round 3");
  await page.screenshot({ path: "test-results/workspace-desktop.png", fullPage: true });
  page.on("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Clear session" }).click();
  await expect(page.getByText("Your documents,", { exact: false })).toBeVisible();
  await expect(page.locator(".document")).toHaveCount(0);
});

test("failed file does not prevent good upload and session can expire", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await page.getByLabel("Choose documents").setInputFiles([
    { name: "bad.pdf", mimeType: "application/pdf", buffer: Buffer.from("bad") },
    { name: "good.txt", mimeType: "text/plain", buffer: Buffer.from("June launch") },
  ]);
  await page.getByRole("button", { name: "Process documents" }).click();
  await expect(page.locator(".failure")).toContainText("bad.pdf");
  await expect(page.getByText("1 searchable passages")).toBeVisible();
  await request.post("http://127.0.0.1:8100/test/expire");
  await page.reload();
  await expect(page.locator("main").getByRole("alert")).toContainText("expired");
  await page.getByRole("button", { name: "Start new session" }).click();
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  await expect(page.locator(".document")).toHaveCount(0);
});

test("reload during research recovers without submitting twice", async ({ page }) => {
  await upload(page);
  await ask(page, "When is launch?");
  await expect(page.getByRole("status")).toContainText(/Planning|Searching|Checking|Preparing/);
  await page.reload();
  await expect(page.locator(".message.assistant")).toHaveCount(1);
  await expect(page.locator(".message.user")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
});

test("failed chat keeps the question available for manual retry", async ({ page }) => {
  await upload(page);
  await page.route("**/api/chat", route => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Provider temporarily unavailable" }) }));
  await ask(page, "When is launch?");
  await expect(page.locator("main").getByRole("alert")).toContainText("Provider temporarily unavailable");
  await expect(page.getByRole("textbox", { name: "Ask about your documents" })).toHaveValue("When is launch?");
  await expect(page.getByRole("button", { name: "Send question" })).toBeEnabled();
  await expect(page.locator(".message.assistant")).toHaveCount(0);
  await page.unroute("**/api/chat");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".message.assistant")).toHaveCount(1);
});

test("mobile drawers, citation and layout", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Open documents" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("dialog").getByLabel("Choose documents").setInputFiles({ name: "launch.txt", mimeType: "text/plain", buffer: Buffer.from("Launch in June") });
  await page.getByRole("dialog").getByRole("button", { name: "Process documents" }).click();
  await expect(page.getByRole("dialog")).toContainText("1 searchable passages");
  await page.getByRole("button", { name: "Close panel" }).click();
  await ask(page, "When is launch?");
  await page.getByRole("button", { name: "Source 1", exact: true }).click();
  await expect(page.getByRole("dialog").locator(".source-card")).toContainText("June");
  await page.screenshot({ path: "test-results/workspace-mobile-sources.png", fullPage: true });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.screenshot({ path: "test-results/workspace-mobile.png", fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("proxy forwards files larger than the default Next.js 10 MB limit", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByLabel("Choose documents")).toBeEnabled();
  const id = await page.evaluate(() => sessionStorage.getItem("folio-session"));
  const response = await request.post("/api/documents", {
    headers: { "X-Session-ID": id! },
    multipart: { file: { name: "large.txt", mimeType: "text/plain", buffer: Buffer.alloc(11 * 1024 * 1024, "x") } },
  });
  expect(response.status()).toBe(200);
  expect(await response.json()).toMatchObject({ status: "failed", detail: "Too much extracted text. Split the document into smaller files." });
});

test("third-attempt 50 percent acceptance is visible in research", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await upload(page);
  await ask(page, "Give a partial answer about launch");
  await expect(page.getByRole("button", { name: "Source 1", exact: true })).toBeVisible();
  await expect(page.locator(".right-panel")).toContainText("Passed third-attempt confidence threshold");
  await expect(page.locator(".right-panel")).toContainText("Validator confidence: 50%");
  await expect(page.locator(".right-panel")).toContainText("Round 3");
});
