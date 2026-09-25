import { describe, expect, it } from "vitest";
import { readEvents } from "./api";

function stream(text: string, stride = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(controller) {
    for (let i = 0; i < bytes.length; i += stride) controller.enqueue(bytes.slice(i, i + stride));
    controller.close();
  } });
}
describe("SSE parser", () => {
  it("handles split frames, UTF-8 and keepalives", async () => {
    const events: unknown[] = [];
    await readEvents(stream(': keepalive\n\nevent: progress\ndata: {"query":"café"}\n\nevent: answer\ndata: {"content":"June"}\n\n'), event => events.push(event));
    expect(events).toEqual([{ event: "progress", data: { query: "café" } }, { event: "answer", data: { content: "June" } }]);
  });
  it("supports CRLF framing and error terminal events", async () => {
    const events: unknown[] = [];
    await readEvents(stream('event: error\r\ndata: {"detail":"Retry"}\r\n\r\n'), event => events.push(event));
    expect(events).toEqual([{ event: "error", data: { detail: "Retry" } }]);
  });
  it("rejects a truncated stream instead of submitting again", async () => {
    await expect(readEvents(stream('event: progress\ndata: {}\n\n'), () => {})).rejects.toThrow("interrupted");
  });
});
