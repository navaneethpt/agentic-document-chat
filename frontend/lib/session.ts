import { api } from "./api";

export const SESSION_KEY = "folio-session";
export const WORKFLOW_KEY = "folio-workflow";

let creating: Promise<string> | null = null;

export async function getSession(): Promise<string> {
  const saved = sessionStorage.getItem(SESSION_KEY);
  if (saved) return saved;
  if (!creating) creating = api<{ id: string }>("/sessions", undefined, { method: "POST" })
    .then(({ id }) => { sessionStorage.setItem(SESSION_KEY, id); return id; })
    .finally(() => { creating = null; });
  return creating;
}

export function forgetSession(): void {
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(WORKFLOW_KEY);
}
