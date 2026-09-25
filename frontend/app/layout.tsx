import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "Folio · Document research", description: "Ask your documents. Follow the evidence." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
