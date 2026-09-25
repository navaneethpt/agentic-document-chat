import type { NextConfig } from "next";

const config: NextConfig = {
  distDir: process.env.NEXT_TEST_BUILD === "1" ? ".next-e2e" : ".next",
  compress: false,
  devIndicators: false,
  experimental: { proxyClientMaxBodySize: "22mb", proxyTimeout: 900_000 },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${process.env.API_ORIGIN || "http://127.0.0.1:8000"}/api/:path*` }];
  },
};
export default config;
