import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on :5173; proxy metrics to the backend (FastAPI or Axum) on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/stats": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/sse": "http://localhost:8000",
    },
  },
});
