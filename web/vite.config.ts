import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  build: { sourcemap: false },
  plugins: [tailwindcss(), react()],
  server: { host: "127.0.0.1", port: 4173, strictPort: true },
});
