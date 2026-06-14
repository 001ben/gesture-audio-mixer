import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];

export default defineConfig({
  base: process.env.GITHUB_ACTIONS && repositoryName ? `/${repositoryName}/` : "/",
  plugins: process.env.VITE_USE_HTTPS === "1" ? [basicSsl()] : []
});
