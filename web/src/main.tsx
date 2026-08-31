import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./app/App";
import { loadConsoleData } from "./app/loader";
import "./styles.css";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("root element is missing");

const root = createRoot(rootElement);
root.render(<p role="status">Verifying local evidence…</p>);

loadConsoleData()
  .then(({ evidence, trace, traceMode }) => {
    root.render(
      <StrictMode>
        <App evidence={evidence} trace={trace} traceMode={traceMode} />
      </StrictMode>,
    );
  })
  .catch((error: unknown) => {
    const message = error instanceof Error ? error.message : "unknown failure";
    const retry = () => location.reload();
    root.render(
      <main>
        <h1>Evidence unavailable</h1>
        <p role="alert">The console stopped safely: {message}</p>
        <button onClick={retry} type="button">Retry local verification</button>
      </main>,
    );
  });
