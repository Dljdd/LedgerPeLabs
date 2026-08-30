import { useEffect, useState } from "react";

import { isRoutePath } from "./routes";
import { Shell } from "./Shell";
import type { ConsoleEvidence, RoutePath, TraceMode, VerifiedTrace } from "./types";
import { Assurance } from "./views/Assurance";
import { Defenses } from "./views/Defenses";
import { Investigation } from "./views/Investigation";
import { Overview } from "./views/Overview";
import { Replay } from "./views/Replay";
import { Scenario } from "./views/Scenario";

interface AppProps {
  evidence: ConsoleEvidence;
  trace: VerifiedTrace;
  traceMode?: TraceMode;
}

function resolveRoute(): RoutePath {
  if (isRoutePath(location.pathname)) return location.pathname;
  history.replaceState({}, "", "/overview");
  return "/overview";
}

export function App({ evidence, trace, traceMode = "hash_bound_verified_fallback" }: AppProps) {
  const [route, setRoute] = useState<RoutePath>(resolveRoute);

  useEffect(() => {
    const onPopState = () => setRoute(resolveRoute());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = (path: RoutePath) => {
    if (path !== route) history.pushState({}, "", path);
    setRoute(path);
    window.scrollTo?.({ top: 0, behavior: "instant" });
  };

  let view;
  switch (route) {
    case "/overview": view = <Overview evidence={evidence} navigate={navigate} trace={trace} />; break;
    case "/scenario": view = <Scenario evidence={evidence} navigate={navigate} trace={trace} />; break;
    case "/replay": view = <Replay evidence={evidence} trace={trace} traceMode={traceMode} />; break;
    case "/investigation": view = <Investigation evidence={evidence} />; break;
    case "/defenses": view = <Defenses evidence={evidence} />; break;
    case "/assurance": view = <Assurance evidence={evidence} trace={trace} traceMode={traceMode} />; break;
  }

  return (
    <Shell navigate={navigate} route={route} traceMode={traceMode}>{view}</Shell>
  );
}
