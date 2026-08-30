import type { ReactNode } from "react";

import { Icon } from "./Icon";
import { routes } from "./routes";
import type { RoutePath, TraceMode } from "./types";

interface ShellProps {
  children: ReactNode;
  route: RoutePath;
  navigate: (path: RoutePath) => void;
  traceMode: TraceMode;
}

export function RouteLink({
  children,
  className = "",
  navigate,
  path,
}: {
  children: ReactNode;
  className?: string;
  navigate: (path: RoutePath) => void;
  path: RoutePath;
}) {
  return (
    <a
      className={className}
      href={path}
      onClick={(event) => {
        if (event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey) {
          event.preventDefault();
          navigate(path);
        }
      }}
    >
      {children}
    </a>
  );
}

export function Shell({ children, route, navigate, traceMode }: ShellProps) {
  const currentIndex = routes.findIndex((item) => item.path === route);
  const next = routes[currentIndex + 1];

  return (
    <div className="shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className="sidebar">
        <div className="brand" aria-label="APAR payment assurance">
          <span className="brand-mark" aria-hidden="true"><i /><i /></span>
          <span><strong>APAR</strong><small>Payment assurance</small></span>
        </div>
        <nav aria-label="Primary">
          <ol className="nav-list">
            {routes.map((item) => (
              <li key={item.path}>
                <RouteLink
                  className={item.path === route ? "nav-link is-active" : "nav-link"}
                  navigate={navigate}
                  path={item.path}
                >
                  <span className="nav-index">{item.index}</span>
                  <span>{item.label}</span>
                  {item.path === route ? <span className="sr-only">, current page</span> : null}
                </RouteLink>
              </li>
            ))}
          </ol>
        </nav>
        <div className="sidebar-foot">
          <span className="eyebrow">Runtime posture</span>
          <span className="runtime-state"><i aria-hidden="true" /> {traceMode === "live_local_scorer" ? "Local scorer · verified" : "Verified fallback · offline"}</span>
          <span className="runtime-arm">ensemble_with_graph</span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <p><span className="muted">Competition console</span><span className="slash">/</span> Synthetic evidence</p>
          <div className="top-status" aria-label="Evidence status: verified local inputs">
            <span className="status-dot" aria-hidden="true" />
            Verified local inputs
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>{children}</main>
        {next ? (
          <footer className="journey-next">
            <span><small>Continue the judge journey</small>{next.label}</span>
            <RouteLink className="round-link" navigate={navigate} path={next.path}>
              <span className="sr-only">Go to {next.label}</span><Icon name="arrow" />
            </RouteLink>
          </footer>
        ) : null}
      </div>
    </div>
  );
}
