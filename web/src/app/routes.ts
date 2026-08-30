import type { RoutePath } from "./types";

export const routes: { path: RoutePath; label: string; index: string }[] = [
  { path: "/overview", label: "Overview", index: "01" },
  { path: "/scenario", label: "Scenario", index: "02" },
  { path: "/replay", label: "Replay", index: "03" },
  { path: "/investigation", label: "Investigation", index: "04" },
  { path: "/defenses", label: "Defenses", index: "05" },
  { path: "/assurance", label: "Assurance", index: "06" },
];

export function isRoutePath(value: string): value is RoutePath {
  return routes.some((route) => route.path === value);
}
