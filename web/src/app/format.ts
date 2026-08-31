export function formatPercent(value: number, digits = 1): string {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatMoney(value: number | string, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

export function formatMetric(value: number, metric: string): string {
  if (metric.includes("rate") || metric === "f1" || metric === "precision" || metric === "recall" || metric.includes("fraction")) {
    return formatPercent(value, value < 0.001 ? 3 : 1);
  }
  return `${value.toFixed(2)} ms`;
}

export function shortHash(value: string, leading = 10): string {
  return `${value.slice(0, leading)}…${value.slice(-6)}`;
}

export function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
