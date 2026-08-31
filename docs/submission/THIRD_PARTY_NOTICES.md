# APAR submission notices and dependency inventory

Project license status: unspecified.

This repository does not currently contain a project-level license grant. The
submission archive therefore makes no new licensing claim for APAR source or
model artifacts. Judges may inspect and run the archive for the competition use
case, subject to the competition's own terms.

The archive vendors no third-party package code, wheels, or `node_modules`. Its exact Python
runtime resolution is in `release/requirements-judge.txt`; the matching
CycloneDX record is in `release/dependency-sbom.cdx.json`. The packages are
downloaded from the judge's configured Python package index during first-time
setup and retain their upstream licenses.

| Package | Version | Declared license inventory |
|---|---:|---|
| annotated-types | 0.8.0 | MIT |
| catboost | 1.2.10 | Apache-2.0 |
| contourpy | 1.3.3 | BSD-3-Clause |
| cycler | 0.12.1 | BSD-3-Clause |
| fonttools | 4.63.0 | MIT |
| graphviz | 0.21 | MIT |
| joblib | 1.5.3 | BSD-3-Clause |
| kiwisolver | 1.5.1 | BSD-3-Clause |
| matplotlib | 3.11.1 | PSF-2.0 |
| narwhals | 2.25.0 | MIT |
| numpy | 2.5.2 | BSD-3-Clause |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | BSD-3-Clause |
| pillow | 12.3.0 | HPND |
| plotly | 7.0.0 | MIT |
| pydantic | 2.13.5 | MIT |
| pydantic-core | 2.46.5 | MIT |
| pyparsing | 3.3.2 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| six | 1.17.0 | MIT |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| tzdata | 2026.3 | Apache-2.0 |

The archive ships the React/TypeScript console source and the complete npm
dependency lock at `web/package-lock.json`. `npm ci` resolves the exact
transitive frontend graph during first-time setup; no third-party frontend code
is committed into the archive. Direct dependencies include React and React DOM
(MIT); development/build dependencies include TypeScript, Vite, Vitest,
Playwright, Testing Library, Axe, ESLint, Tailwind CSS, and their plugins. The
npm lock remains the authoritative version inventory for that toolchain.
