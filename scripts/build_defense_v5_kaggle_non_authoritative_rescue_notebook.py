"""Build the private, explicitly non-authoritative Kaggle OOM-rescue notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = Path("/private/tmp/apar-v5-notebooks.jCJuKB/defense_v5/70_metrics.ipynb")
DESTINATION = Path("/private/tmp/apar-sentinel-v5-rescue-v4.ipynb")
HELPER = ROOT / "src/apar/evaluation/v5_kaggle_rescue.py"
RUNNER = ROOT / "scripts/run_defense_v5_kaggle_non_authoritative_rescue.py"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encoded(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_bytes())
    if not isinstance(notebook, dict) or len(notebook.get("cells", [])) != 3:
        raise RuntimeError("approved Stage 70 notebook template differs")

    helper_payload = HELPER.read_bytes()
    runner_text = RUNNER.read_text()
    source_import = "from apar.evaluation.v5_kaggle_rescue import ("
    if runner_text.count(source_import) != 1:
        raise RuntimeError("rescue runner helper import count differs")
    runner_payload = runner_text.replace(
        source_import,
        "from v5_kaggle_rescue import (",
    ).encode()

    cell_source = f"""import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OVERLAY_ROOT = Path("/kaggle/working/apar-v5-rescue-overlay")
OVERLAY_ROOT.mkdir(parents=True, exist_ok=False)


def _write_rescue_source(name: str, encoded: str, expected_sha256: str) -> None:
    payload = base64.b64decode(encoded.encode())
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("non-authoritative rescue source binding differs")
    (OVERLAY_ROOT / name).write_bytes(payload)


_write_rescue_source(
    "v5_kaggle_rescue.py",
    "{_encoded(helper_payload)}",
    "{_sha256(helper_payload)}",
)
_write_rescue_source(
    "rescue_runner.py",
    "{_encoded(runner_payload)}",
    "{_sha256(runner_payload)}",
)

PREDECESSOR_CHAIN = (
    Path("/kaggle/input/notebooks/dylanmoraes")
    / "apar-sentinel-v5-60-single-class-controls"
    / "apar-v5-chain"
)
if not PREDECESSOR_CHAIN.is_dir():
    raise RuntimeError("accepted Stage 60 predecessor chain is absent")

OUTPUT_ROOT = Path("/kaggle/working/apar-v5-rescue")
child_environment = os.environ.copy()
child_environment["PYTHONPATH"] = (
    str(OVERLAY_ROOT)
    + (
        os.pathsep + child_environment["PYTHONPATH"]
        if child_environment.get("PYTHONPATH")
        else ""
    )
)
child_environment.update(
    {{
        "APAR_V5_RESCUE_PARENT_SOURCE_ROOT": str(SOURCE_ROOT),
        "APAR_V5_RESCUE_PARENT_PREDECESSOR_CHAIN": str(PREDECESSOR_CHAIN),
        "APAR_V5_RESCUE_PARENT_OUTPUT_ROOT": str(OUTPUT_ROOT),
    }}
)

CHILD_PROGRAM = r'''
import json
import os
from pathlib import Path

from rescue_runner import run_non_authoritative_rescue

receipt = run_non_authoritative_rescue(
    root=Path(os.environ["APAR_V5_RESCUE_PARENT_SOURCE_ROOT"]),
    predecessor_chain=Path(os.environ["APAR_V5_RESCUE_PARENT_PREDECESSOR_CHAIN"]),
    output_root=Path(os.environ["APAR_V5_RESCUE_PARENT_OUTPUT_ROOT"]),
    approved_source_commit="40fb4a131da36556d2b8a04564cb62d73152c7c1",
    execution_mode="kaggle_capacity_validation",
)
print(
    json.dumps(
        {{
            "non_authoritative_rescue_complete": True,
            "authoritative": False,
            "accepted_capacity_evidence": False,
            "receipt": receipt,
        }},
        sort_keys=True,
    ),
    flush=True,
)
'''
subprocess.run(
    [sys.executable, "-c", CHILD_PROGRAM],
    check=True,
    env=child_environment,
    text=True,
)
"""

    notebook["cells"][2] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_source,
    }
    for cell in notebook["cells"]:
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["metadata"]["apar"] = {
        "generator_schema": "apar-sentinel-v5-non-authoritative-rescue-notebook/1",
        "stage": "70_metrics_non_authoritative_rescue_v4",
        "authoritative": False,
        "accepted_capacity_evidence": False,
    }
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {
                "notebook": str(DESTINATION),
                "notebook_sha256": _sha256(DESTINATION.read_bytes()),
                "helper_sha256": _sha256(helper_payload),
                "rewritten_runner_sha256": _sha256(runner_payload),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
