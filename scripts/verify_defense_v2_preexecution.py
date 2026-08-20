"""Render the public Defend v2 pre-execution status without starting an evaluation."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from apar.evaluation.v2_preexecution import verify_v2_preexecution
from apar.evaluation.v2_preregistration import V2Preregistration
from apar.runs.wire import canonical_json_bytes

_PUBLIC_PREREGISTRATION = base64.b64decode(
    b"eyJib290c3RyYXBfbWFuaWZlc3Rfc2hhMjU2IjoiMzMzYzA0ZGQxNTFhMmE2ODMxYzAzOWNiOWE2NTFk"
    b"ZjI5MTk4YmU4YTA0ZTE2Y2U4NjFkNGI2YTM0YTExYzk1NCIsImNhbmRpZGF0ZV9ncmlkX3NoYTI1NiI6"
    b"IjhjYTAwMzgwZGFmNzRiZTVhYWRiNzgyM2RjNDg3ZWJiZThiY2FiODViYWUzZmRiZDI0MDcwZGQ0NTEz"
    b"Mzc4MzIiLCJjb250cm9sc19tYW5pZmVzdF9zaGEyNTYiOiIxZTIxMzVkMWI1MGYxNGQzZDNkNmNiZTI0"
    b"YTQyOTRiY2I5ZTRkNWQzYzI4ZWIxY2U0OTgwZmYyYTE3YzVkM2I2IiwiZXZhbHVhdG9yX2NhcGFiaWxp"
    b"dHlfc2hhMjU2IjoiNWNhNjBkZjhjNzA5ZGU2OTdmYjNjYjI2ZTZmN2VlZjVkZTI0OTMwZDlmMjA2ZGIw"
    b"MWU0NWY3NTFmMzUzODcyZiIsImV2YWx1YXRvcl9rZXlfaWQiOiJkZTUyYzViN2QzOTY0MDU5OTBiOGRm"
    b"ODA4NzViYWQ0OWEyZTg0NWIwNmE1NzZlNjY4ZmIxZWEyOTJhNTVlZTdlIiwiZXZhbHVhdG9yX3B1Ymxp"
    b"Y19rZXlfYmFzZTY0IjoidGZrQWxUR1BQd1JNMk9RTFdFSlQyR2Qzdm1LL0J5SVROT1g5SVpuUHAvYz0i"
    b"LCJleGVjdXRpb25fbm9uY2UiOiI0YjRhOTI0MDVhOGM4NGFlNTAzNWJjYmM1MTBlMDZlMTcyOTIzOGZm"
    b"ZTBlNzhmMTA2NTE1ZDc4ZDNjNjNjOThhIiwiZmVhdHVyZV9tYW5pZmVzdF9zaGEyNTYiOiIyYWQ1NjIz"
    b"MTk3NjcxNTcwODdkZGEwZGVjNjM5MWY0NDc5ZjhhMDQ4NjlhYjBjYzhkM2E5YzM2MzdkYWU3M2I1Iiwi"
    b"ZmlkZWxpdHlfdmFsaWRhdGlvbl9idW5kbGVfc2hhMjU2IjoiZjYxMWJlYWI0NjQzN2M3NjljMmM0YTY2"
    b"NWQwYWI4ZmJlMjk4MjdjMWVjNDJjYjE3MzgzOGI2Njg2M2UwNzcxYyIsIm1heGltdW1fY29uZmlybWF0"
    b"b3J5X2F0dGVtcHRzIjoxLCJtZXRyaWNzX21hbmlmZXN0X3NoYTI1NiI6IjE3N2E3ZWEzNjExZmU2YjEz"
    b"ODU1N2RlYWY0NDk0MTYxNGI2NmMwZWNlMTc2NjMwODA0NGM4YjY1YzhiYTIxMjMiLCJwb3B1bGF0aW9u"
    b"X21hbmlmZXN0X3NoYTI1NiI6IjRmOWVkYzRkM2VkMGEwZTU3NDFhMWY1NWI5YThmNjhlZmZmNjA4OWRj"
    b"YzMwMDk1MTJlYWJkNzA5YzBkNDBiZjEiLCJwcmVyZWdpc3RyYXRpb25faWQiOiJhcGFyLWRlZmVuZC12"
    b"MiIsInJlcG9ydGluZ19zY2hlbWFfc2hhMjU2IjoiNjM3ZDdiZWNiOTk4MzkzN2Q1OGY1M2FmOTcyZGM4"
    b"NzM3M2YzYjdhMjAxMGNmYjY4MDU4ODBmMTQ2M2ZjZWY4ZCIsInNjaGVtYV92ZXJzaW9uIjoiMS4wLjAi"
    b"LCJzZWVkX2NvbW1pdG1lbnRzIjpbeyJjb21taXRtZW50X3NoYTI1NiI6IjRkM2Q1NDViZDUzNTc1YjEz"
    b"ODgzM2M1NDI5ZGMzODk1YzgzOTk5ZGQ3NGQxNDgxNjZmYWUzNTY4YWI2MGE4YjEiLCJuYW1lIjoib3Bl"
    b"cmF0aW5nX3BvcHVsYXRpb24ifSx7ImNvbW1pdG1lbnRfc2hhMjU2IjoiYzU5NjAxNjc1MjQ4NzE4Yjli"
    b"MjRhNzA0YjZkZGUzMmQ4NzliN2Q4ZTA0MTgyNTJjOTY1YThmMWQ1NmM0NDg4ZSIsIm5hbWUiOiJjYW1w"
    b"YWlnbl9pbmplY3Rpb24ifV0sInNpZ25hdHVyZV9iYXNlNjQiOiJxMVZjQ3BqL2g1UDRrVk5uR1NXVVgy"
    b"ekxDQ3pvOHA4RUJXQ0lxbXhMdFVJYWxnY1R6eW02MHlWVUpsWmdyZzcrZUdLWFR3WXNJTWllelFsSENL"
    b"TnhCZz09Iiwic291cmNlX21hbmlmZXN0X3NoYTI1NiI6IjQxY2Y2Nzk0YmE0MjAwYjgzOWM1MzUzMTU1"
    b"NWYwZjM5OThkZjRjYmIwMWE0ZDVjYjBiOTRlM2NhNWUyMzk0N2QiLCJzeW50aGV0aWNfc2NvcGUiOiJT"
    b"eW50aGV0aWMtb25seSBldmFsdWF0aW9uOyBub3QgYSByZWFsLXdvcmxkIHByZXZhbGVuY2Ugb3IgZXh0"
    b"ZXJuYWwtdmFsaWRpdHkgY2xhaW0uIiwic3ludGhldGljX3Njb3BlX3NoYTI1NiI6IjlkMTViZWIxMDk5"
    b"OGRjYWU3YjViYTc3NjVjMmM3YTkxMWRlNTdjYTA3NjBjYmE5OGQxNTNiYTVhMmIwODM0NDcifQ=="
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args()
    payload = (
        _PUBLIC_PREREGISTRATION
        if args.preregistration is None
        else args.preregistration.read_bytes()
    )
    preregistration = V2Preregistration.from_json(payload)
    report = verify_v2_preexecution(args.root.resolve(), preregistration)
    print(canonical_json_bytes(report.model_dump(mode="json")).decode("utf-8"))
    return 0 if report.admissible else 1


if __name__ == "__main__":
    raise SystemExit(main())
