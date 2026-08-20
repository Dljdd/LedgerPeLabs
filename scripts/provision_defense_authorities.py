#!/usr/bin/env python3
"""Provision the four one-shot defense-v1 signing authorities."""

from apar.defense.orchestration import script_main

if __name__ == "__main__":
    script_main("provision_defense_authorities")
