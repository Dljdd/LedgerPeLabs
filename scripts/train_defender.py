#!/usr/bin/env python3
"""Train and freeze the preregistered synthetic defender."""

from apar.defense.orchestration import script_main

if __name__ == "__main__":
    script_main("train_defender")
