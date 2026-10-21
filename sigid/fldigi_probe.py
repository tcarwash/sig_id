"""Probe a running Fldigi process and report its catalog without claiming decode support."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .fldigi import FldigiClient, ModeCapability


def probe(url: str) -> dict[str, object]:
    client = FldigiClient(url)
    version = client.version()
    modes = [ModeCapability(name, catalog_present=True) for name in client.modem_names()]
    return {
        "backend": asdict(version),
        "current_mode": client.current_modem(),
        "catalog_count": len(modes),
        "modes": [asdict(mode) for mode in modes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7362/RPC2")
    args = parser.parse_args()
    print(json.dumps(probe(args.url), indent=2))


if __name__ == "__main__":
    main()
