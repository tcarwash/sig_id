"""Start the local browser workbench."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("sigid.api:create_app", factory=True, host=args.host, port=args.port)
