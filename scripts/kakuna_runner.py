"""Kakuna subprocess — runs in the metamon_ft conda env (poke-env 0.8.x + metamon).

Accepts N challenges from a named challenger, then prints JSON results to stdout.
Spawned by eval_vs_kakuna.py; do not run directly.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

_MFT_DEFAULT = os.path.expanduser("~/code/metamon-format-transfer")
_CACHE_DEFAULT = os.path.expanduser("~/.metamon")


async def run(args: argparse.Namespace) -> None:
    # Must be set before importing metamon
    os.environ.setdefault("METAMON_CACHE_DIR", os.path.expanduser(args.metamon_cache))

    mft_repo = os.path.expanduser(args.mft_repo)
    if mft_repo not in sys.path:
        sys.path.insert(0, mft_repo)

    from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: I001
    from src.neural_player import MetamonNeuralPlayer

    kwargs: dict = dict(
        battle_format=args.format,
        account_configuration=AccountConfiguration(args.username, None),
        log_level=40,
    )
    if args.server_port != 8000:
        from poke_env.ps_client.server_configuration import ServerConfiguration

        kwargs["server_configuration"] = ServerConfiguration(
            f"ws://localhost:{args.server_port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

    kakuna = MetamonNeuralPlayer(**kwargs)

    # Give poke-env time to connect and authenticate
    await asyncio.sleep(3)
    print("READY", flush=True)

    await kakuna.accept_challenges(args.challenger, args.n)

    result = {
        "wins": kakuna.n_won_battles,
        "losses": kakuna.n_lost_battles,
        "total": kakuna.n_won_battles + kakuna.n_lost_battles,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--format", default="gen1randombattle")
    parser.add_argument("--mft_repo", default=_MFT_DEFAULT)
    parser.add_argument("--metamon_cache", default=_CACHE_DEFAULT)
    parser.add_argument("--server_port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(run(args))
