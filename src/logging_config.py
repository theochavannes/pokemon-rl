"""
Centralized logging configuration for training runs.

Provides:
  - Console + file logging (one log file per run)
  - DuplicateFilter: suppresses repeated identical messages (e.g. poke-env
    emitting the same warning hundreds of times per training run)
  - poke-env noise suppression: raises poke-env loggers to WARNING level

Usage (in train.py / selfplay_train.py):
    from src.logging_config import setup_logging
    logger = setup_logging(log_dir="runs/run_001/logs")
"""

import logging
import sys
from pathlib import Path


class DuplicateFilter(logging.Filter):
    """Suppresses repeated identical log messages.

    After a message is seen `max_repeats` times, it is silenced until a
    *different* message comes through, at which point the counter resets.
    A summary line ("suppressed N duplicates") is emitted when the burst ends.
    """

    def __init__(self, max_repeats: int = 3):
        super().__init__()
        self.max_repeats = max_repeats
        self._last_msg: str = ""
        self._count: int = 0
        self._suppressed: int = 0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg == self._last_msg:
            self._count += 1
            if self._count > self.max_repeats:
                self._suppressed += 1
                return False
            return True
        else:
            # New message — emit suppression summary if we silenced anything
            if self._suppressed > 0:
                summary = logging.LogRecord(
                    name=record.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"  [log] suppressed {self._suppressed} duplicate message(s)",
                    args=(),
                    exc_info=None,
                )
                # Emit directly to all handlers on the root logger
                for handler in logging.getLogger().handlers:
                    handler.emit(summary)
            self._last_msg = msg
            self._count = 1
            self._suppressed = 0
            return True


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a training run.

    Args:
        log_dir: Directory to write training.log into. If None, console-only.
        level: Root log level (default INFO).

    Returns:
        The 'pokemon_rl' logger, ready to use.
    """
    logger = logging.getLogger("pokemon_rl")

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — same output as before, but structured
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    console.addFilter(DuplicateFilter(max_repeats=3))
    logger.addHandler(console)

    # File handler — write everything to disk for post-run debugging
    if log_dir:
        log_path = Path(log_dir) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        file_handler.addFilter(DuplicateFilter(max_repeats=5))
        logger.addHandler(file_handler)

    # Suppress poke-env noise: raise to WARNING so per-battle info messages
    # (like "Vaporeon used ...") don't flood the console/log hundreds of times
    for name in ("poke_env", "poke-env", "websockets", "asyncio"):
        noisy_logger = logging.getLogger(name)
        noisy_logger.setLevel(logging.WARNING)
        noisy_logger.addFilter(DuplicateFilter(max_repeats=3))

    return logger
