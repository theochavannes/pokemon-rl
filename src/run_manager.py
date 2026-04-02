"""
Run management — creates and resumes numbered training runs.

Every training run gets an isolated directory: runs/run_NNN/
  models/          saved checkpoints (gitignored)
  logs/            TensorBoard logs (gitignored)
  replays/         battle replays for this run (gitignored)
  training_log.md  human-readable win rate table (committed)
  run_info.json    config + status snapshot (committed)

Auto-resume logic:
  - If the latest run has checkpoints but no "complete" marker → resume it.
  - Otherwise → create a new run.
  - Pass --new-run flag (or new_run=True) to force a new run regardless.
"""

import json
import re
from datetime import datetime
from pathlib import Path


RUNS_DIR = Path("runs")


class RunManager:
    def __init__(self, run_type: str, config: dict, new_run: bool = False):
        """
        run_type  : "curriculum" | "selfplay" — used in run_info.json
        config    : hyperparameters dict to record
        new_run   : force creation of a new run even if a resumable one exists
        """
        self.run_type = run_type
        self.config = config
        self.run_dir = self._resolve_run(new_run)
        self._init_dirs()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def models_dir(self) -> str:
        return str(self.run_dir / "models")

    @property
    def logs_dir(self) -> str:
        return str(self.run_dir / "logs")

    def replays_dir(self, phase: str = "") -> str:
        p = self.run_dir / "replays" / phase if phase else self.run_dir / "replays"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def training_log(self) -> str:
        return str(self.run_dir / "training_log.md")

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def latest_checkpoint(self) -> str | None:
        """Return path to the best available checkpoint in this run's models dir.

        Priority: highest-step periodic checkpoint > best_model.
        Returns path without .zip extension (SB3 convention).
        """
        models = Path(self.models_dir)

        # Periodic checkpoints from CheckpointCallback (ppo_pokemon_50000_steps.zip)
        best_step, best_path = -1, None
        for p in models.glob("ppo_pokemon_*_steps.zip"):
            m = re.search(r"ppo_pokemon_(\d+)_steps", p.stem)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best_path = str(p.with_suffix(""))

        # Fall back to best_model if no periodic checkpoint
        if best_path is None:
            best_model = models / "best_model.zip"
            if best_model.exists():
                best_path = str(best_model.with_suffix(""))

        return best_path

    def save_progress(self, phase_name: str, timesteps: int, epsilon: float | None = None) -> None:
        """Save resume state so interrupted runs can continue."""
        progress = {"phase": phase_name, "timesteps": timesteps}
        if epsilon is not None:
            progress["epsilon"] = epsilon
        self._update_info({"progress": progress})

    def load_progress(self) -> dict:
        """Load saved progress. Returns {} if no progress saved."""
        info = self._read_info(self.run_dir)
        return info.get("progress", {})

    def mark_complete(self) -> None:
        self._update_info({"status": "complete", "completed_at": datetime.now().isoformat()})
        print(f"  [{self.run_id}] marked complete")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_run(self, new_run: bool) -> Path:
        existing = self._all_runs()

        if not new_run and existing:
            latest = existing[-1]
            info = self._read_info(latest)
            if info.get("status") != "complete":
                # Has checkpoints or was never finished — resume
                ckpts = list((latest / "models").glob("ppo_pokemon_*_steps.zip"))
                if ckpts or (latest / "models").exists():
                    print(f"  Resuming {latest.name} (status: {info.get('status', 'in_progress')})")
                    return latest

        return self._create_run(existing)

    def _create_run(self, existing: list) -> Path:
        n = len(existing) + 1
        run_dir = RUNS_DIR / f"run_{n:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "run_id": run_dir.name,
            "run_type": self.run_type,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "config": self.config,
        }
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        print(f"  New run: {run_dir.name}")
        return run_dir

    def _init_dirs(self) -> None:
        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "replays").mkdir(parents=True, exist_ok=True)
        self._update_info({"status": "in_progress"})

    def _all_runs(self) -> list:
        if not RUNS_DIR.exists():
            return []
        runs = sorted(
            [d for d in RUNS_DIR.iterdir() if d.is_dir() and re.match(r"run_\d+", d.name)],
            key=lambda d: int(d.name.split("_")[1]),
        )
        return runs

    def _read_info(self, run_dir: Path) -> dict:
        p = run_dir / "run_info.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _update_info(self, updates: dict) -> None:
        p = self.run_dir / "run_info.json"
        info = json.loads(p.read_text()) if p.exists() else {}
        info.update(updates)
        p.write_text(json.dumps(info, indent=2))
