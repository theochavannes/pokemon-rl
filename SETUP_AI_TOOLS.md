# AI Tools & Developer Workflow Setup

Prompt for Claude Code to automate setup of all recommended tools and practices for this repository.

---

## Context

This is a Pokemon RL project (Python 3.11, PyTorch CUDA, stable-baselines3, poke-env) developed by a solo developer using Claude Code. The repo is on GitHub at `theochavannes/pokemon-rl`. The goal is to set up modern tooling, CI/CD, and AI-assisted workflows to maximize development efficiency.

**IMPORTANT: Do NOT modify any project source code (src/, scripts/, tests/). Only create/modify config files, CI workflows, and tooling setup.**

---

## Step 1: Code Quality — Replace Pylint with Ruff

Ruff replaces Pylint, Flake8, Black, and isort in one tool. 10-100x faster (Rust).

### 1a. Create `pyproject.toml` with Ruff config + project metadata

```toml
[project]
name = "pokemon-rl"
version = "0.1.0"
description = "Reinforcement learning agent for Gen 1 Pokemon battles"
requires-python = ">=3.11"
dependencies = [
    "poke-env>=0.13.0",
    "stable-baselines3>=2.0",
    "sb3-contrib>=2.0",
    "torch>=2.0",
    "numpy",
    "gymnasium",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.11",
    "pytest>=8.0",
    "pytest-cov",
    "pytest-timeout",
    "pre-commit",
]

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
# E/F/W = pyflakes+pycodestyle, I = isort, UP = pyupgrade, B = bugbear, SIM = simplify
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 30
```

### 1b. Delete or rename `.github/workflows/pylint.yml`

Replace it in step 3.

---

## Step 2: Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: detect-private-key
```

Then run: `pip install pre-commit && pre-commit install`

---

## Step 3: GitHub Actions CI

Replace the single pylint workflow with a proper CI pipeline. Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v3
        with:
          args: check
      - uses: astral-sh/ruff-action@v3
        with:
          args: format --check

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Run tests
        run: pytest tests/ -v --timeout=30 --tb=short
```

Delete `.github/workflows/pylint.yml` after creating this.

---

## Step 4: Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "npm"
    directory: "/showdown"
    schedule:
      interval: "monthly"
```

---

## Step 5: Claude Code Hooks (auto-format on edit)

Add to `.claude/settings.json` (create if needed, merge with existing):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "ruff format --quiet \"$CLAUDE_FILE_PATH\" 2>/dev/null; ruff check --fix --quiet \"$CLAUDE_FILE_PATH\" 2>/dev/null; exit 0"
      }
    ]
  }
}
```

This auto-formats every Python file Claude edits.

---

## Step 6: MCP Servers

### Context7 — Real-time documentation for poke-env, SB3, PyTorch
```bash
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest
```

### Sequential Thinking — Structured reasoning for complex decisions
```bash
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
```

---

## Step 7: Claude Code Custom Commands

Create `.claude/commands/` directory with useful shortcuts:

### `.claude/commands/train.md`
```
Start the Showdown server in the background, then run training with the specified arguments.
1. cd showdown && node pokemon-showdown start --no-security &
2. Wait 3 seconds for server startup
3. Run: python src/train.py $ARGUMENTS
```

### `.claude/commands/benchmark.md`
```
Run the heuristic benchmark suite.
1. Ensure Showdown server is running (cd showdown && node pokemon-showdown start --no-security &)
2. Run: python scripts/benchmark_heuristic.py
3. Summarize the results in a table
```

---

## Step 8: Install Antigravity Awesome Skills (optional)

Community-curated library of 1,340+ Claude Code skills for common tasks (debugging, TDD, PR creation, security auditing, etc.).

```bash
npx antigravity-awesome-skills --claude
```

Useful skills for this project:
- `@debugging-strategies` — systematic troubleshooting
- `@test-driven-development` — TDD workflows
- `@create-pr` — clean PR creation
- `@security-auditor` — security reviews

---

## Step 9: CodeRabbit (GitHub App)

AI-powered code review on every PR. Free for public repos.

Install from: https://github.com/marketplace/coderabbitai
- Grant access to `theochavannes/pokemon-rl`
- No config needed — works out of the box
- Already installed and working on PR #1

---

## Execution Order

1. Create `pyproject.toml` (Step 1a)
2. Create `.pre-commit-config.yaml` (Step 2)
3. Create `.github/workflows/ci.yml` and delete `pylint.yml` (Step 3)
4. Create `.github/dependabot.yml` (Step 4)
5. Set up Claude Code hooks in `.claude/settings.json` (Step 5)
6. Create `.claude/commands/` shortcuts (Step 7)
7. Commit all changes with message: "Add modern tooling: Ruff, pre-commit, CI, Dependabot, Claude hooks"
8. Run `pip install pre-commit && pre-commit install` (user action)
9. Run MCP server setup commands (Step 6) (user action)
10. Install antigravity skills if desired (Step 8) (user action)

Steps 8-10 require user action in terminal — they cannot be committed to git.
