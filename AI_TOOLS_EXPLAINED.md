# AI Tools & Workflow — Why Each One Matters

A plain-language explanation of every tool recommended for this project, what problem it solves, and how it fits into the Pokemon RL development workflow.

---

## 1. Ruff (replaces Pylint, Black, Flake8, isort)

**What it is:** A Python linter and formatter written in Rust. One tool that replaces 4+ separate tools.

**What problem it solves:** Right now we have a `pylint.yml` GitHub Action that runs Pylint with half its rules disabled (`--disable=C,R`) and a low bar (`--fail-under=7.0`). It catches some errors but misses formatting inconsistencies, unsorted imports, and security issues. We also have no auto-formatter — every file could look different.

**Why Ruff specifically:**
- **10-100x faster than Pylint** — written in Rust, lints the entire codebase in milliseconds instead of seconds
- **Replaces 4 tools in one config** — linting (Pylint/Flake8), formatting (Black), import sorting (isort), and code modernization (pyupgrade). All configured in one `[tool.ruff]` section in `pyproject.toml`
- **Auto-fix** — `ruff check --fix` automatically fixes many issues (unused imports, wrong import order, deprecated syntax)
- It's become the de facto standard for Python projects in 2025-2026. Most major open-source projects (FastAPI, Pydantic, Django) have switched

**How it fits our workflow:** Every time you save a file or Claude edits one, Ruff auto-formats it. In CI, it blocks PRs with lint errors. No more style debates or inconsistent code.

---

## 2. Pre-commit Hooks

**What it is:** A framework that runs checks automatically before every `git commit`. If a check fails, the commit is blocked until you fix it.

**What problem it solves:** Without pre-commit hooks, bad code gets committed and you only find out when CI fails (minutes later, after you've moved on). With hooks, problems are caught instantly, before they enter git history.

**What our hooks do:**
- **`ruff --fix`** — auto-fixes lint issues in staged files
- **`ruff-format`** — auto-formats staged files
- **`trailing-whitespace`** — removes trailing spaces
- **`end-of-file-fixer`** — ensures files end with newline
- **`check-added-large-files (500KB)`** — **critical for ML projects** — prevents accidentally committing model checkpoints (`.pt`, `.zip` files that are 100MB+). One accidental `git add .` with a model file would bloat the repo permanently
- **`detect-private-key`** — catches accidentally committed credentials

**How it fits our workflow:** You run `pre-commit install` once. After that, every `git commit` automatically formats and checks your code. You never push broken formatting or accidentally commit a 200MB model checkpoint.

---

## 3. GitHub Actions CI (replacing pylint.yml)

**What it is:** Automated checks that run on every push and pull request on GitHub.

**What problem it solves:** Our current CI only runs Pylint. It doesn't run tests, doesn't check formatting, and installs dependencies without caching (slow). The new CI pipeline has two parallel jobs:

**Lint job:**
- Uses the official `astral-sh/ruff-action` — faster and more thorough than our Pylint setup
- Checks both linting rules AND formatting in one pass
- Runs in ~5 seconds vs ~30 seconds for Pylint

**Test job:**
- Runs `pytest` on our 23+ unit tests
- Uses pip caching so dependency install is fast after the first run
- Installs CPU-only PyTorch (smaller, faster, no GPU needed for unit tests)
- `--timeout=30` prevents hung tests from blocking CI forever

**How it fits our workflow:** When you create a PR (like the codebase-audit PR), both lint and tests run automatically. CodeRabbit reviews the logic, CI verifies the code actually works. You merge with confidence.

---

## 4. Dependabot

**What it is:** A GitHub bot that automatically creates PRs to update dependencies when new versions are released or security vulnerabilities are found.

**What problem it solves:** Dependencies go stale. A library you use gets a security fix, but you don't know about it until something breaks. Dependabot watches for:
- **GitHub Actions updates** — e.g., `actions/checkout@v4` → `@v5`
- **npm updates** for the Showdown submodule
- **Security advisories** — if poke-env or PyTorch has a CVE, you get a PR immediately

**Why it matters for this project:** We depend on poke-env, stable-baselines3, sb3-contrib, PyTorch, and a full Node.js Pokemon Showdown server. That's a lot of surface area for security issues.

**How it fits our workflow:** You do nothing. Dependabot opens PRs automatically. CI runs on those PRs. If tests pass, you merge with one click.

---

## 5. Claude Code Hooks (auto-format on edit)

**What it is:** Shell commands that run automatically when Claude Code uses certain tools (Edit, Write, etc.).

**What problem it solves:** When Claude edits Python files, the formatting might not match your project style. Without hooks, you'd need to manually run `ruff format` after every Claude edit. With hooks, every file Claude touches is automatically formatted.

**The key difference from CLAUDE.md instructions:** CLAUDE.md is "advisory" — Claude tries to follow it but might not always. Hooks are **deterministic** — they run 100% of the time, on every edit, no exceptions. It's the difference between asking someone to format their code vs having an auto-formatter run on save.

**How it fits our workflow:** You're using Claude Code as your primary development tool. Every single edit Claude makes gets auto-formatted. The code always looks clean.

---

## 6. MCP Servers (Context7, Sequential Thinking)

**What it is:** MCP (Model Context Protocol) servers are plugins that give Claude Code access to external data sources and capabilities in real-time.

### Context7
**What problem it solves:** Claude's training data has a knowledge cutoff. When you ask about poke-env's `SinglesEnv` API or SB3's `MaskablePPO` parameters, Claude might give you slightly outdated information or hallucinate parameter names. Context7 fetches the **real, current documentation** from the actual library repos and feeds it to Claude.

**Concrete example:** If you ask "what parameters does MaskablePPO.learn() accept?", without Context7, Claude answers from memory (possibly outdated). With Context7, it pulls the actual current docstring from the sb3-contrib repo.

### Sequential Thinking
**What problem it solves:** For complex architectural decisions (like "should we change the observation space?" or "how should reward shaping work?"), Claude sometimes jumps to a solution too fast. Sequential Thinking forces a structured reasoning loop — it breaks the problem into steps, considers alternatives, and arrives at a more thorough answer.

**How they fit our workflow:** These run in the background. You don't interact with them directly — Claude uses them automatically when relevant.

---

## 7. Claude Code Custom Commands

**What it is:** Markdown files in `.claude/commands/` that define reusable workflows you can invoke with `/command-name`.

**What problem it solves:** Starting a training run requires: (1) start Showdown server, (2) wait for it, (3) run the right Python command with the right args. That's 3 steps you do every time. A `/train` command wraps it into one step.

**Commands we're creating:**
- **`/train`** — starts Showdown + runs training with specified args
- **`/benchmark`** — starts Showdown + runs heuristic benchmark + summarizes results

**How it fits our workflow:** Instead of remembering `cd showdown && node pokemon-showdown start --no-security` every time, you type `/train` and Claude handles the rest.

---

## 8. Antigravity Awesome Skills

**What it is:** A community library of 1,340+ "agentic skills" — structured prompts that teach Claude Code how to approach specific task domains. Think of it as a curated prompt library that installs as plugins.

**What problem it solves:** When you ask Claude to debug something, create a PR, or write tests, it uses its general knowledge. Skills give it **battle-tested methodologies** for these tasks. For example:
- `@debugging-strategies` — systematic troubleshooting playbook instead of ad-hoc guessing
- `@test-driven-development` — TDD workflow that writes tests first, then implementation
- `@create-pr` — structured PR creation with proper description, test plan, etc.

**How it fits our workflow:** After installation, you can reference skills in your prompts: "Use @debugging-strategies to figure out why the reward shaping isn't decaying." Claude follows the structured methodology instead of winging it.

**Note:** This is optional. The skills are community-contributed and quality varies. The "universal starter" skills (debugging, TDD, PR creation) are the most reliable.

---

## 9. CodeRabbit (AI Code Review)

**What it is:** A GitHub app that automatically reviews every PR with AI, posting line-by-line comments with severity rankings and fix suggestions.

**What problem it solves:** As a solo developer, you have no code reviewer. CodeRabbit acts as an automated "second pair of eyes" that catches:
- Logic errors (like the temperature annealing bug it could have caught)
- Inconsistencies between code and documentation
- Missing edge cases
- Style issues and markdown formatting problems

**Already working:** CodeRabbit reviewed PR #1 (codebase-audit) and found 5 actionable issues — missing audit entry, markdown formatting, stale dimension references, and a Ruff lint warning. We already applied 3 of those fixes.

**Cost:** Free for public repos. $12-24/month per developer for private repos.

**How it fits our workflow:** You create a PR, CodeRabbit reviews it automatically within minutes, you fix what's relevant, merge with confidence. Zero configuration needed — it just works.

---

## 10. Weights & Biases (W&B) — Future Addition

**What it is:** An ML experiment tracking platform. Think "GitHub for training runs" — it logs metrics, hyperparameters, and model checkpoints to a web dashboard.

**What problem it solves:** Right now, training metrics go to TensorBoard logs and custom `training_log.md` files in each run directory. To compare runs, you need to start a local TensorBoard server and flip between tabs. W&B gives you:
- **Real-time dashboards** viewable from any browser (even your phone)
- **Run comparison** — overlay Phase A vs Phase B training curves, compare hyperparameters
- **Hyperparameter sweeps** — automatically try different learning rates, reward weights, etc.
- **Model versioning** — checkpoints uploaded and linked to their training metrics

**Why it's perfect for this project:** SB3 has a first-class W&B integration. It takes ~5 lines of code to add. And `sync_tensorboard=True` means your existing TensorBoard logging keeps working — W&B just mirrors it to the cloud.

**Why "future":** It requires modifying `src/train.py` and `src/selfplay_train.py`, which is project code. The setup prompt focuses on tooling/config only. When you're ready for W&B, it's a small change.

**Cost:** Free tier gives unlimited public projects and 100GB storage.

---

## How It All Connects

```text
You write code (or Claude does)
    ↓
Claude Code hook → auto-formats with Ruff
    ↓
You commit
    ↓
Pre-commit hooks → Ruff lint/format + large file check + secret detection
    ↓
You push / create PR
    ↓
GitHub Actions CI → Ruff lint + pytest (sequential: test runs after lint passes)
CodeRabbit → AI code review with line comments
    ↓
You merge
    ↓
Dependabot → keeps dependencies fresh automatically
```

Every step adds a safety net. No single tool does everything, but together they catch different classes of problems at different stages.
