# Generalisable Practices

Transferable lessons from the `pokemon_rl` project that apply to any future software project on this laptop. Update this file as new patterns prove themselves. The point is to start the *next* project with everything already dialled in.

**How to use this document**: skim top to bottom when starting a new project. Copy the parts that apply. Delete the parts that don't. This is a living checklist, not a manifesto.

---

## 1. Project bootstrap — files to create on day 1

Every non-trivial project should have these at the root from the start. Ordered by priority.

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project-specific instructions that Claude Code loads automatically. Include: tech stack, how to run the dev server/tests, git workflow rules, any "never do X" rules, key architecture notes. Keep under ~150 lines. |
| `.coderabbit.yaml` | Tell CodeRabbit what *not* to review — docs, submodules, generated artifacts, vendor code. Preserves free-tier quota. Template in §5 below. |
| `.pre-commit-config.yaml` | Hooks that run before every commit: linter, formatter, trailing whitespace, YAML validation, large-file check, private-key detection. |
| `.github/workflows/ci.yml` | Mirror the pre-commit hooks in CI as a safety net, plus run tests. |
| `.github/dependabot.yml` | Weekly bumps for GitHub Actions, language package manager (npm/pip/cargo), and any submodule-linked ecosystem. |
| `README.md` | One-paragraph "what is this", plus a fresh-clone setup section (clone → init submodules → install deps → first command). |
| `AGENTS.md` *(optional)* | If using a multi-persona team pattern, define the personas and ownership here. |

A skeleton repo with all of these pre-filled would save ~30 minutes per project and guarantees no bootstrapping mistake. Worth creating once as a template repo.

---

## 2. CLAUDE.md — structure that works

Patterns that proved useful in this project:

- **Hard rules first, at the top**. E.g., "Never include Co-Authored-By: Claude in commits." Rules buried deep get skipped.
- **Explicit git workflow** as a numbered list (branch → commit → push → PR → wait for review → address feedback → merge → sync). Claude follows numbered steps reliably.
- **Tech stack section** listing the Python/Node versions, the conda env path, and the key libraries. Eliminates "what version does this project use?" back-and-forth.
- **"Complex decisions" clause**: when facing non-trivial architecture choices, break the problem into steps, consider at least 2 alternatives, compare trade-offs before recommending. Without this, Claude tends to jump to the first reasonable answer.
- **Ship-every-task clause**: explicitly require that every task ending with tracked-file changes ships as a merged PR. Without it, work accumulates in stale branches.
- **"Note on content-only PRs"**: fold documentation updates into the PR they describe, rather than opening follow-ups. Cleaner history, less AI-review quota burned.

Anti-patterns to avoid in CLAUDE.md:
- Don't repeat what's in code (file structure, function signatures). Code drifts; CLAUDE.md becomes stale.
- Don't write "when to use X vs Y" taste essays. Keep it actionable.
- Don't bloat with narrative — if it's not a rule, a pointer, or a command, cut it.

---

## 3. Git & PR workflow

Defaults that should apply to every project unless there's a specific reason otherwise.

### Commit messages
- Imperative mood, concise subject under ~70 chars.
- Use heredoc for multi-line: `git commit -m "$(cat <<'EOF' ... EOF)"`.
- Never `--no-verify`. If a hook fails, fix the underlying issue.
- Never `Co-Authored-By: Claude` or any AI attribution. Commits appear solely authored by the human.
- Prefer new commits over `--amend` for anything already pushed.

### Branches and PRs
- Always work on a feature branch, never directly on `master`/`main`.
- Branch name should describe the change (`reduce-coderabbit-quota-usage`, not `fix-1`).
- PR title under 70 chars; PR body has `## Summary` (bullets) and `## Test plan` (checklist).
- No AI attribution in PR bodies either.
- Standard merge strategy: **`gh pr merge <N> --squash --delete-branch`**.

### Why squash-merge as default

Collapses all commits on the PR branch into a single commit on master. Full per-commit history stays visible in the closed PR on GitHub, so nothing is "lost". Benefits:
- Cleaner `git log` / `git bisect` on master — each commit is a self-contained reviewed unit.
- Granular local commits become free — commit as often as you want while working, it all collapses at merge.
- Removes the tension between "commit often" and "every push triggers an AI review".

Exception: long-running branches with genuinely independent commits each deserving to stand alone (rare). Dependabot / single-commit PRs: squash is fine, no cost.

### Addressing AI-review feedback
- Batch all comments from one review round into a **single commit** titled `Address CodeRabbit feedback on PR #<N>` before pushing. Not one commit per comment. Matches OSS convention and (under squash-merge) collapses to nothing anyway.
- Ignore pure nitpicks on docs/notes files — they're not worth a round trip.
- If a suggestion is wrong, reply explaining why rather than silently ignoring.

---

## 4. Code quality — pre-commit + CI

Two layers. Both cheap to set up, both catch different things.

### Pre-commit hooks (`.pre-commit-config.yaml`)

Minimum viable set, regardless of language:
- Trailing whitespace
- End-of-file fixer
- YAML validator
- Large-file check
- Private-key detection

Then language-specific:
- Python: `ruff` (lint) + `ruff-format` (format). Replaces black + flake8 + isort with one fast tool.
- JS/TS: `eslint` + `prettier`, or `biome` as a single-tool alternative.
- Go: `gofmt` + `go vet`.
- Rust: `cargo fmt` + `cargo clippy`.

Install once per clone with `pre-commit install`. The hooks then run on every `git commit`.

### CI (`.github/workflows/ci.yml`)

Mirror the pre-commit hooks (safety net for anyone who skips them) plus run the test suite. Keep it fast — under 2 minutes for a standard repo — otherwise people stop waiting for it. Cache dependencies aggressively.

Don't bloat CI with "also check this and that". Each added step becomes a flakiness surface. Add checks only when you've been bitten by the class of bug they catch.

---

## 5. AI code review — CodeRabbit setup

CodeRabbit's free tier allows ~2 reviews/hour. Every push to an open PR triggers a new review, so quota burns fast without config.

### Template `.coderabbit.yaml`

```yaml
# CodeRabbit configuration
# Docs: https://docs.coderabbit.ai/reference/yaml-template
language: en-US
reviews:
  profile: chill              # reduces nitpick volume
  request_changes_workflow: false
  high_level_summary: true
  poem: false                 # skip the poem, saves tokens and noise
  review_status: true
  auto_review:
    enabled: true
    drafts: false             # don't review draft PRs
    base_branches:
      - master                # or main, depending on repo
  path_filters:
    # Exclude everything AI review can't meaningfully critique:
    - "!docs/**"              # prose documentation
    - "!notes/**"             # planning / scratch notes
    - "!content/**"           # if the project has a content dir for narrative
    - "!runs/**"              # training/experiment artifacts
    - "!data/**"              # datasets, fixtures
    - "!vendor/**"            # vendored dependencies
    - "!*.md"                 # root-level planning markdown (adjust if README/CLAUDE should be reviewed)
    # If the project has a submodule (separate repo), exclude it:
    # - "!<submodule_dir>/**"
chat:
  auto_reply: true
```

Adjust the path filters per project. Leave `CLAUDE.md` and `README.md` reviewable — they affect workflow and deserve AI eyes.

### Other quota-saving habits

- Self-review your diff before the first push — catches obvious nits preemptively.
- Batch multiple local commits into a single push, rather than push-per-commit.
- Bundle documentation-only follow-ups into their parent code PR (see §3).
- Open as Draft while iterating; mark Ready for Review when done (`drafts: false` keeps CodeRabbit off drafts).

### When to consider paying

If you're regularly hitting the cap despite the above, Lite is ~$15/dev/mo. For solo hobby projects, config usually suffices. For a funded team where devs lose 10 min waiting on review quota, it's trivial to justify.

---

## 6. Claude Code configuration

### Persistent memory

Claude maintains per-project memory at `~/.claude/projects/<encoded-project-path>/memory/`:
- `MEMORY.md` is the index (always loaded, truncated after ~200 lines)
- Individual memory files carry frontmatter with `name`, `description`, `type` (`user`, `feedback`, `project`, `reference`).

Use memory for:
- Stable user preferences ("no AI attribution in commits", "don't use VecNormalize in this RL project").
- Corrections you've given that you don't want to repeat.
- References to external systems (dashboards, issue trackers).

Don't use memory for:
- Anything derivable by reading the code or `git log`.
- Ephemeral task state — use plan mode or task list instead.
- Duplicates of CLAUDE.md content.

### MCP servers to consider

- **`context7`** — fetches current library docs. Prevents Claude from giving outdated API advice based on training data. Install once; Claude uses it automatically for library questions.
- **Gmail / Calendar / Drive** — only if you actually want Claude doing those things in-IDE. Added surface area if not.

### Claude hooks

Configurable in `settings.json` for automated behaviors — e.g., run a command when a session stops, alert on specific patterns. Use when you have a *genuinely repetitive automated* need; skip for one-offs.

### Auto mode vs plan mode

- **Auto mode** — Claude executes autonomously, minimises interruptions. Good for well-specified tasks where you trust the agent.
- **Plan mode** — Claude investigates and writes a plan file before touching anything. Good for large changes where a mistake is expensive. Exit via `ExitPlanMode`; the plan file serves as a durable spec.
- Switch based on task risk, not habit. Small bugfix → auto. New subsystem → plan first.

---

## 7. Decision-making patterns that worked

Across many sessions in this project, these patterns consistently produced good outcomes:

- **Two alternatives minimum.** Before recommending, explicitly consider at least one alternative and explain why it's worse. Forces actual reasoning instead of confirmation bias.
- **Diagnose before fixing.** When a behaviour seems wrong, understand *why* it's happening before changing code. The surface fix often masks the real issue.
- **One commit per review round.** Amplifies feedback signal, reduces re-reviews, cleaner history.
- **Rollback cost is a real criterion.** Prefer changes whose rollback cost is "delete a file and revert an edit" over changes that require data migration or state reconciliation.
- **Decide, document, move on.** Architectural decisions get recorded in a decision log (a `decisions.md` or equivalent) so they're not re-litigated. Future-you will forget why a specific choice was made; the log is your cheapest defence.

---

## 8. Anti-patterns observed — avoid these

- **Shipping multiple tiny commits per PR**. If they'll squash anyway, just don't push between them. If they won't squash, they bloat the master log.
- **Content-only follow-up PRs**. "Update decision log to reflect the thing I just did in PR #N" should be *part of* PR #N, not PR #N+1.
- **Letting AI review run on prose/docs**. It doesn't help and burns quota. Path-filter from day 1.
- **Writing CLAUDE.md as a design doc**. It's instructions. Anything that reads like "we chose X because Y" belongs in a decision log, not CLAUDE.md.
- **Over-broad pre-commit hooks**. Every added hook is friction. Start minimal, add only when you've been bitten.
- **Skipping hooks (`--no-verify`)**. If a hook is wrong, fix the hook. If the hook is right, fix the code.
- **Comments that explain *what* instead of *why***. Delete them.
- **Letting stale branches accumulate**. They rot. Merge or delete.
- **Starting with a complex architecture "because we'll need it"**. Three similar lines beats a premature abstraction. Add structure when a third use case appears, not before.

---

## 9. Editing this document

This file is meant to evolve. When a pattern proves itself in a new project, promote it here. When a pattern turns out to be wrong or outdated, edit it out. Each section stands on its own — cut what doesn't apply.

Last updated: 2026-04-19.
