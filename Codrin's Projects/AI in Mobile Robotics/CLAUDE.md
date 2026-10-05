# AI in Mobile Robotics

Coursework and robot programs that drive LEGO(R) Education Computer Science & AI kit
hardware from Python, via the [`legoeducation`](https://pypi.org/project/legoeducation/)
package.

## Rule 0: the LEGO API docs are mandatory

<https://github.com/LEGO/LEGOEducation> is the **only** source of truth for this API.
Before writing, editing, or reviewing any `legoeducation` call, read the vendored copy in
`reference/LEGOEducation/` — refresh it with `python scripts/sync_lego_docs.py` if it is
missing.

Never invent a method, property, keyword argument, or constant: if it is not in
`reference/LEGOEducation/function_description.md` or `constants.md`, it does not exist.
The API deliberately breaks Python convention and does not resemble Pybricks, ev3dev, or
SPIKE Prime, so recall is not a substitute for grepping the docs. Say which doc file you
verified against when you hand back code.

Full guidance, including the API map and pre-flight checklist, is in the
**`lego-education-python`** skill (`.claude/skills/lego-education-python/SKILL.md`).
Invoke it for any LEGO hardware work.

## Layout

| Path | What it holds |
|---|---|
| `src/mobile_robotics/` | Reusable helpers — config-driven connect, `session()` teardown |
| `examples/` | Runnable programs, one behavior each |
| `scripts/sync_lego_docs.py` | Vendors the upstream LEGO docs into `reference/` |
| `reference/LEGOEducation/` | Upstream docs (gitignored, read-only, never edit) |
| `hardware.example.json` | Template for `hardware.json` (gitignored, per-kit) |

## Conventions

- Import the library as `le`: `import legoeducation as le`.
- Connect through `mobile_robotics.session(...)`, not raw `connect()` calls — it filters by
  Connection Card from `hardware.json` and guarantees `disconnect()` on every exit path.
- Always check `.connected` and give a real failure path; a failed connect is routine.
- Use symbolic `le.*` constants, never bare integers.
- Prefer bounded motion (`*_for_time`, `*_for_degrees`) and always stop in a `finally`.
- Never commit `hardware.json` — Connection Card colors and serials are per-kit.

## Environment

This project targets **Python 3.14** (upstream's stated minimum; the wheel metadata itself
says `>=3.11`). Use the repo venv: `.venv\Scripts\python.exe`, created with `py -3.14`.
`legoeducation` needs Bluetooth Low Energy access from Python.
Hardware must be charged, powered on, and broadcasting before connecting. Chromebooks
cannot connect at all (their Linux container has no Bluetooth access).
