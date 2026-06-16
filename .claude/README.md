# .claude Directory (Project Scope)

This folder stores project-scoped Claude Code configuration and reusable prompts for the INFLUX Impact pipeline.

## Layout

- `settings.json`: project Claude settings (team-shared)
- `rules/*.md`: persistent project instruction rules
- `commands/*.md`: reusable slash-command prompts
- `skills/<name>/SKILL.md`: reusable workflow skill
- `output-styles/*.md`: response style templates
- `agents/*.md`: subagent prompt definitions
- `agent-memory/`: persistent subagent memory files

## Notes

- The project roadmap and instruction source is in `Claude.md` at repo root.
- Keep `.claude` files committed for team-shared behavior.
- Keep personal overrides in local-only files (for example `CLAUDE.local.md` or `.claude/settings.local.json`).

