# Hermes Dashboard

The local Hermes dashboard now exposes richer chat and session inspection features.

## Chat Improvements

- Tool activity is rendered as structured entries instead of best-effort text parsing.
- Tool requests and tool outputs are both shown.
- Tool blocks are collapsed by default and expandable inline.
- The chat header shows recent token usage and the estimated prompt-context token count when Hermes provides it.

## Session File Browser

- Session detail now includes a file browser built from persisted file-tool history in `state.db`.
- The browser currently reconstructs `read_file`, `write_file`, and `patch` activity.
- Clicking a file loads a text preview when the path resolves inside allowed dashboard roots.

## Safe File Preview Rules

The dashboard only previews files when the resolved path stays within an allowed root.

Allowed roots are:

- `HERMES_WRITE_SAFE_ROOT` when set
- the dashboard process current working directory
- the active `HERMES_HOME`

Binary files and paths outside those roots are shown as metadata only.

## Implementation Notes

- `gateway/platforms/api_server.py` now emits extra Hermes metadata chunks during streaming chat completions for:
  - tool calls
  - tool outputs
  - prompt-context metadata
- `~/.hermes/dashboard/app.py` forwards those metadata chunks to the browser and adds session file/content endpoints.
- `~/.hermes/dashboard/templates/index.html` renders the structured tool UI, context pills, and file browser.
