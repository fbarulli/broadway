# MAC_APPENDIX.md — appendix beside MAIN_AGENT_CONTRACT.md

Class: **SSOT for relocated MAIN_AGENT_CONTRACT content** (appendix-class
material under the §14 contract-size cap law: worked examples, recipes, and
doctrine bullets too long for the main file). The main contract points here;
this file owns the relocated text VERBATIM.

Provenance: relocated from MAIN_AGENT_CONTRACT §5 per D32(6) (eight-packet
ruling batch, human, this session, 2026-08-25); the board-access recipe
relocated from MAIN_AGENT_CONTRACT §14 under the same batch's ~30 KB cap
remedy (§14: "worked examples and recipes move to appendix files beside the
contract").

## Stamp semantics (dispatch stamps are RELATIVE)

- **Stamp semantics:** a dispatch stamp is RELATIVE — the HEAD at dispatch
  open, recorded in the worker's report. Absolute SHAs appear only inside
  immutable records as provenance anchors, never as executable preconditions
  in standing contracts.

## Accessing the board via gh

Repo `fbarulli/broadway`, board = issue #5, conversation-locked.
Read-only inspection is allowed and expected at any tier; WRITES are
owner-only and follow the store-then-hash recipe below — workers,
seniors, and adversaries never post.

```bash
# Row index — one line per comment (id · created_at · subject):
gh api repos/fbarulli/broadway/issues/5/comments --jq \
  '.[] | [.id, .created_at, (.body | split("\n")[0])] | @tsv'

# Active rows only:
gh api repos/fbarulli/broadway/issues/5/comments --jq \
  '.[] | select(.body | test("^status: active", "m")) | .id'

# Full text of ONE row — take <comment-id> from STATE.md ## EVENTS:
gh api repos/fbarulli/broadway/issues/comments/<comment-id> --jq '.body'

# Issue metadata / lock check:
gh api repos/fbarulli/broadway/issues/5 --jq '{title, locked, open}'

# Verify an event-id recomputes (D4/D8: paste command WITH output):
gh api repos/fbarulli/broadway/issues/comments/<id> --jq '.body' > /tmp/b
python3 - <<'PY'
import hashlib
lines=[l for l in open('/tmp/b').read().splitlines()
       if not l.startswith(('event-id:','recorded-time:'))]
print(hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:8])
PY
```

Cite rows as `issues/5#issuecomment-<id> event-id <sha8>`; resolution
rows live in STATE.md `## EVENTS`. If github.com is unreachable, no
gate blocks — the tree registry (GIT-WINS) remains authoritative.
