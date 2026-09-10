"""Copy run logs into the supplementary bundle with home directories removed.

Python tracebacks in these logs embed the interpreter's absolute path, which on
this machine contains the author's username:

    File "C:\\Users\\<name>\\AppData\\Local\\...\\openai\\_utils\\_utils.py"

That is a deanonymisation vector in a double-blind submission, and it is easy to
miss because it appears only in stack traces, not in ordinary output. Six of the
370 logs carried it.

Kept as a separate file rather than inlined in build_supplementary.sh: the
pattern needs literal backslashes, and passing those through a shell heredoc
into sed was silently collapsing them, so the redaction ran and changed nothing.

    python research/redact_logs.py <src_root> <dst_root>
"""
from __future__ import annotations

import glob
import os
import re
import sys

# One literal backslash is written '\\' inside this raw string; in the regex it
# is an escaped backslash, matching a single backslash in the log.
HOME = re.compile(
    r'[A-Za-z]:\\Users\\[^\\/"\' ]+'      # C:\Users\<name>
    r'|[A-Za-z]:/Users/[^/"\' ]+'          # C:/Users/<name>
    r'|/Users/[^/"\' ]+'                   # /Users/<name>
    r'|/home/[^/"\' ]+'                    # /home/<name>
)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    src, dst = argv[1], argv[2]
    out_dir = os.path.join(dst, 'research', 'results', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    n = redacted = 0
    for f in glob.glob(os.path.join(src, 'research', 'results', 'logs', '*.log')):
        text = open(f, encoding='utf-8', errors='replace').read()
        clean, hits = HOME.subn('<HOME>', text)
        open(os.path.join(out_dir, os.path.basename(f)), 'w',
             encoding='utf-8').write(clean)
        n += 1
        redacted += 1 if hits else 0
    print(f'   {n} logs copied, {redacted} contained a home path and were redacted')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
