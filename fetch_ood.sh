#!/usr/bin/env sh
# Fetch the modern out-of-distribution mail used by ood_test.py:
# - python-list and python-announce-list posts, January to August 2026 (public Mailman archives)
# - the August 2026 spam-trap archive from untroubled.org, which grants permission to use it without
#   restriction and asks for a reference in published work. Please fetch it once, not repeatedly.
set -eu
cd "$(dirname "$0")"
mkdir -p ood-corpus/lists ood-corpus/spamtrap
for list in python-list python-announce-list; do
  for m in 01 02 03 04 05 06 07 08; do
    next=$(printf "%02d" $((10#$m + 1)))
    file="ood-corpus/lists/$list-2026-$m.mbox"
    if [ ! -f "$file" ]; then
      curl -fsSL "https://mail.python.org/archives/list/$list@python.org/export/$list@python.org-2026-$m.mbox.gz?start=2026-$m-01&end=2026-$next-01" \
        | gunzip > "$file"
    fi
  done
done
if [ ! -f ood-corpus/spamtrap/2026-08.7z ]; then
  curl -fsSL http://untroubled.org/spam/2026-08.7z -o ood-corpus/spamtrap/2026-08.7z
fi
echo "d4f156ec5a169ceb9771e6dc0703924d69f059a13bb13e997225055e0831106c  ood-corpus/spamtrap/2026-08.7z" | shasum -a 256 -c -
uv run --with py7zr python -c "import py7zr; py7zr.SevenZipFile('ood-corpus/spamtrap/2026-08.7z').extractall('ood-corpus/spamtrap')"
echo "Modern mail is in ood-corpus/"
