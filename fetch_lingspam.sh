#!/usr/bin/env sh
# Fetch the Ling-Spam corpus (Androutsopoulos et al., ECML 2000) and check it matches the copy used here.
# Its readme asks anyone publishing work that uses it to credit the corpus and notify the author.
set -eu
URL=https://www.aueb.gr/users/ion/data/lingspam_public.tar.gz
SHA256=e3a0bb61dff2f1e47c368d1c376c27f79319cd600a8784ea3872e4b77d22c148
cd "$(dirname "$0")"
mkdir -p lingspam
if [ ! -f lingspam/lingspam_public.tar.gz ]; then
  curl -fsSL "$URL" -o lingspam/lingspam_public.tar.gz
fi
echo "$SHA256  lingspam/lingspam_public.tar.gz" | shasum -a 256 -c -
tar -xzf lingspam/lingspam_public.tar.gz -C lingspam
echo "Ling-Spam unpacked in lingspam/lingspam_public"
