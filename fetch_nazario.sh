#!/usr/bin/env sh
# Fetch mailboxes from Jose Nazario's phishing corpus (CC-BY-4.0) and check they match the copies used here.
set -eu
BASE=https://monkey.org/~jose/phishing
cd "$(dirname "$0")"
mkdir -p phishing-corpus
for name in phishing2.mbox phishing3.mbox phishing-2024 phishing-2025 LICENSE.txt README.txt; do
  if [ ! -f "phishing-corpus/$name" ]; then
    curl -fsSL "$BASE/$name" -o "phishing-corpus/$name"
  fi
done
shasum -a 256 -c - <<SUMS
5113277984eae759a5ff958ccf408b542441ce033a49b923ab3b0f324075e6bd  phishing-corpus/phishing2.mbox
b29336d2e31c2dff19639415e98ed2aced5b4d63675ea5e29bea1a7d4e452841  phishing-corpus/phishing3.mbox
60afa40a2757170093a2bde31ec9cda4e54d33992203ce04bfbbc86d2e5e5b99  phishing-corpus/phishing-2024
f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f  phishing-corpus/phishing-2025
SUMS
echo "Phishing corpus mailboxes are in phishing-corpus/"
