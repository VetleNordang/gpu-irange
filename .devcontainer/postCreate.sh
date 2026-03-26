#!/usr/bin/env bash
set -e

BASHRC="$HOME/.bashrc"
LINE='[[ "$TERM_PROGRAM" == "vscode" ]] && . "$(code --locate-shell-integration-path bash)"'

touch "$BASHRC"
if ! grep -Fq 'code --locate-shell-integration-path bash' "$BASHRC"; then
  echo '' >> "$BASHRC"
  echo '# VS Code shell integration' >> "$BASHRC"
  echo "$LINE" >> "$BASHRC"
fi

echo "postCreate complete"