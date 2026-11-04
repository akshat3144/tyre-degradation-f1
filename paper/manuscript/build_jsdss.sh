#!/usr/bin/env bash
# Build paper/manuscript/main_jsdss.pdf (JSDSS submission): refs -> pdflatex -> bibtex -> pdflatex x2.
# Needs a TeX distribution (pdflatex, bibtex); no journal style file is required.
set -eu
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ../../.venv/Scripts/python.exe ] && PY=../../.venv/Scripts/python.exe
[ -x ../../.venv/bin/python ] && PY=../../.venv/bin/python
"$PY" make_refs.py
pdflatex -interaction=nonstopmode -halt-on-error main_jsdss.tex
bibtex main_jsdss
pdflatex -interaction=nonstopmode -halt-on-error main_jsdss.tex
pdflatex -interaction=nonstopmode -halt-on-error main_jsdss.tex
echo "built main_jsdss.pdf"
