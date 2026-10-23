"""Resolve numerical-occurrence source lines using a fresh LaTeX SyncTeX file."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-root', type=Path, required=True)
    p.add_argument('--occurrences', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    paper = a.paper_root.resolve()
    pdf = paper / 'full_paper.pdf'
    if not shutil.which('synctex'):
        p.error('Install LaTeX SyncTeX and compile with -synctex=1 first.')
    rows = list(csv.DictReader(a.occurrences.open(encoding='utf-8-sig', newline='')))
    pairs = sorted({(r['paper_source'].removeprefix('paper/'), int(r['source_line']))
                    for r in rows})
    pages, candidates = {}, {}
    for source, line in pairs:
        result = subprocess.run(['synctex', 'view', '-i', f'{line}:1:{paper/source}',
                                 '-o', str(pdf)], capture_output=True, text=True, check=True)
        found = sorted({int(v) for v in re.findall(r'^Page:(\d+)$', result.stdout, re.M)})
        key = f'paper/{source}:{line}'
        candidates[key] = found
        if len(found) == 1:
            pages[key] = found[0]
    output = a.out or paper / 'page_index.json'
    output.write_text(json.dumps({
        'method': 'SyncTeX source-line lookup; ambiguous multi-page results remain candidates',
        'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'sources': {s: hashlib.sha256((paper/s).read_bytes()).hexdigest() for s,_ in pairs},
        'pages': pages, 'candidates': candidates}, indent=2) + '\n', encoding='utf-8')
    print(f'Resolved {len(pages)}/{len(pairs)} unique source lines; wrote {output}')

if __name__ == '__main__':
    main()
