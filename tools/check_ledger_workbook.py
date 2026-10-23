"""Read-only check that the Excel accompaniment agrees with its occurrence CSV."""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter
from openpyxl import load_workbook

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=Path(__file__).resolve().parents[1] / 'numerical_ledger')
    args = parser.parse_args()
    rows = list(csv.DictReader((args.ledger / 'occurrences.csv').open(encoding='utf-8-sig', newline='')))
    book = load_workbook(args.ledger / 'Numerical_Provenance.xlsx', read_only=True, data_only=True)
    sheet = book['Numerical ledger']
    values = list(sheet.iter_rows(min_row=7, max_row=6+len(rows), values_only=True))
    for r, cells in zip(rows, values):
        assert cells[0] == r['occurrence_id'], r['occurrence_id']
        assert cells[1] == (int(r['pdf_page']) if r['pdf_page'] else None), r['occurrence_id']
        assert cells[3] == (r['readable_token'] or r['printed_token']), r['occurrence_id']
        assert cells[5] == r['status'], r['occurrence_id']
    counts = Counter(r['status'] for r in rows)
    for cell, expected in {'B4':len(rows), 'E4':counts['CHECKED'],
                           'H4':counts['PROVENANCE_MAPPED'], 'K4':counts['PENDING_MAPPING']}.items():
        assert sheet[cell].value == expected, (cell, sheet[cell].value, expected)
    for sh in book:
        errors = [(c.coordinate, c.value) for row in sh for c in row if c.data_type == 'e']
        assert not errors, (sh.title, errors[:5])
    result = {'status':'PASS', 'occurrence_rows':len(rows), 'worksheets':book.sheetnames,
              'matched_columns':['id','page','printed value','status'], 'cached_summary_counts':dict(counts),
              'scope':'CSV/Excel export consistency, not independent verification of paper calculations.'}
    print(json.dumps(result, indent=2))
    book.close()

if __name__ == '__main__':
    main()
