"""Rerun Section 7 using checked baseline artifacts; full Run All is also supported."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rebuilt', type=Path, help='Optional newly built notebook whose baseline outputs will be preserved.')
args = parser.parse_args()
here = Path(__file__).resolve().parent
path = here / 'literature_triage_topics.ipynb'
old = nbformat.read(path, as_version=4)
new = nbformat.read(args.rebuilt or path, as_version=4)
start = next(i for i, c in enumerate(new.cells) if c.cell_type == 'markdown' and c.source.startswith('## 7.'))
if args.rebuilt:
    assert len(old.cells) >= start
    for i in range(start):
        a, b = old.cells[i], new.cells[i]
        if a.source != b.source:
            # This one permitted helper change only validates future overrides.
            assert 'def apply_overrides(' in a.source and 'def apply_overrides(' in b.source
            assert not (here / 'category_overrides.csv').exists(), 'Run all cells when overrides exist.'
        if b.cell_type == 'code':
            b.outputs, b.execution_count = a.outputs, a.execution_count

bootstrap = nbformat.v4.new_code_cell(r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, re
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
OUTPUT_DIR = Path.cwd() / 'outputs'
manifest = json.loads((OUTPUT_DIR / 'run_manifest.json').read_text(encoding='utf-8'))
for source, expected in manifest['inputs'].items():
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
        raise ValueError('An input workbook changed; run the entire notebook first.')
override = Path.cwd() / 'category_overrides.csv'
override_hash = hashlib.sha256(override.read_bytes()).hexdigest() if override.exists() else None
if override_hash != manifest['override_sha256']:
    raise ValueError('Manual overrides changed; run the entire notebook first.')
full_paper_decisions = pd.read_csv(OUTPUT_DIR / 'full_decision_audit.csv', keep_default_na=False)
assert full_paper_decisions.paper_id.is_unique
assert len(full_paper_decisions) == manifest['before_unique_papers']
assert full_paper_decisions.triage_decision.eq('Y').sum() == manifest['explicit_Y_unique']
assert full_paper_decisions.triage_decision.eq('N').sum() == manifest['explicit_N_unique']
assert set(full_paper_decisions.classifier_version.astype(str)) == {manifest['classifier_version']}
assert full_paper_decisions.loc[full_paper_decisions.is_review_article, 'category'].eq('Theory & modeling').all()
print('Verified cached baseline, unchanged workbook hashes, and manual overrides.')
''')
helper_index = next(i for i, c in enumerate(new.cells) if 'def apply_overrides(' in c.source)
classifier_index = next(i for i, c in enumerate(new.cells) if 'def classify_topic(' in c.source)
assert old.cells[classifier_index].source == new.cells[classifier_index].source
indices = [helper_index, classifier_index] + [i for i in range(start, len(new.cells)) if new.cells[i].cell_type == 'code']
mini = nbformat.v4.new_notebook(cells=[bootstrap] + [nbformat.v4.new_code_cell(new.cells[i].source) for i in indices])

def completed(cell, cell_index, **kwargs):
    print('Executed comparison cell {} of {}'.format(cell_index + 1, len(mini.cells)), flush=True)

NotebookClient(mini, timeout=600, kernel_name='python3',
               resources={'metadata': {'path': str(here)}}, on_cell_executed=completed).execute()
for i, executed in zip(indices, mini.cells[1:]):
    new.cells[i].outputs = executed.outputs
    new.cells[i].execution_count = executed.execution_count
    new.cells[i].metadata = executed.metadata
new.metadata['comparison_execution'] = {
    'mode': 'Section 7 executed using baseline artifacts after checking original workbook and override hashes',
    'baseline_outputs': 'Preserved from the completed full notebook run',
}
nbformat.validate(new)
nbformat.write(new, path)
print('Saved executed comparison notebook:', path, flush=True)
