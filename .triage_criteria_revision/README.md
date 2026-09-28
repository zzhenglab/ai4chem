# Literature triage with four topic-definition comparisons

**New: Section 7 compares four fixed topic definitions**, each with its own green-Y /
orange-N plot and an identical full corpus. The definitions are current synthesis-inclusive v5,
framework synthesis scope, property priority with stricter synthesis/structure
evidence, and primary computation priority (including mixed studies).
All identified reviews remain Theory & modeling. Topic rules never receive Y/N.

The notebook checks Chemical synthesis **>=4:1**, Crystal engineering **>6:1**
and separately **>7:1**, and Functional materials **Y>N** after assigning topics.
These ratios do not establish topic accuracy or choose the preferred definition.
All outcomes and unresolved papers are shown. The alternatives were defined before
calculating their results, with knowledge of the requested targets; this is an
exploratory sensitivity comparison, not preregistered validation.

New exports are in `outputs/criteria_comparison/`:

- `all_criteria_Y_N.png` / `.pdf`: all four definitions on the same count scale.
- `<criterion>_Y_N.png` / `.pdf`: a separate chart for each definition.
- `counts_by_criterion.csv`: before totals, retained Y, rejected N, and Y:N ratios.
- `target_checks.csv`: requested thresholds, pass/fail, and coverage.
- `topic_criteria_comparison.xlsx`: counts, definitions, per-paper evidence, and changes.
- `paper_assignments.csv` / `category_changes.csv`: inspect every label and transition.
- `comparison_manifest.json`: inputs, criteria, decision totals, and code hash.

`topic_criteria.py` contains the fixed alternatives; `criteria_analysis.py` checks
the common corpus, calculates ratios, and draws the plots. The notebook embeds
both, along with `run_criteria_comparison.py`, so it still runs on its own.
Run `python -m unittest discover -s . -p "test_*.py"` for the counting, review-policy,
topic-boundary, and ratio-threshold checks. The previous notebook and builder are
preserved in `archive/v3_before_criteria/`.
`python verify_criteria_results.py` checks the saved notebook, original input hashes,
per-paper counts, ratios, reviews, and Excel exports. To rerun only Section 7 after
a completed full run, use `python execute_comparison.py`; it requires unchanged
input workbooks and overrides. A conflicting manual override of an identified
review is rejected; correct the genre metadata/detection first if it is mistaken.

# Literature topics and triage decisions

Open **literature_triage_topics.ipynb** to inspect the executed analysis. The final
table shows **Before triage**, **After triage (Y)**, and **After triage (N)** together.
The horizontal chart compares Y (green) with N (orange).

| Category | Before triage | After triage (Y) | After triage (N) |
| --- | ---: | ---: | ---: |
| Chemical synthesis | 5,360 | 3,168 | 2,192 |
| Theory & modeling | 477 | 72 | 405 |
| Crystal engineering | 3,557 | 2,057 | 1,500 |
| Functional materials | 4,376 | 2,138 | 2,238 |
| Unclassified | 0 | 0 | 0 |
| TOTAL | 13,770 | 7,435 | 6,335 |

All 124 previously unclassified papers have a provisional topic: 47 now match
specific rules and 77 use broad lexical fallback with review flags. None required
the unsupported domain default. The displayed categories now use the broader
synthesis-inclusive definition below; the former main-topic categories and all saved
Y/N decisions are preserved separately for comparison.

Chemical synthesis now contains **3,168 Y and 2,192 N** (59.10% Y). The added
framework-preparation rule includes 3,343 papers: 23 formerly grouped under Theory &
modeling, 1,305 under Crystal engineering, and 2,015 under Functional materials.
Every reassignment retains its explicit preparation evidence. The prior primary-topic
Chemical synthesis counts remain 879 Y and 1,138 N in the separate audit, so the effect
of the definition change can be inspected directly.

## Splitting and counting

The complete corpus contains **13,773 spreadsheet rows and 13,770 unique DOI papers**.
The existing full-corpus `Agent_YN` decisions divide the papers into **7,435 Y** and
**6,335 N**, with no unknown, conflicting, or unmatched decisions. The row totals are
7,437 Y and 6,336 N. The selected workbook's Y DOI set matches the full decision
workbook exactly. The three extra rows are duplicate DOIs, not additional papers.

DOIs are normalized by removing DOI URL/prefix wrappers and normalizing case and
whitespace. WOS ID and unambiguous normalized title are matching fallbacks when a DOI
is missing. Conflicting identifiers are reported. For each duplicate DOI, the longest
title/abstract record supplies the topic evidence, and that topic is reused everywhere.
Duplicate full-corpus Y/N disagreements stay `Conflict`; missing or unsupported labels
stay `Unknown`. Absence from the selected workbook is never treated as N.

Both DOI and row tables verify **Before = Y + N + Unknown + Conflict**. The row table
maps canonical decisions onto the original full-corpus rows. The saved triage decisions
are preserved; this notebook does not rerun screening or alter Y/N labels.

## Rough topic classification

**The topic classifications are not generated by an LLM. They are rough, deterministic
rule-based estimates from titles, abstracts, and document types, not validated expert
annotations.** They are separate from the saved model-generated Y/N triage decisions.
Topic rules receive no Y/N label. The synthesis-inclusive definition is specified
from article content before tallying Y/N counts.

| Display category | Definition |
| --- | --- |
| Chemical synthesis | Synthetic methods, molecular/ligand synthesis, or post-synthetic chemistry as before, plus original papers explicitly reporting experimental MOF/coordination-framework preparation, including structural and application papers |
| Theory & modeling | Computational/theoretical work, plus all identified review articles under the project's taxonomy convention |
| Crystal engineering | Crystal/network structure, topology, packing, and assembly without an explicit framework-preparation statement qualifying for Chemical synthesis |
| Functional materials | Applications and performance without an explicit framework-preparation statement qualifying for Chemical synthesis, including MOF-catalyzed molecular reactions |

This is a **change in category scope**: a paper can focus on crystal structures or
material performance and still report experimental synthesis. Such a paper now appears
under Chemical synthesis when its title/abstract explicitly reports framework
preparation. Its original v4 category remains in `primary_topic_category`. Original
scores and primary-topic reasons remain intact, and each new assignment records its
framework-synthesis evidence and separate classification reason. This does not change
triage eligibility or establish improved screening accuracy.

Using an existing MOF to catalyze product synthesis, preparing a MOF-derived material
alone, and background, hypothetical, or computational preparation statements do not
qualify for the additional framework-preparation rule. Reported preparation followed
by later conversion does qualify. Review articles retain their priority rule.

Broader lexical fallback rules cover papers missed by the specific contribution rules.
Fallback assignments retain their matching evidence and low-confidence review flags.
A domain default, if ever needed for text without matching evidence, is marked
unsupported; absent title and abstract remain Unclassified. Scores are heuristic
evidence strengths, not probabilities. Review articles are assigned to Theory & modeling
regardless of their subject, based on document type or explicit genre descriptions.

## Outputs

- `outputs/doi_classification.csv`: compact **DOI, Classification** list for the complete corpus.
- `outputs/doi_classification_triage.csv`: the same list with saved triage decision and review flag.
- `outputs/counts_by_category.csv`: unique-paper Before/Y/N counts, unresolved decisions, and review counts.
- `outputs/row_counts_by_category.csv`: corresponding full-corpus spreadsheet-row counts.
- `outputs/Y_N_counts_by_topic.csv`: disjoint Y/N counts and Y percentages.
- `outputs/resolved_unclassified.csv`: previously unclassified papers, new topics, and evidence.
- `outputs/triage_Y_N_by_topic.png` and `.pdf`: the Y/N chart.
- `outputs/full_decision_audit.csv`: complete topic evidence and existing decisions.
- `outputs/primary_topic_counts.csv`: preserved v4 main-topic counts.
- `outputs/taxonomy_reassignments.csv`: every synthesis-inclusive reassignment and its preparation evidence.
- `outputs/taxonomy_transition_counts.csv`: reconciliation of original and displayed categories.
- `outputs/paper_topic_audit.xlsx`: count tables, compact lists, and detailed audits.
- `outputs/review_queue.csv`: weak or overlapping assignments needing inspection.
- `outputs/run_manifest.json`: input hashes, classifier version, settings, and partition checks.

Additional outputs preserve source rows, duplicates, label comparisons, review-article
detections, synthesis evidence, and transitions from the original method. The previous
executed notebooks and counts are preserved in `archive/v3/` and `archive/v4/`; older
versions remain in `archive/v1/` and `archive/v2/`.

## Reproduction and corrections

Install dependencies with `python -m pip install -r requirements.txt`. The notebook
looks for its three source workbooks beside itself or in the project's `14489 paper`
folder; the settings cell allows another `DATA_DIR`. It reads source workbooks without
modifying them and requires no API key or network requests. Its analysis code is
embedded, so the companion Python modules are not required to run the notebook.

For a reviewed correction, create `category_overrides.csv` beside the notebook with
`paper_id,manual_category,review_note` columns. Use one of the four category names.
Rerunning Sections 3–6 updates every occurrence of that paper consistently and records
the manual override. The published rough analysis uses no manual overrides.

Maintain the Python sources, then run:

```bash
python -m unittest discover -s . -p "test_*.py"
python build_notebook.py
python execute_notebook.py
```

The article and SI inventories with their classifications are published in
[MOFinder's literature retrieval metadata](../MOFinder/data/metadata/literature_retrieval/README.md).
The separate DOI list and detailed audit remain local analysis outputs.
Those public inventories omit historical download flags; download progress belongs to
local working inventories. The public README documents the rough non-LLM topic method.
