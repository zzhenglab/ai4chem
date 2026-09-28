"""Notebook section: evaluate all fixed criteria, then join recorded decisions."""
COMPARISON_DIR = OUTPUT_DIR / 'criteria_comparison'
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
definitions = pd.DataFrame([
    {'criterion': key, 'Label': CRITERION_LABELS[key],
     'Definition': CRITERION_DESCRIPTIONS[key]} for key in CRITERIA
])
display(definitions.style.set_properties(**{'white-space': 'normal'}))

# No decision field is passed to the classifier. Freeze all assignments before
# joining Y/N labels or calculating ratios. Manual corrections apply equally.
assignment_rows = []
for record in full_paper_decisions.to_dict('records'):
    variants = assign_topic_criteria(
        record['Article Title'], record['Abstract'], record.get('Document Type', ''),
        baseline_category=record['category'])
    for variant in variants:
        result = dict(variant)
        result.update({
            'paper_id': record['paper_id'], 'Article Title': record['Article Title'],
            'DOI': record.get('DOI', ''), 'baseline_category': record['category'],
            'manual_reviewed': bool(record.get('manual_reviewed', False)),
            'is_review_article': bool(record.get('is_review_article', False)),
        })
        if result['category'] == record['category']:
            result['review_needed'] = bool(result['review_needed'] or record.get('review_needed', False))
        if result['criterion'] == 'baseline_current':
            result['reason'] = record.get('classification_reason', record.get('primary_topic_reason', result['reason']))
            result['evidence'] = record.get('framework_synthesis_evidence', '') or record.get('evidence', result['evidence'])
        if result['manual_reviewed']:
            result['category'] = record['category']
            result['reason'] = 'Existing manual category correction retained across criteria.'
            result['evidence'] = record.get('review_note', '')
            result['review_needed'] = False
        # The review-paper convention takes precedence over all topic variants.
        if result['is_review_article']:
            result['category'] = 'Theory & modeling'
            result['reason'] = 'Identified review article: Theory & modeling by user convention.'
            result['evidence'] = record.get('review_detection_evidence', '')
        assignment_rows.append(result)
assignments = pd.DataFrame(assignment_rows)
assignments['changed_from_baseline'] = assignments.category.ne(assignments.baseline_category)
comparison_papers, comparison_counts, target_checks = summarize_criteria(
    assignments, full_paper_decisions[['paper_id', 'triage_decision']], CRITERIA)

print('Counts: Before total is the full corpus; Y is retained after triage; N is rejected.')
display(comparison_counts[['criterion', 'Category', 'Before total', 'Y', 'N',
                           'Y:N', 'Needs topic review']])
print('Target checks (True = pass; False = fail). No criterion is selected by these checks.')
display(target_checks)
passes = target_checks['All targets (crystal >6)'].sum()
print('{} of {} definitions meet all requested targets using crystal >6:1.'.format(passes, len(CRITERIA)))
passes7 = target_checks['All targets (crystal >7)'].sum()
print('{} of {} definitions meet all requested targets using crystal >7:1.'.format(passes7, len(CRITERIA)))

comparison_figure = plot_criterion_comparison(
    comparison_counts, target_checks, CRITERIA, CRITERION_LABELS, COMPARISON_DIR)
display(comparison_figure)
plt.close(comparison_figure)

transition_counts = comparison_papers.groupby(
    ['criterion', 'baseline_category', 'category', 'triage_decision']).size().rename('Papers').reset_index()
changes = comparison_papers.loc[comparison_papers.changed_from_baseline].copy()
review_check = comparison_papers.loc[comparison_papers.is_review_article]
assert review_check.category.eq('Theory & modeling').all()
assert len(comparison_papers) == len(full_paper_decisions) * len(CRITERIA)
for criterion in CRITERIA:
    subset = comparison_papers.loc[comparison_papers.criterion.eq(criterion)]
    assert subset.triage_decision.value_counts().to_dict() == full_paper_decisions.triage_decision.value_counts().to_dict()

comparison_tables = {
    'criterion_definitions': definitions, 'counts_by_criterion': comparison_counts,
    'target_checks': target_checks, 'paper_assignments': comparison_papers,
    'category_changes': changes, 'transition_counts': transition_counts,
}
for name, table in comparison_tables.items():
    table.to_csv(COMPARISON_DIR / (name + '.csv'), index=False, encoding='utf-8-sig')
write_comparison_workbook(comparison_tables, COMPARISON_DIR / 'topic_criteria_comparison.xlsx')

# Hash the actual embedded criteria code, so the notebook stays self-contained.
comparison_manifest = {
    'run_time_utc': datetime.now(timezone.utc).isoformat(),
    'baseline_classifier_version': CLASSIFIER_VERSION,
    'inputs': manifest['inputs'], 'criteria': definitions.to_dict('records'),
    'corpus_size_per_criterion': len(full_paper_decisions),
    'recorded_decisions_per_criterion': full_paper_decisions.triage_decision.value_counts().to_dict(),
    'classifier_uses_triage_decisions': False,
    'criterion_selected_by_ratio': None,
    'validation': 'Exploratory rules; no expert-labeled accuracy evaluation',
    'review_policy': 'Every identified review remains Theory & modeling',
    'manual_corrections_retained': int(full_paper_decisions.manual_reviewed.sum()),
    'thresholds': {'chemical': 'Y >= 4*N and Y > 0', 'crystal': 'Y > 6*N or Y > 7*N, and Y > 0',
                   'functional': 'Y > N'},
}
if 'In' in globals():
    comparison_manifest['executed_criteria_code_sha256'] = next((
        hashlib.sha256(cell.encode('utf-8')).hexdigest() for cell in reversed(In)
        if re.search(r'(?m)^def assign_topic_criteria\(', cell)), None)
(COMPARISON_DIR / 'comparison_manifest.json').write_text(
    json.dumps(comparison_manifest, indent=2), encoding='utf-8')
print('Verified the same corpus and decisions for every definition. Saved to:', COMPARISON_DIR.resolve())
print('Individual PNG/PDF plots, per-paper evidence, count tables, and Excel audit are saved.')
