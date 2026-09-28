"""Local, auditable identity matching and counts for literature triage."""
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd


TRIAGE_COLUMNS = ['gpt-5', 'gpt-4o-mini', 'ground truth by gpt-5', 'Agent_YN']
METADATA_COLUMNS = ['Article Title', 'Abstract', 'DOI', 'UT (Unique WOS ID)',
                    'Publication Year', 'Source Title', 'Document Type']


def normalize_text(value):
    if value is None or pd.isna(value):
        return ''
    return unicodedata.normalize('NFKC', html.unescape(str(value))).strip()


def normalize_doi(value):
    value = normalize_text(value).casefold()
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value)
    value = re.sub(r'^doi\s*:\s*', '', value)
    return value.strip()


def normalize_title(value):
    return ''.join(c for c in normalize_text(value).casefold() if c.isalnum())


def is_y(values):
    """Only an exact Y, after trimming and case normalization, is included."""
    return values.fillna('').astype(str).str.strip().str.upper().eq('Y')


def prepare_records(frame, source):
    frame = frame.copy().fillna('')
    for col in METADATA_COLUMNS:
        if col not in frame:
            frame[col] = ''
    frame['source_file'] = str(source)
    frame['excel_row'] = range(2, len(frame) + 2)
    frame['doi_key'] = frame['DOI'].map(normalize_doi)
    frame['wos_key'] = frame['UT (Unique WOS ID)'].map(lambda x: normalize_text(x).casefold())
    frame['title_key'] = frame['Article Title'].map(normalize_title)
    frame['paper_id'] = [
        'doi:' + doi if doi else 'wos:' + wos if wos else 'title:' + title if title
        else 'row:{}:{}'.format(Path(source).name, row)
        for doi, wos, title, row in zip(frame.doi_key, frame.wos_key,
                                      frame.title_key, frame.excel_row)
    ]
    return frame


def read_records(path, sheet_name='Sheet1'):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Set DATA_DIR to the folder containing both input files: ' + str(path))
    frame = pd.read_excel(path, sheet_name=sheet_name, dtype=str, keep_default_na=False,
                          usecols=lambda c: c in METADATA_COLUMNS + TRIAGE_COLUMNS)
    if 'Article Title' not in frame or 'Abstract' not in frame:
        raise ValueError('Expected Article Title and Abstract columns in ' + str(path))
    return prepare_records(frame, path)


def match_after_to_before(before, after):
    """Match DOI first, then WOS ID, then an unambiguous normalized title.

    Duplicate rows sharing a DOI resolve to the same paper. Conflicting IDs
    remain unmatched; a supplied, different DOI is never silently replaced.
    """
    maps = {}
    for column in ['doi_key', 'wos_key', 'title_key']:
        maps[column] = before.loc[before[column].ne('')].groupby(column)['paper_id'].agg(set).to_dict()
    known_dois = before.groupby('paper_id')['doi_key'].agg(lambda s: set(s) - {''}).to_dict()
    before_ids = set(before.paper_id)
    records = []
    for _, row in after.iterrows():
        candidates = [(col, maps[col].get(row[col], set()))
                      for col in ['doi_key', 'wos_key', 'title_key'] if row[col]]
        hits = [(col, ids) for col, ids in candidates if ids]
        chosen, method, issue = None, 'unmatched', ''
        for col, ids in hits:
            if len(ids) == 1:
                chosen, method = next(iter(ids)), col.replace('_key', '')
                break
        if chosen is not None:
            # IDs are stronger than titles, but disagreement in either is visible.
            conflicts = [col for col, ids in hits if chosen not in ids]
            if row.doi_key and known_dois[chosen] and row.doi_key not in known_dois[chosen]:
                conflicts.append('different DOI')
            if conflicts:
                chosen, method, issue = None, 'conflict', '; '.join(conflicts)
        elif hits:
            method, issue = 'ambiguous', 'More than one before-paper matches the available identifiers.'
        paper_id = chosen if chosen is not None else row.paper_id
        if chosen is None and paper_id in before_ids:
            paper_id = 'unmatched:' + paper_id
        records.append((paper_id, chosen is not None, method, issue))
    out = after.copy()
    out[['paper_id', 'matched_to_before', 'match_method', 'match_issue']] = pd.DataFrame(
        records, index=after.index, columns=['paper_id', 'matched_to_before', 'match_method', 'match_issue'])
    return out


def unique_papers(records):
    """One representative per DOI (fallback WOS/title); retain source row provenance."""
    out = records.copy()
    out['_text_length'] = out['Article Title'].str.len() + out['Abstract'].str.len()
    out = out.sort_values('_text_length', ascending=False, kind='stable').drop_duplicates('paper_id')
    provenance = records.groupby('paper_id', sort=False).agg(
        source_rows=('excel_row', lambda s: ', '.join(str(v) for v in s)),
        source_record_count=('excel_row', 'size'))
    return out.drop(columns='_text_length').merge(provenance, on='paper_id', validate='one_to_one')


def label_inventory(records, label_columns=TRIAGE_COLUMNS):
    rows = []
    for col in label_columns:
        if col in records:
            labels = records[col].fillna('').astype(str).str.strip().str.upper()
            for label, n in labels.value_counts(dropna=False).items():
                rows.append({'label_column': col, 'label': label or '(blank)',
                             'rows': int(n),
                             'unique_papers': records.loc[labels.eq(label), 'paper_id'].nunique()})
    return pd.DataFrame(rows)


def apply_overrides(classified, override_path, categories):
    result = classified.copy()
    result['automatic_category'] = result['category']
    result['manual_reviewed'] = False
    result['review_note'] = ''
    result['classification_method'] = 'topic rules'
    path = Path(override_path)
    if not path.exists():
        return result
    edits = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'paper_id', 'manual_category'}
    if not required.issubset(edits):
        raise ValueError('Override CSV must have paper_id and manual_category columns.')
    edits['manual_category'] = edits.manual_category.str.strip()
    edits = edits.loc[edits.manual_category.ne('')].copy()
    if edits.paper_id.duplicated().any():
        raise ValueError('Only one manual override per paper_id is allowed.')
    invalid = set(edits.manual_category) - set(categories)
    unknown = set(edits.paper_id) - set(result.paper_id)
    if invalid or unknown:
        raise ValueError('Invalid override categories or unknown paper IDs: ' + repr((invalid, unknown)))
    if 'is_review_article' in result.columns:
        review_ids = set(result.loc[result.is_review_article.eq(True), 'paper_id'])
        incompatible = edits.loc[
            edits.paper_id.isin(review_ids)
            & edits.manual_category.ne('Theory & modeling'), 'paper_id'].tolist()
        if incompatible:
            raise ValueError(
                'Review articles must remain Theory & modeling under the requested taxonomy. '
                'Incompatible manual category override for paper IDs: ' + repr(incompatible)
                + '. If a paper is not a review, correct its genre metadata/detection before '
                'applying the category override.')
    for _, edit in edits.iterrows():
        mask = result.paper_id.eq(edit.paper_id)
        result.loc[mask, 'category'] = edit.manual_category
        result.loc[mask, 'manual_reviewed'] = True
        result.loc[mask, 'review_needed'] = False
        result.loc[mask, 'review_reason'] = ''
        result.loc[mask, 'review_note'] = edit.get('review_note', '')
        result.loc[mask, 'classification_method'] = 'manual override'
    return result


def attach_classification(records, classified):
    metadata = set(METADATA_COLUMNS + ['doi_key', 'wos_key', 'title_key', 'source_file',
                                       'excel_row', 'source_rows', 'source_record_count',
                                       'match_method', 'match_issue', 'matched_to_before',
                                       'selected_Y', 'in_before', 'after_selected_Y'])
    annotation_cols = [c for c in classified if c not in metadata and c not in TRIAGE_COLUMNS]
    return records.merge(classified[annotation_cols], on='paper_id', how='left', validate='many_to_one')


def count_categories(before_classified, after_y_classified, categories, unique=True, subset=True,
                     full_decisions=None):
    """Count topics, optionally splitting the full corpus by explicit decisions.

    With full_decisions, Y/N/Unknown/Conflict partition the same before population.
    The selected workbook validates the Y paper identities and their categories;
    row-mode split counts use before-source multiplicities, not selected-file
    multiplicities. Missing or conflicting decisions are never inferred to be N.
    Calls without full_decisions retain the original before/selected-Y behavior.
    """
    if before_classified.groupby('paper_id').category.nunique(dropna=False).gt(1).any():
        raise ValueError('Duplicate before rows must share one category per paper.')
    if after_y_classified.groupby('paper_id').category.nunique(dropna=False).gt(1).any():
        raise ValueError('Duplicate selected rows must share one category per paper.')
    before = before_classified.drop_duplicates('paper_id') if unique else before_classified
    after = after_y_classified.drop_duplicates('paper_id') if unique else after_y_classified
    canonical_categories = before_classified.drop_duplicates('paper_id').set_index('paper_id').category
    matched_after = after.loc[after.paper_id.isin(canonical_categories.index)]
    if not matched_after.category.eq(matched_after.paper_id.map(canonical_categories)).all():
        raise ValueError('Selected papers must reuse their before-triage category.')
    split = None
    if full_decisions is not None:
        if not full_decisions.paper_id.is_unique:
            raise ValueError('Full decisions require one row per paper.')
        if set(full_decisions.paper_id) != set(before.paper_id):
            raise ValueError('Full decisions must cover exactly the before-triage paper IDs.')
        if not full_decisions.triage_decision.isin(['Y', 'N', 'Unknown', 'Conflict']).all():
            raise ValueError('Full decisions must use Y, N, Unknown, or Conflict.')
        if 'category' in full_decisions and not full_decisions.category.eq(
                full_decisions.paper_id.map(canonical_categories)).all():
            raise ValueError('Full decisions must reuse their before-triage category.')
        explicit_y = set(full_decisions.loc[full_decisions.triage_decision.eq('Y'), 'paper_id'])
        if explicit_y != set(after.paper_id):
            raise ValueError('Selected Y paper IDs differ from explicit full-corpus Y decisions.')
        decisions = before.paper_id.map(full_decisions.set_index('paper_id').triage_decision)
        split = {label: before.loc[decisions.eq(label)]
                 for label in ['Y', 'N', 'Unknown', 'Conflict']}
        after = split['Y']
    order = list(categories) + ['Unclassified']
    result = pd.DataFrame(index=pd.Index(order, name='Category'))
    result['Before triage'] = before.category.value_counts().reindex(order, fill_value=0)
    result['After triage (Y)'] = after.category.value_counts().reindex(order, fill_value=0)
    if split is not None:
        for label in ['N', 'Unknown', 'Conflict']:
            result['After triage (' + label + ')'] = split[label].category.value_counts().reindex(order, fill_value=0)
    result['Before needing review'] = before.groupby('category').review_needed.sum().reindex(order, fill_value=0)
    result['After Y needing review'] = after.groupby('category').review_needed.sum().reindex(order, fill_value=0)
    if split is not None:
        result['After N needing review'] = split['N'].groupby('category').review_needed.sum().reindex(order, fill_value=0)
    result = result.astype(int)
    result.loc['TOTAL'] = result.sum()
    result['Before share (%)'] = (100 * result['Before triage'] / len(before)).round(2) if len(before) else float('nan')
    result['After Y share (%)'] = (100 * result['After triage (Y)'] / len(after)).round(2) if len(after) else float('nan')
    if split is not None:
        result['After N share (%)'] = (100 * result['After triage (N)'] / len(split['N'])).round(2) if len(split['N']) else float('nan')
        assert result[['After triage (' + label + ')' for label in split]].sum(axis=1).eq(result['Before triage']).all()
    if subset and unique:
        result['Retained (%)'] = (100 * result['After triage (Y)'] /
                                  result['Before triage'].replace(0, float('nan'))).round(2)
    assert result.loc['TOTAL', 'Before triage'] == len(before)
    assert result.loc['TOTAL', 'After triage (Y)'] == len(after)
    if unique and subset:
        assert set(after.paper_id).issubset(set(before.paper_id))
        assert result['After triage (Y)'].le(result['Before triage']).all()
    return result


def attach_full_decisions(before_papers, decisions, column='Agent_YN'):
    """Attach explicit decisions without inventing N for missing/ambiguous records.

    before_papers must contain one row per canonical paper ID. The decisions
    input has been identity-matched against the full corpus. Mixed Y/N duplicate
    records remain Conflict, and blanks/unsupported labels remain Unknown.
    """
    if not before_papers.paper_id.is_unique:
        raise ValueError('Expected one row per before paper.')
    if column not in decisions:
        raise ValueError('Missing explicit decision column: ' + column)
    mapped = decisions.loc[decisions.matched_to_before.astype(bool)].copy()
    mapped['_decision'] = mapped[column].fillna('').astype(str).str.strip().str.upper()

    def consensus(values):
        labels = set(values) - {''}
        if labels == {'Y'}:
            return 'Y'
        if labels == {'N'}:
            return 'N'
        if 'Y' in labels and 'N' in labels:
            return 'Conflict'
        return 'Unknown'

    decision = mapped.groupby('paper_id')['_decision'].agg(consensus)
    result = before_papers.copy()
    result['triage_decision'] = result.paper_id.map(decision).fillna('Unknown')
    result['decision_column'] = column
    result['decision_source'] = result.paper_id.map(
        mapped.groupby('paper_id').source_file.first()).fillna('')
    return result


def count_yn(papers, categories):
    """Disjoint unique-paper Y/N groups, with unresolved decisions explicit."""
    if not papers.paper_id.is_unique:
        raise ValueError('Y/N counts require unique papers.')
    order = list(categories) + ['Unclassified']
    out = pd.crosstab(papers.category, papers.triage_decision).reindex(
        index=order, columns=['Y', 'N', 'Unknown', 'Conflict'], fill_value=0).astype(int)
    out.index.name = 'Category'
    out.columns.name = None
    out['Before total'] = out.sum(axis=1)
    out.loc['TOTAL'] = out.sum()
    out['Y among decided (%)'] = (100 * out.Y / (out.Y + out.N).replace(0, float('nan'))).round(2)
    assert out.loc['TOTAL', 'Before total'] == len(papers)
    return out
