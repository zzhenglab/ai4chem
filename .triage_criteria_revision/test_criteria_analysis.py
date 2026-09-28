"""Synthetic boundary checks for fixed-topic Y/N comparison and audit export.

Run: python -m unittest discover -s .triage_criteria_revision -p test_criteria_analysis.py -v
"""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from criteria_analysis import (
    TOPIC_ORDER, ratio_text, summarize_criteria, write_comparison_workbook, yn_ratio,
)


CHEM, THEORY, CRYSTAL, FUNCTIONAL, UNCLASSIFIED = TOPIC_ORDER


def corpus(groups, criteria=('base',)):
    """Expand category -> decision/count mappings into invented unique papers."""
    decisions, assignments = [], []
    for category, labels in groups.items():
        for decision, size in labels.items():
            for _ in range(size):
                paper_id = 'paper-{}'.format(len(decisions))
                decisions.append({'paper_id': paper_id, 'triage_decision': decision})
                for criterion in criteria:
                    assignments.append({'criterion': criterion, 'paper_id': paper_id,
                                        'category': category, 'review_needed': False})
    return pd.DataFrame(assignments), pd.DataFrame(decisions)


class ThresholdTests(unittest.TestCase):
    def target(self, chemical=(4, 1), crystal=(8, 1), functional=(2, 1), extra=None):
        groups = {CHEM: dict(zip(('Y', 'N'), chemical)),
                  CRYSTAL: dict(zip(('Y', 'N'), crystal)),
                  FUNCTIONAL: dict(zip(('Y', 'N'), functional))}
        groups.update(extra or {})
        assignments, decisions = corpus(groups)
        return summarize_criteria(assignments, decisions, ['base'])[2].iloc[0]

    def test_chemical_includes_exact_four_to_one(self):
        for y, n, expected in [(3, 1, False), (4, 1, True), (7, 2, False), (8, 2, True)]:
            with self.subTest(y=y, n=n):
                target = self.target(chemical=(y, n))
                self.assertEqual(target['Chemical >=4:1'], expected)
                self.assertEqual(target['All targets (crystal >7)'], expected)

    def test_crystal_six_and_seven_are_strict_thresholds(self):
        for y, n, passes_six, passes_seven in [
                (6, 1, False, False), (7, 1, True, False), (8, 1, True, True),
                (12, 2, False, False), (13, 2, True, False), (14, 2, True, False),
                (15, 2, True, True)]:
            with self.subTest(y=y, n=n):
                target = self.target(crystal=(y, n))
                self.assertEqual(target['Crystal >6:1'], passes_six)
                self.assertEqual(target['Crystal >7:1'], passes_seven)
                self.assertEqual(target['All targets (crystal >6)'], passes_six)
                self.assertEqual(target['All targets (crystal >7)'], passes_seven)

    def test_functional_requires_y_strictly_greater_than_n(self):
        for y, n, expected in [(1, 2, False), (2, 2, False), (3, 2, True)]:
            with self.subTest(y=y, n=n):
                target = self.target(functional=(y, n))
                self.assertEqual(target['Functional Y>N'], expected)
                self.assertEqual(target['All targets (crystal >6)'], expected)

    def test_zero_n_with_positive_y_is_infinite_and_passes(self):
        target = self.target(chemical=(1, 0), crystal=(1, 0), functional=(1, 0))
        self.assertTrue(target['All targets (crystal >7)'])
        for field in ['Chemical Y:N', 'Crystal Y:N', 'Functional Y:N']:
            self.assertEqual(target[field], 'infinite (N=0)')
        self.assertTrue(math.isinf(yn_ratio(1, 0)))

    def test_empty_topic_is_undefined_and_never_passes(self):
        for argument, pass_field, ratio_field in [
                ('chemical', 'Chemical >=4:1', 'Chemical Y:N'),
                ('crystal', 'Crystal >6:1', 'Crystal Y:N'),
                ('functional', 'Functional Y>N', 'Functional Y:N')]:
            with self.subTest(argument=argument):
                target = self.target(**{argument: (0, 0)})
                self.assertFalse(target[pass_field])
                self.assertFalse(target['All targets (crystal >6)'])
                self.assertFalse(target['All targets (crystal >7)'])
                self.assertEqual(target[ratio_field], 'undefined')
        self.assertTrue(math.isnan(yn_ratio(0, 0)))
        self.assertEqual(ratio_text(yn_ratio(0, 0)), 'undefined')

    def test_n_without_y_has_zero_ratio_and_fails(self):
        target = self.target(chemical=(0, 1), crystal=(0, 1), functional=(0, 1))
        self.assertFalse(target['Chemical >=4:1'])
        self.assertFalse(target['Crystal >6:1'])
        self.assertFalse(target['Functional Y>N'])
        self.assertEqual(target['Chemical Y:N'], '0.00:1')

    def test_unknown_or_conflict_prevents_overall_pass(self):
        for decision in ['Unknown', 'Conflict']:
            with self.subTest(decision=decision):
                target = self.target(extra={THEORY: {decision: 1}})
                self.assertFalse(target['Complete decisions'])
                self.assertFalse(target['All targets (crystal >6)'])
                self.assertFalse(target['All targets (crystal >7)'])
                self.assertTrue(target['Chemical >=4:1'])
                self.assertTrue(target['Crystal >7:1'])


class CorpusIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.order = ['specific', 'broad']
        self.assignments, self.decisions = corpus({
            CHEM: {'Y': 1, 'N': 1}, THEORY: {'Unknown': 1},
            CRYSTAL: {'Conflict': 1}, UNCLASSIFIED: {'Y': 1},
        }, self.order)

    def summarize(self, assignments=None, decisions=None, order=None):
        return summarize_criteria(
            self.assignments if assignments is None else assignments,
            self.decisions if decisions is None else decisions,
            self.order if order is None else order)

    def test_every_criterion_preserves_same_ids_decisions_and_total(self):
        assignments = self.assignments.copy()
        mask = assignments.criterion.eq('broad') & assignments.category.eq(UNCLASSIFIED)
        assignments.loc[mask, 'category'] = FUNCTIONAL
        assignments.loc[mask, 'review_needed'] = True
        annotated, counts, targets = self.summarize(assignments=assignments)
        expected = self.decisions.set_index('paper_id').triage_decision.to_dict()
        for criterion in self.order:
            group = annotated.loc[annotated.criterion.eq(criterion)]
            self.assertEqual(group.set_index('paper_id').triage_decision.to_dict(), expected)
            table = counts.loc[counts.criterion.eq(criterion)]
            self.assertEqual(table['Before total'].sum(), 5)
            self.assertEqual(table[['Y', 'N', 'Unknown', 'Conflict']].sum().to_dict(),
                             {'Y': 2, 'N': 1, 'Unknown': 1, 'Conflict': 1})
            self.assertEqual(set(table.Category), set(TOPIC_ORDER))
        by_criterion = targets.set_index('criterion')
        self.assertEqual(by_criterion.loc['specific', 'Topic coverage (%)'], 80)
        self.assertEqual(by_criterion.loc['broad', 'Topic coverage (%)'], 100)
        broad_functional = counts.loc[counts.criterion.eq('broad') & counts.Category.eq(FUNCTIONAL)]
        self.assertEqual(broad_functional.iloc[0]['Needs topic review'], 1)
        self.assertTrue(targets['Total papers'].eq(5).all())
        self.assertFalse(targets['Complete decisions'].any())

    def test_duplicate_decision_records_are_rejected(self):
        duplicates = pd.concat([self.decisions, self.decisions.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'one row per paper'):
            self.summarize(decisions=duplicates)

    def test_duplicate_assignment_within_criterion_is_rejected(self):
        duplicates = pd.concat([self.assignments, self.assignments.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'multiple assignments'):
            self.summarize(assignments=duplicates)

    def test_dropped_paper_is_rejected_even_when_other_criterion_is_complete(self):
        with self.assertRaisesRegex(ValueError, 'entire same corpus'):
            self.summarize(assignments=self.assignments.iloc[1:])

    def test_extra_paper_without_decision_is_rejected(self):
        extra = self.assignments.iloc[[0]].assign(paper_id='not-in-decisions')
        with self.assertRaisesRegex(ValueError, 'entire same corpus'):
            self.summarize(assignments=pd.concat([self.assignments, extra], ignore_index=True))

    def test_missing_or_undeclared_criterion_is_rejected(self):
        for order in [['specific'], ['specific', 'broad', 'missing']]:
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, 'declared criterion'):
                self.summarize(order=order)

    def test_unknown_topic_and_unsupported_decisions_are_rejected(self):
        invalid = self.assignments.copy()
        invalid.loc[0, 'category'] = 'unlisted category'
        with self.assertRaisesRegex(ValueError, 'Unknown topic category'):
            self.summarize(assignments=invalid)
        for value in ['Yes', '', None]:
            invalid = self.decisions.copy()
            invalid.loc[0, 'triage_decision'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Unsupported decision'):
                self.summarize(decisions=invalid)


class WorkbookTests(unittest.TestCase):
    def test_streamed_text_is_text_and_nonfinite_values_are_explicit(self):
        from openpyxl import load_workbook
        table = pd.DataFrame({
            'Article Title': ['=1+1', '=HYPERLINK("https://example.org","title")', 'Normal title'],
            'ratio': [float('inf'), float('-inf'), float('nan')],
            'count': [3, 2, 1],
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'comparison.xlsx'
            write_comparison_workbook({'papers': table}, path)
            workbook = load_workbook(path, data_only=False)
            try:
                sheet = workbook['papers']
                self.assertEqual(sheet['A2'].value, '=1+1')
                self.assertEqual(sheet['A2'].data_type, 's')
                self.assertEqual(sheet['A3'].value, table.iloc[1, 0])
                self.assertEqual(sheet['A3'].data_type, 's')
                self.assertEqual(sheet['B2'].value, 'infinite')
                self.assertEqual(sheet['B3'].value, '-infinite')
                self.assertIsNone(sheet['B4'].value)
                self.assertEqual(sheet['C2'].value, 3)
                self.assertEqual(sheet.freeze_panes, 'A2')
                self.assertEqual(sheet.auto_filter.ref, 'A1:C4')
            finally:
                workbook.close()


if __name__ == '__main__':
    unittest.main()
