"""Manual category edits cannot silently override the review-article convention."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from triage_analysis import apply_overrides


CATEGORIES = (
    'Chemical synthesis', 'Theory & modeling', 'Crystal engineering', 'Functional materials',
)


class OverrideReviewPolicyTests(unittest.TestCase):
    def setUp(self):
        self.classified = pd.DataFrame([
            {'paper_id': 'review-paper', 'category': 'Theory & modeling',
             'is_review_article': True, 'review_needed': True,
             'review_reason': 'Genre inferred from abstract'},
            {'paper_id': 'original-paper', 'category': 'Theory & modeling',
             'is_review_article': False, 'review_needed': True,
             'review_reason': 'Mixed experimental and theoretical study'},
        ])

    def apply(self, edits):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manual_overrides.csv'
            pd.DataFrame(edits).to_csv(path, index=False)
            return apply_overrides(self.classified, path, CATEGORIES)

    def test_non_theory_override_for_review_is_rejected_before_applying_edits(self):
        for category in ['Chemical synthesis', 'Crystal engineering', 'Functional materials']:
            with self.subTest(category=category):
                with self.assertRaisesRegex(ValueError, 'Review articles must remain Theory & modeling') as caught:
                    self.apply([
                        {'paper_id': 'original-paper', 'manual_category': 'Chemical synthesis'},
                        {'paper_id': 'review-paper', 'manual_category': category},
                    ])
                self.assertIn('review-paper', str(caught.exception))
                self.assertIn('correct its genre metadata/detection', str(caught.exception))
        self.assertTrue(self.classified.category.eq('Theory & modeling').all())
        self.assertNotIn('manual_reviewed', self.classified)

    def test_confirmed_review_can_be_manually_kept_in_theory(self):
        result = self.apply([{
            'paper_id': 'review-paper', 'manual_category': 'Theory & modeling',
            'review_note': 'Confirmed the article is a review.',
        }]).set_index('paper_id')
        self.assertEqual(result.loc['review-paper', 'category'], 'Theory & modeling')
        self.assertTrue(result.loc['review-paper', 'manual_reviewed'])
        self.assertFalse(result.loc['review-paper', 'review_needed'])
        self.assertEqual(result.loc['review-paper', 'review_note'], 'Confirmed the article is a review.')

    def test_non_review_theory_paper_can_be_corrected_to_another_topic(self):
        result = self.apply([{
            'paper_id': 'original-paper', 'manual_category': 'Chemical synthesis',
        }]).set_index('paper_id')
        self.assertEqual(result.loc['original-paper', 'category'], 'Chemical synthesis')
        self.assertTrue(result.loc['original-paper', 'manual_reviewed'])
        self.assertEqual(result.loc['review-paper', 'category'], 'Theory & modeling')

    def test_absent_override_file_preserves_automatic_review_category(self):
        with tempfile.TemporaryDirectory() as directory:
            result = apply_overrides(self.classified, Path(directory) / 'not_created.csv', CATEGORIES)
        self.assertTrue(result.category.eq('Theory & modeling').all())
        self.assertFalse(result.manual_reviewed.any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
