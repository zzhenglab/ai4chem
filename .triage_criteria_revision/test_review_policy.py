"""Review genre takes priority under the user's specified taxonomy."""
import unittest
from topic_classifier import classify_topic, detect_review_article, CATEGORIES


class ReviewPolicyTests(unittest.TestCase):
    def test_document_type_has_priority_over_topic(self):
        r = classify_topic('Solvothermal synthesis of new frameworks', 'We synthesized new networks.',
                           'Review; Early Access')
        self.assertEqual(r['category'], CATEGORIES[1])
        self.assertEqual(r['review_detection_source'], 'Document Type')
        self.assertTrue(r['review_policy_applied'])

    def test_review_title_and_abstract_variants(self):
        for title, abstract in [
            ('A review of MOFs for sensing', ''),
            ('MOF applications', 'This comprehensive review summarizes recent sensing work.'),
            ('MOF applications', 'This highlight review discusses synthetic approaches.'),
            ('MOF applications', 'In the mini review, published catalysts are discussed.'),
            ('MOF applications', 'In this work, we specifically review the recent literature.'),
            ('MOF applications', 'We will review recent advances in framework synthesis.'),
            ('MOF applications', 'Herein, we make a broad critical review of the literature.'),
            ('MOF applications', 'A comprehensive review of recent synthesis strategies is provided herein.'),
            ('MOF applications', 'Recent synthesis approaches are reviewed in this paper.'),
            ('MOF applications', 'The review focuses on metal organic framework applications.'),
            ('MOF applications', 'This study provides an overview of recent advances in sensing.'),
        ]:
            with self.subTest(abstract=abstract, title=title):
                result = classify_topic(title, abstract)
                self.assertEqual(result['category'], CATEGORIES[1])
                self.assertTrue(result['is_review_article'])
                self.assertTrue(result['review_detection_evidence'])
                self.assertGreater(result['score_theory_modeling'], result['score_functional_materials'])

    def test_generic_perspective_and_advances_are_not_review_proof(self):
        for title in ['Sensing from a solution perspective', 'Recent advances in a new synthetic route']:
            result = classify_topic(title, 'We synthesized a new material and measured its properties.')
            self.assertFalse(result['is_review_article'])

    def test_original_work_can_discuss_previous_literature(self):
        for abstract in [
            'The concept is reviewed. We describe the preparation and crystal structures of three new MOFs.',
            'A brief overview of the literature is presented together with three new coordination polymers.',
            'Herein a survey of the stability is presented. We synthesized samples in different solvents.',
            'We present a survey of about 40 different solvents for the synthesis of new crystals.',
            'This paper provides an overview of the solvent survey. We prepared new frameworks.',
            'A previous review by Smith discussed gas adsorption. We synthesized a new adsorbent.',
        ]:
            with self.subTest(abstract=abstract):
                self.assertFalse(detect_review_article('New framework synthesis', abstract)[0])

    def test_non_review_document_type_does_not_override_explicit_review_description(self):
        result = classify_topic('Functional materials', 'This review surveys adsorption performance.', 'Article')
        self.assertEqual(result['category'], CATEGORIES[1])
        self.assertTrue(result['review_policy_applied'])

    def test_blank_inputs_do_not_become_reviews(self):
        self.assertFalse(detect_review_article(None, float('nan'), '')[0])


if __name__ == '__main__':
    unittest.main()
