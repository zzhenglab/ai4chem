"""Boundary tests for fixed text-only criteria; no real paper decisions are used."""

import inspect
import unittest

from topic_criteria import CRITERIA, assign_topic_criteria


class TopicCriteriaBoundaries(unittest.TestCase):
    def classify(self, title, abstract="", **kwargs):
        rows = assign_topic_criteria(title, abstract, **kwargs)
        self.assertEqual(tuple(row["criterion"] for row in rows), CRITERIA)
        self.assertTrue(all(set(row) == {"criterion", "category", "reason", "evidence", "review_needed"}
                            for row in rows))
        return {row["criterion"]: row for row in rows}

    def test_review_policy_in_all_variants(self):
        rows = self.classify("Synthesis and adsorption in MOFs", "We describe their properties.", document_type="Review")
        self.assertTrue(all(row["category"] == "Theory & modeling" for row in rows.values()))
        self.assertTrue(all(not row["review_needed"] for row in rows.values()))

    def test_catalytic_product_is_application(self):
        rows = self.classify(
            "MOF-catalyzed synthesis of cyclic carbonates",
            "We use a metal organic framework as a catalyst for cycloaddition. The catalyst exhibits high conversion and selectivity.",
            baseline_category="Chemical synthesis")
        self.assertEqual(rows["baseline_current"]["category"], "Chemical synthesis")
        for key in CRITERIA[1:]:
            self.assertEqual(rows[key]["category"], "Functional materials")

    def test_framework_route_is_synthesis(self):
        rows = self.classify(
            "Mechanochemical synthesis of metal organic frameworks",
            "We developed a solvent-free synthetic method. The frameworks were synthesized by ball milling.")
        self.assertTrue(all(row["category"] == "Chemical synthesis" for row in rows.values()))

    def test_organic_synthesis_target_with_framework_catalyst(self):
        rows = self.classify(
            "One-pot synthesis of cyclic carbonates with a MOF catalyst",
            "We investigate substrate conversion and observe high catalytic activity.",
            baseline_category="Chemical synthesis")
        self.assertEqual(rows["framework_scope"]["category"], "Functional materials")

    def test_background_catalysis_does_not_change_route(self):
        rows = self.classify(
            "Synthesis optimization of zirconium MOFs",
            "Frameworks can be catalysts for cycloaddition. We developed a synthetic method and optimized synthesis kinetics.",
            baseline_category="Chemical synthesis")
        self.assertEqual(rows["framework_scope"]["category"], "Chemical synthesis")

    def test_title_property_gets_priority_over_structure(self):
        rows = self.classify(
            "Synthesis, crystal structures and luminescence of new coordination polymers",
            "Two new coordination polymers were synthesized. Crystal structures reveal interpenetrated networks.",
            baseline_category="Crystal engineering")
        self.assertEqual(rows["framework_scope"]["category"], "Crystal engineering")
        self.assertEqual(rows["property_priority"]["category"], "Functional materials")
        self.assertTrue(rows["property_priority"]["review_needed"])

    def test_brief_abstract_property_does_not_override_design(self):
        rows = self.classify(
            "Synthesis and topological design of new coordination polymers",
            "Two new coordination polymers were synthesized. Single-crystal X-ray diffraction reveals their network topology. Luminescence was also measured.",
            baseline_category="Crystal engineering")
        self.assertEqual(rows["property_priority"]["category"], "Crystal engineering")

    def test_xrd_alone_does_not_establish_crystal_engineering(self):
        rows = self.classify(
            "Crystal structure characterization of a material",
            "The sample was characterized by X-ray diffraction.",
            baseline_category="Crystal engineering")
        self.assertEqual(rows["property_priority"]["category"], "Unclassified")
        self.assertEqual(rows["computation_inclusive"]["category"], "Unclassified")

    def test_primary_computation_in_mixed_work(self):
        rows = self.classify(
            "Experimental and density functional theory study of MOF adsorption",
            "A new metal organic framework was synthesized. We investigate adsorption using density functional theory calculations.",
            baseline_category="Functional materials")
        self.assertEqual(rows["property_priority"]["category"], "Functional materials")
        self.assertEqual(rows["computation_inclusive"]["category"], "Theory & modeling")

    def test_supporting_computation_remains_application(self):
        rows = self.classify(
            "A fluorescent MOF for sensing pollutants supported by DFT calculations",
            "The framework was synthesized. DFT calculations explain the observed response.",
            baseline_category="Functional materials")
        self.assertEqual(rows["computation_inclusive"]["category"], "Functional materials")

    def test_non_framework_synthetic_method_is_retained_for_review(self):
        rows = self.classify(
            "A synthetic route for stereoselective synthesis of chiral alcohols",
            "We developed a synthetic method to prepare chiral alcohols.",
            baseline_category="Chemical synthesis")
        self.assertEqual(rows["property_priority"]["category"], "Chemical synthesis")
        self.assertTrue(rows["property_priority"]["review_needed"])

    def test_empty_text_is_unclassified(self):
        rows = self.classify(None, float("nan"))
        self.assertTrue(all(row["category"] == "Unclassified" for row in rows.values()))
        self.assertTrue(all(row["review_needed"] for row in rows.values()))

    def test_interface_has_no_decision_input_and_calls_are_independent(self):
        self.assertEqual(list(inspect.signature(assign_topic_criteria).parameters),
                         ["title", "abstract", "document_type", "baseline_category"])
        first = assign_topic_criteria("MOF-catalyzed cycloaddition", "We observe high catalytic activity.")
        assign_topic_criteria("A review of MOF synthesis", "This review describes synthesis.")
        self.assertEqual(first, assign_topic_criteria("MOF-catalyzed cycloaddition", "We observe high catalytic activity."))
        with self.assertRaises(TypeError):
            assign_topic_criteria("A paper", "", triage_decision="Y")


if __name__ == "__main__":
    unittest.main(verbosity=2)
