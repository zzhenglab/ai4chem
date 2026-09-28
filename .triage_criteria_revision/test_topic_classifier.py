"""Contribution-boundary regression cases; not a measured accuracy benchmark.

Short abstracts here are synthetic examples. They exercise distinctions that
matter to primary-topic classification without any triage-label input.
"""

import inspect
import unittest

from topic_classifier import CATEGORIES, CLASSIFIER_VERSION, UNCLASSIFIED, classify_topic


class TopicContributionTests(unittest.TestCase):
    def assert_topic(self, title, abstract, expected):
        result = classify_topic(title, abstract)
        self.assertEqual(result["category"], expected, result)
        self.assertTrue(result["primary_topic_reason"])
        return result

    def test_computer_simulations_in_title(self):
        self.assert_topic(
            "Computer simulations for the adsorption and separation of CH4/H2/CO2/N2 gases",
            "Grand canonical Monte Carlo simulations were performed to investigate gas separation.",
            CATEGORIES[1])

    def test_primary_theory_in_abstract_overrides_application_title(self):
        result = self.assert_topic(
            "Water coordination and dehydration processes in defective UiO-66 frameworks",
            "The present work provides a theoretical understanding of water adsorption by means of "
            "periodic density functional theory calculations and ab initio molecular dynamics simulations.",
            CATEGORIES[1])
        self.assertTrue(result["abstract_computational_focus"])
        self.assertFalse(result["has_experimental_synthesis"])

    def test_computational_screening_of_photocatalysts(self):
        self.assert_topic(
            "The search for efficient and stable frameworks for photocatalysis",
            "We perform computational screening of frameworks. Density functional theory calculations "
            "predict the band structures of candidate photocatalysts for nitrogen fixation.",
            CATEGORIES[1])

    def test_supporting_dft_is_not_primary_theory(self):
        result = self.assert_topic(
            "A fluorescent framework for sensing aqueous pollutants",
            "We synthesized a porous framework and measured its detection limit. "
            "DFT calculations explain the observed fluorescence response.",
            CATEGORIES[3])
        self.assertFalse(result["abstract_computational_focus"])
        self.assertTrue(result["supporting_computation_evidence"])
        self.assertTrue(result["has_experimental_synthesis"])

    def test_molecular_reaction_with_auxiliary_computation(self):
        result = self.assert_topic(
            "Combined experimental and theoretical study of a MOF catalyst for Pechmann condensation",
            "We prepared a catalyst and optimized the reaction yield. "
            "DFT calculations support the observed molecular reaction mechanism.",
            CATEGORIES[0])
        self.assertTrue(result["mixed_experimental_computational"])
        self.assertTrue(result["review_needed"])

    def test_structure_report_with_incidental_luminescence(self):
        self.assert_topic(
            "Syntheses, Structures, and Photoluminescent Properties of Coordination Polymers",
            "Nine new coordination polymers have been synthesized. Their crystal structures reveal "
            "three dimensional interpenetrated networks. The topology and coordination modes are "
            "discussed in detail. Finally, their luminescence was measured.",
            CATEGORIES[2])

    def test_hydrothermal_structure_report(self):
        self.assert_topic(
            "Hydrothermal synthesis, crystal structure and luminescence of four novel metal-organic frameworks",
            "Four novel metal-organic frameworks were synthesized hydrothermally. "
            "Single-crystal X-ray diffraction reveals distinct two dimensional networks. "
            "The coordination modes and topology are compared. Their luminescence is also measured.",
            CATEGORIES[2])

    def test_crystal_construction_has_synthesis_tag(self):
        result = self.assert_topic(
            "Ligand-directed assembly of coordination polymers",
            "Two new coordination polymers have been synthesized. "
            "Their crystal structures display different interpenetration and network topology.",
            CATEGORIES[2])
        self.assertTrue(result["has_experimental_synthesis"])
        self.assertIn("synthesized", result["experimental_synthesis_evidence"])

    def test_generic_xrd_is_not_crystal_engineering(self):
        self.assert_topic(
            "New metal organic frameworks",
            "Materials were characterized by single-crystal X-ray diffraction.",
            UNCLASSIFIED)

    def test_routine_preparation_does_not_override_application(self):
        self.assert_topic(
            "One-step synthesis of mesoporous ZnO@ZIF-8 composites for CO2 adsorption and separation",
            "The composite was prepared under solvothermal conditions. "
            "We measured adsorption capacities and gas separation selectivities.",
            CATEGORIES[3])

    def test_transferable_synthetic_method(self):
        self.assert_topic(
            "Mechanochemical synthesis of metal-organic frameworks",
            "We developed a solvent-free synthetic route to frameworks using ball milling.",
            CATEGORIES[0])

    def test_postsynthetic_route_with_structural_consequences(self):
        self.assert_topic(
            "Dual-ligand framework crystals and oriented films derived from a metastable precursor",
            "We developed a post-synthetic ligand exchange route. Ligand exchange proceeds through "
            "a heterogeneous nucleation mechanism and allows the synthesis of a new framework. "
            "The resulting topology was characterized.",
            CATEGORIES[0])

    def test_derived_carbons_are_functional_materials(self):
        self.assert_topic(
            "Metal-organic framework-derived porous carbons as high-performance supercapacitor electrodes",
            "We prepared nitrogen-doped carbons by pyrolysis. The electrodes show high capacitance "
            "and long cycling stability.",
            CATEGORIES[3])

    def test_biomedical_uptake(self):
        self.assert_topic(
            "Cellular uptake and cytotoxicity of nanoscale metal-organic frameworks",
            "We investigate cellular uptake and drug delivery performance.",
            CATEGORIES[3])

    def test_molecular_vs_energy_catalysis(self):
        self.assert_topic("C-H functionalization by a porous framework",
                          "A coupling reaction gives high product yields.", CATEGORIES[0])
        self.assert_topic("Photocatalytic hydrogen production by a porous framework",
                          "Hydrogen evolution activity is measured under light.", CATEGORIES[3])

    def test_application_led_photo_oxidative_coupling(self):
        self.assert_topic(
            "A donor-acceptor MOF for boosting photocatalytic oxidative coupling of amines",
            "We prepared a photocatalyst and measured its activity for solar energy conversion. "
            "DFT calculations support the observed charge-transfer mechanism.",
            CATEGORIES[3])

    def test_missing_synthesis_does_not_imply_theory(self):
        self.assert_topic("Unusual properties of a new material", "More work is needed.", UNCLASSIFIED)

    def test_review_is_flagged(self):
        result = self.assert_topic(
            "Recent advances in metal-organic frameworks for drug delivery",
            "This review summarizes drug loading and delivery performance.", CATEGORIES[1])
        self.assertTrue(result["is_review_or_perspective"])
        self.assertTrue(result["review_needed"])

    def test_blanks_are_not_evidence(self):
        result = self.assert_topic(None, float("nan"), UNCLASSIFIED)
        self.assertTrue(result["review_needed"])
        self.assertFalse(result["has_experimental_synthesis"])

    def test_inputs_exclude_triage_labels(self):
        self.assertEqual(tuple(inspect.signature(classify_topic).parameters), ("title", "abstract", "document_type"))
        self.assertEqual(CLASSIFIER_VERSION, "3.0.0")

    def test_repeated_group_does_not_multiply_score(self):
        one = classify_topic("New framework", "Adsorption was studied.")
        repeated = classify_topic("New framework", "Adsorption was studied. " * 30)
        self.assertEqual(one["score_functional_materials"], repeated["score_functional_materials"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
