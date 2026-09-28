"""Topic classifier v5: synthesis-inclusive grouping with a v4 topic audit.

The public category is synthesis-inclusive: an original paper with an explicit,
affirmative preparation claim for a MOF/coordination framework, network, or
polymer is Chemical synthesis, including routine preparation in structure and
application papers. Review articles retain Theory & modeling priority. Generic
material preparation, use of an existing MOF catalyst, derivative-only synthesis,
and background, hypothetical, or computational preparation are not such claims.
This is an additive policy; v4 molecular/method/post-synthetic Chemical synthesis
assignments are retained. The v4 primary topic, reason, evidence, scores and
diagnostics are preserved independently of the public grouping. The evidence
rule is specified without reading or fitting any triage Y/N decisions.

The retained v4 primary-topic policy follows:

The four primary topics overlap. Chemical synthesis means developing synthetic
routes, molecular/ligand synthesis, or post-synthetic chemistry. Using a MOF or
other solid as a catalyst to make a molecular product is Functional materials;
the presence of a reaction name alone is not synthesis of the framework.
Crystal engineering means
constructing/understanding coordination structures, networks, and their topology;
new-framework synthesis plus substantial structural analysis belongs here.
Functional materials means application/property performance, including derived
materials. Theory & modeling means computation or theory is the main study.

Phrase groups provide a fallback; contribution rules add evidence from title
grammar and individual abstract sentences. An application studied primarily by
simulation remains Theory & modeling. Supporting calculations alone do not
override an experimental application. Hydrothermal/solvothermal conditions and
generic synthesis are weak evidence. Construction/structure reports with a
brief final luminescence/magnetism measurement can be Crystal engineering.

Title group scores are capped at 20/category and abstract scores at 6/category.
Explicit primary-contribution evidence adds 30 points to the selected topic.
Scores are evidence bookkeeping, NOT probabilities or measured accuracy.
Conflicting/mixed contributions, weak evidence, and missing data are flagged.
If specific rules find no evidence, broad lexical evidence assigns a provisional
topic and records its matched words, confidence, and fallback status. With text
but no lexical evidence, the corpus-domain default is Crystal engineering,
explicitly marked unsupported. Empty title AND abstract remain Unclassified.
The rules are heuristic and need human validation before scientific reporting.
Review articles have priority under the user's taxonomy, regardless of their
covered topic. Document Type metadata and explicit title/abstract genre cues
identify reviews; generic 'recent advances' or 'perspective' alone are not enough.
Only title, abstract, and optional document type are accepted; triage Y/N labels are NEVER inputs and
the method is not fitted to any desired category proportions. Python >= 3.8.
"""

import math
import re
import unicodedata


CATEGORIES = (
    "Chemical synthesis",
    "Theory & modeling",
    "Crystal engineering",
    "Functional materials",
)
UNCLASSIFIED = "Unclassified"
CLASSIFIER_VERSION = "5.0.0"
TAXONOMY_POLICY = "synthesis-inclusive-v1"

_SCORE_KEYS = (
    "score_chemical_synthesis",
    "score_theory_modeling",
    "score_crystal_engineering",
    "score_functional_materials",
)

# Each group is (name, title base weight, abstract weight, regex alternatives).
# Alternatives in the same group cannot accumulate points by repetition.
_GROUP_DEFINITIONS = {
    CATEGORIES[0]: [
        ("synthetic routes", 3.0, 1.7, [
            r"mechanochem\w*",
            r"electrochemical synthes\w*", r"electrosynth\w*",
            r"microwave (?:assisted )?synth\w*", r"sonochemical\w*",
            r"solvent free synthes\w*", r"solventless synthes\w*",
            r"solid state synthes\w*", r"vapor (?:phase )?synth\w*",
            r"vapour (?:phase )?synth\w*", r"flow synthes\w*",
            r"continuous flow", r"room temperature synthes\w*",
            r"ambient (?:condition\w* )?synth\w*", r"ball milling",
            r"microfluidic\w*", r"layer by layer (?:growth|deposition|synthes\w*)",
            r"(?:controlled|controllable|modular|epitaxial|colloid assisted) growth",
            r"(?:emulsion|surfactant|soft|hard) templat\w*",
        ]),
        ("routine synthesis conditions (weak)", 0.45, 0.1, [
            r"solvothermal\w*", r"hydrothermal\w*",
        ]),
        ("synthesis development", 2.6, 1.8, [
            r"synthetic (?:route\w*|method\w*|strateg\w*|protocol\w*|approach\w*)",
            r"(?:scalable|scale up|large scale|high yield|rapid|one pot|one step|green|controlled|controllable) synthes\w*",
            r"(?:synthesis|synthetic) (?:optimization|optimisation|kinetics|mechanism\w*)",
            r"(?:modulated|modulator assisted|template assisted|seed mediated) synthes\w*",
            r"(?:nucleation|formation) mechanism\w*",
            r"(?:design|synthesis) of (?:complex )?organic (?:building blocks?|ligands?)",
        ]),
        ("post-synthetic chemistry", 3.2, 1.9, [
            r"post\s?synthetic(?:ally)? (?:\w+ ){0,2}(?:modification\w*|modified|functionalization\w*|functionalisation\w*|exchange\w*|metalation\w*|metallation\w*)",
            r"(?:linker|ligand) exchange", r"transmetalation\w*",
            r"transmetallation\w*", r"covalent functionalization\w*",
            r"covalent functionalisation\w*",
            r"(?:ligand|linker) (?:synthes\w*|functionalization\w*|functionalisation\w*)",
            r"(?:grafting|grafted) (?:from|onto|on)\b",
        ]),
        ("molecular reaction development", 3.3, 1.8, [
            r"(?:c h|c c|c n|c o) (?:bond )?(?:activation\w*|functionalization\w*|functionalisation\w*|formation|coupling)",
            r"(?:suzuki|heck|sonogashira|buchwald hartwig|knoevenagel|hantzsch|biginelli|aldol|pechmann)\w*(?: \w+){0,2} (?:coupling|reaction|condensation)",
            r"(?:cross coupling|cycloaddition|esterification|transesterification|amidation|acetalization|acetalisation|alkylation|arylation)",
            r"(?:total|asymmetric|enantioselective|stereoselective) synthes\w*",
            r"(?:oxidation|hydrogenation) of (?:\w+ ){0,3}(?:alcohol\w*|aldehyde\w*|alkene\w*|alkyne\w*|amine\w*|nitroarene\w*)",
            r"synthes\w* of (?:\w+ ){0,3}cyclic carbonate\w*",
            r"(?:olefination|annulation|cyanosilylation|dimerization|dimerisation)",
            r"(?:cross dehydrogenative|cross dehydrogenation) coupling",
            r"(?:intramolecular|intermolecular) (?:\w+ ){0,4}rearrangement\w*",
            r"nucleophilic aromatic substitution",
        ]),
        ("generic synthesis report (weak)", 0.35, 0.1, [
            r"synthes(?:is|es)", r"preparation and characterization",
            r"preparation and characterisation",
        ]),
    ],
    CATEGORIES[1]: [
        ("explicit computational focus", 3.8, 2.6, [
            r"(?:computational|theoretical|atomistic|in silico) (?:study|studies|investigation\w*|analysis|screening|prediction\w*|design|approach\w*|insight\w*|model\w*|assessment)",
            r"(?:computer|computational|molecular|atomistic|multiscale|quantum) simulation\w*",
            r"(?:computational|theoretical) chem\w*",
        ]),
        ("quantum electronic calculations", 3.0, 0.55, [
            r"density functional(?: theory)?", r"dft", r"ab initio",
            r"first principles?", r"quantum chemical (?:calculation\w*|model\w*)",
            r"qm mm", r"quantum mechanics and molecular mechanics",
        ]),
        ("molecular and statistical simulations", 3.0, 0.7, [
            r"molecular dynamics", r"monte carlo", r"gcmc",
            r"force field\w*", r"statistical mechanic\w*",
            r"coarse grained (?:simulation\w*|model\w*)",
            r"molecular (?:modeling|modelling)",
        ]),
        ("predictive and mathematical methods", 3.2, 1.8, [
            r"machine learning", r"deep learning", r"neural network\w*",
            r"graph neural", r"data driven", r"mathematical model\w*",
            r"(?:kinetic|thermodynamic|continuum|multiscale) (?:modeling|modelling)",
            r"high throughput computational", r"virtual screening",
        ]),
    ],
    CATEGORIES[2]: [
        ("network topology and assembly", 3.3, 2.0, [
            r"topolog\w*", r"reticular (?:chemistry|design|synthesis)",
            r"isoreticular\w*", r"interpenetrat\w*", r"catenat\w*",
            r"supramolecular (?:assembl\w*|architectur\w*|synthons?|networks?|chemistry)",
            r"(?:directed|hierarchical|coordination driven) self assembl\w*",
            r"(?:network|framework) (?:connectivity|catenation)",
        ]),
        ("crystal packing and structural design", 3.2, 1.7, [
            r"crystal engineering", r"crystal packing", r"crystal structure\w*",
            r"(?:structure|structural) directing", r"supramolecular synthon\w*",
            r"polymorph\w*", r"co\s?crystal\w*", r"packing motif\w*",
            r"structural diversit\w*", r"coordination geometr\w*",
            r"coordination modes?", r"(?:homo)?chiral (?:\w+ ){0,3}(?:framework\w*|coordination polymer\w*)",
            r"(?:secondary|second) building units?", r"(?:framework|pore) engineering",
            r"(?:hydrogen|halogen) bond\w* (?:network\w*|assembl\w*)",
        ]),
        ("crystallization and crystal growth", 3.0, 1.8, [
            r"crystalli[sz]ation", r"crystal growth", r"single crystal to single crystal",
            r"single crystal transformation\w*", r"crystal nucleation",
            r"nucleation and growth", r"crystal habit\w*",
            r"(?:crystalline to amorphous|amorphous amorphous|crystal phase) transition\w*",
        ]),
    ],
    CATEGORIES[3]: [
        ("adsorption, storage, and separation", 3.0, 1.8, [
            r"adsor\w*", r"sorption", r"separat\w*", r"membrane\w*",
            r"(?:gas|hydrogen|methane|carbon dioxide|co2) (?:storage|capture|uptake)",
            r"desalination", r"water harvesting", r"ion exchange",
        ]),
        ("energy conversion and electrochemistry", 3.0, 1.9, [
            r"batter\w*", r"supercapacitor\w*", r"capacitor\w*",
            r"electrode\w*", r"electrolyte\w*", r"fuel cell\w*",
            r"energy (?:storage|conversion)", r"solar cell\w*", r"photovoltaic\w*",
            r"(?:hydrogen|oxygen) (?:evolution|production|generation|reduction)",
            r"water splitting", r"electrocatal\w*",
            r"(?:co2|carbon dioxide) (?:reduction|photoreduction|hydrogenation)",
        ]),
        ("catalytic and environmental performance", 2.5, 1.5, [
            r"photocatal\w*", r"heterogeneous catal\w*",
            r"catalytic (?:performance|activity|application\w*)",
            r"(?:enzyme|protein) immobilization", r"(?:enzyme|protein) immobilisation",
            r"(?:nerve agent\w*|chemical warfare|detoxification)",
            r"(?:pollutant\w*|dye\w*) (?:removal|degradation|adsorption)",
            r"(?:removal|degradation) of (?:\w+ ){0,3}(?:pollutant\w*|dye\w*|antibiotic\w*)",
            r"water (?:purification|remediation|treatment)", r"wastewater",
        ]),
        ("sensing and optical function", 3.0, 1.6, [
            r"sens(?:ing|or\w*)", r"detect(?:ion|ing) of", r"chemosens\w*",
            r"fluorescen\w*", r"luminescen\w*", r"photoluminescen\w*",
            r"phosphorescen\w*", r"light emitting", r"photophysic\w*",
            r"nonlinear optic\w*", r"optical propert\w*",
        ]),
        ("electronic, magnetic, and mechanical function", 3.0, 1.6, [
            r"(?:proton|ion|ionic|electrical|electronic|thermal) conduct\w*",
            r"(?:charge|electron|proton|ion) transport", r"semiconduct\w*",
            r"magnet\w*", r"spin crossover", r"ferroelectric\w*",
            r"piezoelectric\w*", r"dielectric\w*", r"thermoelectric\w*",
            r"mechanical propert\w*", r"elastic propert\w*",
            r"electronic propert\w*", r"electronic device\w*",
            r"thermal expansion", r"thermomechanic\w*",
        ]),
        ("biomedical and delivery applications", 3.0, 1.8, [
            r"drug (?:delivery|release|loading)", r"antimicrobial\w*",
            r"antibacterial\w*", r"anticancer\w*", r"bioimag\w*",
            r"photodynamic therap\w*", r"photothermal therap\w*",
            r"cancer (?:therap\w*|treatment)", r"wound heal\w*",
            r"cellular uptake", r"cytotoxic\w*", r"theranostic\w*",
            r"biomedic\w*", r"biosens\w*", r"biocompatib\w*",
            r"antitumou?r\w*", r"tumou?r (?:therap\w*|treatment|targeting)",
            r"bioactiv\w*", r"biocid\w*", r"biocatal\w*",
        ]),
    ],
}


def _compile(patterns):
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.I)


_GROUPS = {
    category: [(name, title_weight, abstract_weight, _compile(patterns))
               for name, title_weight, abstract_weight, patterns in groups]
    for category, groups in _GROUP_DEFINITIONS.items()
}
_COMPUTATIONAL_TITLE = _compile([
    r"(?:computational|theoretical|atomistic|in silico) (?:study|studies|investigation\w*|analysis|screening|prediction\w*|design|approach\w*|insight\w*|model\w*|assessment)",
    r"(?:computer|computational|molecular|atomistic|multiscale|quantum) simulation\w*",
    r"density functional(?: theory)?", r"dft", r"ab initio",
    r"first principles?", r"molecular dynamics", r"monte carlo", r"gcmc",
    r"machine learning", r"deep learning", r"neural network\w*",
    r"molecular (?:modeling|modelling)", r"data driven",
    r"qm mm", r"quantum mechanics and molecular mechanics",
])
_PURPOSE_MARKER = re.compile(r"\b(?:for|towards?|as)\b")
_EXPERIMENTAL_TITLE = re.compile(r"\bexperimental\w*\b")
_COMPUTATIONAL_METHOD = _compile([
    r"computational\w*", r"theoretical\w*", r"simulat\w*",
    r"density functional(?: theory)?", r"dft", r"ab initio",
    r"first principles?", r"monte carlo", r"gcmc", r"molecular dynamics",
    r"machine learning", r"neural network\w*", r"quantum chemical",
    r"qm mm", r"quantum mechanics and molecular mechanics",
])
_STUDY_ACTION = _compile([
    r"investigat\w*", r"stud(?:y|ies|ied)", r"screen\w*", r"predict\w*",
    r"simulat\w*", r"calculat\w*", r"understanding", r"explor\w*",
    r"examin\w*", r"evaluat\w*", r"assess\w*", r"develop\w*",
])
_SELF_REPORT = _compile([
    r"we", r"herein", r"here", r"(?:this|present) (?:work|study|paper)",
    r"in this (?:article|research)",
])
_PRIMARY_COMPUTATION = _compile([
    r"(?:we|herein we|here we) (?:have )?(?:computationally|theoretically) \w+",
    r"(?:we|this work|this study|present work|present study) (?:\w+ ){0,7}(?:computational|theoretical) (?:study|investigation|understanding|screening|analysis|approach)",
    r"(?:we|this work|this study|present work|present study) (?:\w+ ){0,6}(?:perform\w*|conduct\w*|use\w*|employ\w*) (?:\w+ ){0,7}(?:simulat\w*|density functional|dft|monte carlo|machine learning)",
    r"(?:studied|investigated|examined|evaluated|predicted|screened) (?:\w+ ){0,8}(?:using|by|via|with) (?:\w+ ){0,5}(?:simulat\w*|density functional|dft|monte carlo|ab initio|first principles)",
    r"(?:computational|theoretical) (?:study|investigation|screening|analysis|understanding)",
])
_SUPPORTING_COMPUTATION = _compile([
    r"corroborat\w*", r"rationali[sz]\w*", r"support(?:ing|ed|s)?",
    r"confirm\w*", r"also", r"in addition", r"furthermore", r"moreover",
    r"(?:explain|interpret|elucidate|understand) (?:\w+ ){0,5}(?:observed|experimental|measured)",
    r"(?:consistent|agreement) with (?:\w+ ){0,3}experiment\w*",
])
_SYNTHESIS_VERB = _compile([
    r"synthesi[sz](?:ed|ing)", r"prepared", r"fabricated", r"assembled",
    r"obtained", r"grown", r"constructed",
])
_REPORTED_SYNTHESIS = _compile([
    r"(?:we|herein|here) (?:\w+ ){0,8}(?:synthesi[sz]\w*|prepar\w*|fabricat\w*|assembl\w*|construct\w*)",
    r"(?:has|have|was|were|is|are) (?:\w+ ){0,4}(?:synthesized|synthesised|prepared|fabricated|assembled|constructed)",
    r"(?:new|novel) (?:\w+ ){0,9}(?:synthesized|synthesised|prepared|obtained|constructed)",
])
_FRAMEWORK_SUBJECT = _compile([
    r"coordination (?:polymer\w*|network\w*|framework\w*|compound\w*|complex\w*)",
    r"metal organic (?:framework\w*|network\w*)", r"mofs?",
    r"supramolecular", r"zeolitic imidazolate framework\w*",
])
_STRUCTURAL_TITLE = _compile([
    r"crystal structures?", r"structures?", r"structural (?:diversity|modulation|characterization|characterisation|transformation\w*)",
    r"topolog\w*", r"assembl\w*", r"packing", r"polymorph\w*",
    r"crystal engineering", r"interpenetrat\w*", r"reticular",
])
_STRUCTURAL_DETAILS = _compile([
    r"topolog\w*", r"interpenetrat\w*", r"catenat\w*",
    r"coordination (?:geometr\w*|mode\w*|environment\w*)",
    r"crystal structures?", r"structural (?:diversity|motif\w*|feature\w*)",
    r"(?:1d|2d|3d|one dimensional|two dimensional|three dimensional) (?:\w+ ){0,4}(?:net\w*|chain\w*|layer\w*|framework\w*|structure\w*)",
    r"(?:nodal|connected) (?:\w+ ){0,2}net\w*", r"crystal packing",
    r"(?:metal metal|ag ag|cu cu|au au) (?:interaction\w*|contact\w*|bond\w*)",
])
_DIFFRACTION = _compile([
    r"single crystal (?:x ray|diffraction)", r"x ray (?:single crystal|diffraction|crystallograph\w*)",
])
_NEW_STRUCTURES = _compile([
    r"(?:new|novel) (?:\w+ ){0,10}(?:coordination polymer\w*|framework\w*|coordination network\w*|complex\w*)",
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+) (?:\w+ ){0,4}coordination polymer\w*",
])
# Performance/application evidence excludes generic luminescence or magnetism:
# those measurements frequently supplement a structure-centered report.
_APPLICATION = _compile([
    r"adsor\w*", r"(?:gas|hydrogen|methane|carbon dioxide|co2) (?:storage|capture|uptake|separation)",
    r"membrane\w*", r"desalination", r"water harvesting",
    r"sens(?:ing|or\w*)", r"chemosens\w*", r"detect(?:ion|ing) of",
    r"electrocatal\w*", r"photocatal\w*", r"water splitting",
    r"(?:hydrogen|oxygen) (?:evolution|production|generation|reduction)",
    r"(?:co2|carbon dioxide) (?:reduction|photoreduction|hydrogenation)",
    r"batter\w*", r"supercapacitor\w*", r"electrode\w*", r"fuel cell\w*",
    r"energy (?:storage|conversion)", r"solar cell\w*", r"photovoltaic\w*",
    r"(?:proton|ion|ionic|electrical|electronic|thermal) conduct\w*",
    r"(?:pollutant\w*|dye\w*) (?:removal|degradation)", r"wastewater",
    r"drug (?:delivery|release|loading)", r"cellular uptake", r"cytotoxic\w*",
    r"antibacterial\w*", r"antimicrobial\w*", r"bioimag\w*", r"theranostic\w*",
    r"(?:photodynamic|photothermal|cancer|tumou?r) therap\w*",
    r"thermoelectric\w*", r"biosens\w*", r"chemiresistor\w*",
    r"(?:enzyme|protein) immobili[sz]ation", r"biocatal\w*",
    r"bioactiv\w*", r"biocid\w*", r"detoxification",
    r"(?:nerve agent\w*|chemical warfare)",
    r"(?:mechanical|elastic|electronic) propert\w*",
    r"(?:catalytic|catalysis) (?:performance|activity|application\w*|abilit\w*|behavior|behaviour)",
])
_CATALYSIS = _compile([r"catalys\w*", r"catalyz\w*", r"catalyt\w*", r"catal\w*catalyst\w*"])
_SOLID_CATALYST = _compile([
    r"(?:metal organic|porous|coordination|covalent organic|metalloporphyrinic) (?:\w+ ){0,2}(?:framework\w*|network\w*|polymer\w*)",
    r"mofs?", r"cofs?", r"zeolitic imidazolate framework\w*",
    r"(?:heterogeneous|solid|porous|supported|immobilized|immobilised|polymeric) (?:\w+ ){0,3}catal\w*",
    r"nanocatal\w*", r"nanoparticle\w*", r"nanozyme\w*",
])
_CATALYTIC_OBJECTIVE = _compile([
    r"catalytic (?:performance|activity|application\w*|abilit\w*|behavior|behaviour|efficien\w*)",
    r"(?:as|over|using|with) (?:\w+ ){0,3}catalyst\w*",
    r"cataly[sz](?:ed|es|ing)", r"(?:for|in) (?:\w+ ){0,3}catalysis",
    r"catalysts? for",
])
_FRAMEWORK_REACTION_AGENT = _compile([
    r"(?:over|using|by) (?:a |an |the )?(?:\w+ ){0,3}(?:mofs?|metal organic frameworks?|porous frameworks?)",
])
_POSTSYNTHETIC_MODIFICATION = _compile([
    r"post\s?synthetic(?:ally)? (?:\w+ ){0,2}(?:modification\w*|modified|exchange\w*|transformation\w*|functionalization\w*|functionalisation\w*)",
])
_REPORTED_CATALYST_USE = _compile([
    r"(?:used|applied|employed|utili[sz]ed|tested|evaluated) (?:\w+ ){0,3}(?:as|for) (?:\w+ ){0,3}catal\w*",
    r"catalytic (?:activity|performance|ability|efficiency) (?:\w+ ){0,5}(?:measured|studied|evaluated|demonstrated)",
    r"(?:exhibited|exhibits|displayed|displays|showed|shows) (?:\w+ ){0,3}catalytic (?:activity|performance|ability|efficiency)",
])
_PERFORMANCE = _compile([
    r"performance", r"selectivit\w*", r"capacit\w*", r"efficien\w*",
    r"detection limit", r"sensitivity", r"uptake", r"cytotoxic\w*",
    r"rate", r"cycling", r"conversion", r"yield", r"activity",
])
_METHOD_DEVELOPMENT = _compile([
    r"synthetic (?:route\w*|method\w*|strateg\w*|protocol\w*|approach\w*)",
    r"(?:synthesis|synthetic) (?:optimization|optimisation|kinetics|mechanism\w*)",
    r"(?:optimi[sz]\w*|controll\w*) (?:\w+ ){0,5}synthes\w*",
    r"(?:nucleation|formation|growth) mechanism\w*",
    r"post\s?synthetic(?:ally)? (?:\w+ ){0,2}(?:modification\w*|modified|exchange\w*|transformation\w*|functionalization\w*|functionalisation\w*)",
    r"(?:linker|ligand) exchange",
])
_REVIEW = _compile([
    r"(?:this|the present) review", r"we review", r"review article",
    r"(?:recent|latest) advances", r"perspective", r"mini review",
])

# Last-resort lexical evidence is intentionally broader than the contribution
# rules. It never supplies a +30 contribution bonus, and every use requires
# review. Distinct title matches receive two points, abstract matches one.
# No training examples, paper identifiers, or Y/N labels are consulted.
_FALLBACK_TERMS = {
    CATEGORIES[0]: _compile([
        r"synthesi[sz]\w*", r"synthes\w*", r"prepar\w*", r"fabricat\w*",
        r"functionaliz\w*", r"functionalis\w*", r"grafting", r"etching",
        r"precursor\w*", r"reaction conditions?", r"ligand formation",
        r"formation mechanism\w*", r"printing", r"carbonization", r"carbonisation",
        r"rearrange\w*", r"cycliz\w*", r"cyclis\w*", r"condensation",
    ]),
    CATEGORIES[1]: _compile([
        r"simulat\w*", r"computation\w*", r"theoretic\w*", r"calculat\w*",
        r"modeling", r"modelling", r"quantum", r"statistical mechanic\w*",
    ]),
    CATEGORIES[2]: _compile([
        r"structur\w*", r"crystall\w*", r"coordination", r"network\w*",
        r"chiral\w*", r"assembl\w*", r"topolog\w*", r"porosity",
        r"unit cell", r"space group", r"diffraction", r"building blocks?",
        r"dimensional", r"[123]d", r"entangle\w*", r"framework\w*",
    ]),
    CATEGORIES[3]: _compile([
        r"catal\w*", r"uptake", r"encapsulat\w*", r"immobili[sz]\w*",
        r"capacitance", r"conductiv\w*", r"bioactiv\w*", r"biocid\w*",
        r"sorb\w*", r"adsorb\w*", r"application\w*", r"performance",
        r"sensing", r"detox\w*", r"thermomechanic\w*", r"thermal expansion",
        r"delivery", r"release", r"dechlorination", r"redox", r"capture",
    ]),
}


def _fallback_topic(title_text, abstract_text):
    """Return a provisional assignment and reproducible lexical audit trail."""
    scores = {}
    evidence = []
    for category, pattern in _FALLBACK_TERMS.items():
        title_hits = sorted(set(m.group(0) for m in pattern.finditer(title_text)))
        abstract_hits = sorted(set(m.group(0) for m in pattern.finditer(abstract_text)))
        scores[category] = 2 * len(title_hits) + len(abstract_hits)
        for field, hits in (("title", title_hits), ("abstract", abstract_hits)):
            if hits:
                evidence.append("{} [{}; broad lexical fallback]: {}".format(category, field, ", ".join(hits)))
    # Resolve exact ties by the declared order. These are deliberately NOT
    # merged into the specific phrase-group scores, which remain auditable.
    winner = max(CATEGORIES, key=lambda category: scores[category])
    if not scores[winner]:
        return CATEGORIES[2], "corpus-domain default", "No lexical evidence; provisional Crystal engineering domain default."
    return winner, "broad lexical evidence", " | ".join(evidence)


def _normalize(value):
    """Handle spreadsheet blanks without converting float NaN to evidence."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    if value.strip() in ("nan", "none", "<na>", "nat"):
        return ""
    value = re.sub(r"[-\u2010-\u2015\u2212/]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _abstract_sentences(abstract):
    if not _normalize(abstract):
        return []
    return [(sentence.strip(), _normalize(sentence)) for sentence in
            re.split(r"(?<=[.!?])\s+", str(abstract)) if sentence.strip()]


def _contribution_signals(title_text, abstract):
    """Find stated contributions; never infer theory from missing experiments."""
    sentences = _abstract_sentences(abstract)
    computation = []
    supporting_computation = []
    synthesis = []
    structure = []
    application = []
    method = []
    computation_result = _compile([
        r"(?:calculations?|simulations?) (?:\w+ ){0,3}(?:show\w*|reveal\w*|indicate\w*|predict\w*)",
        r"(?:calculations?|simulations?) (?:were|are|have been) (?:performed|conducted|carried out|used)",
    ])
    background_synthesis = _compile([
        r"previous\w*", r"reported by", r"in the literature",
        r"can be (?:synthesized|prepared)", r"could be (?:synthesized|prepared)",
        r"experimentally (?:reported|synthesized|synthesised) (?:mofs?|frameworks?|structures?|materials?)",
    ])
    for index, (raw, sentence) in enumerate(sentences):
        if _COMPUTATIONAL_METHOD.search(sentence):
            if _SUPPORTING_COMPUTATION.search(sentence):
                supporting_computation.append(raw)
            elif (_PRIMARY_COMPUTATION.search(sentence)
                  or (_SELF_REPORT.search(sentence) and _STUDY_ACTION.search(sentence))
                  or (index < 3 and computation_result.search(sentence))):
                computation.append(raw)
        if _REPORTED_SYNTHESIS.search(sentence) and not background_synthesis.search(sentence):
            synthesis.append(raw)
        if _STRUCTURAL_DETAILS.search(sentence):
            structure.append(raw)
        if _APPLICATION.search(sentence):
            application.append(raw)
        if _METHOD_DEVELOPMENT.search(sentence):
            method.append(raw)

    abstract_text = _normalize(abstract)
    title_synthesis = bool(_compile([r"synthes(?:is|es)", r"construction", r"assembl\w*"]).search(title_text))
    structural_title = bool(_STRUCTURAL_TITLE.search(title_text))
    framework = bool(_FRAMEWORK_SUBJECT.search(title_text + " " + abstract_text))
    new_structure = bool(_NEW_STRUCTURES.search(abstract_text))
    diffraction = bool(_DIFFRACTION.search(abstract_text))
    # XRD alone is characterization, not a structural-design contribution.
    construction_structure = (
        (title_synthesis and structural_title)
        or (bool(synthesis) and framework and
            ((new_structure and (bool(structure) or diffraction))
             or (len(structure) >= 2))))
    title_application = bool(_APPLICATION.search(title_text))
    abstract_application = any(
        _APPLICATION.search(sentence) and _PERFORMANCE.search(sentence)
        and (_SELF_REPORT.search(sentence) or index < 3)
        for index, (_, sentence) in enumerate(sentences))
    # Abstract route development needs a stated study contribution, or several
    # sentences about the route/mechanism; background language alone is weak.
    abstract_method = any(_SELF_REPORT.search(_normalize(raw)) for raw in method) or len(method) >= 2
    return {
        "computation": computation,
        "supporting_computation": supporting_computation,
        "synthesis": synthesis,
        "structure": structure,
        "application": application,
        "method": method,
        "construction_structure": construction_structure,
        "structural_title": structural_title,
        "title_application": title_application,
        "abstract_application": abstract_application,
        "abstract_method": abstract_method,
        "review_article": bool(_REVIEW.search(title_text + " " + abstract_text)
                               or re.search(r"\breview\b", title_text)),
    }


def detect_review_article(title, abstract, document_type=""):
    """Identify review genre from metadata or explicit self-description.

    Discussion of earlier reviews, peer review, or a recent advance alone does
    not establish that the current paper is a review. Keep matched evidence.
    """
    title_text = _normalize(title)
    abstract_text = _normalize(abstract)
    doc_type = _normalize(document_type)
    if re.search(r"\b(?:reviews?|mini\s?review)\b", doc_type):
        return True, "Document Type", str(document_type).strip()
    title_match = re.search(r"\b(?:review|mini\s?review)\b", title_text)
    if title_match:
        return True, "title", str(title).strip()
    genre_patterns = [
        r"\b(?:this|the present|the current|the presented|our) (?:(?:critical|comprehensive|systematic|scoping|mini|short|tutorial|literature|focused|brief|highlight|timely|presented) ){0,3}review\b",
        r"\bwe (?:(?:critically|systematically|comprehensively|briefly|specifically|will) ){0,2}review\b",
        r"\b(?:this|the present|the current) (?:paper|article|work) (?:(?:critically|systematically|comprehensively) )?reviews\b",
        r"\b(?:this|the present|the current) (?:paper|article|work) (?:provides?|presents?|offers?) (?:a |an )?(?:(?:critical|comprehensive|systematic|scoping|tutorial|literature|brief|broad|concise|detailed|up to date) ){0,3}review\b",
        r"\bwe (?:provide|present|offer|make) (?:a |an )?(?:(?:critical|comprehensive|systematic|tutorial|literature|brief|broad|concise|detailed|up to date) ){0,3}review\b",
        r"\b(?:in this|in the present|in the) (?:mini )?review\b",
        r"\b(?:a|an) (?:critical|comprehensive|systematic|literature|tutorial|brief|concise) review\b.{0,140}\b(?:is|has been) (?:presented|provided|given)\b",
        r"\breviewed in (?:this|the present) (?:paper|article|review)\b",
        r"^the review (?:focuses|considers|discusses|summarizes|summarises|covers|examines|deals|is focused)\b",
    ]
    detector = re.compile("|".join(genre_patterns), re.I)
    for raw, sentence in _abstract_sentences(abstract):
        if detector.search(sentence):
            return True, "abstract", raw
    # An overview/survey of literature can indicate review genre, whereas a
    # survey of synthesized samples/solvents is original experimental work.
    overview = re.compile(
        r"\b(?:(?:this|the present) (?:paper|article|study|work) (?:provides|presents|offers)|"
        r"we (?:provide|present|offer)) (?:a |an )?(?:(?:comprehensive|systematic|brief|critical|detailed) )?"
        r"(?:overview|survey)\b.{0,150}\b(?:literature|recent advances|developments|progress|state of the art)\b")
    own_experiments = any(_SELF_REPORT.search(sentence) and _REPORTED_SYNTHESIS.search(sentence)
                          for _, sentence in _abstract_sentences(abstract))
    if not own_experiments:
        for raw, sentence in _abstract_sentences(abstract):
            if overview.search(sentence):
                return True, "abstract literature overview", raw
    return False, "", ""


_PREPARED_FRAMEWORK = (
    r"(?:metal organic|porous coordination|coordination|zeolitic imidazolate) "
    r"(?:frameworks?|networks?|polymers?)|mofs?|"
    r"(?:uio|zif|mil|hkust|mfm|pcn|dut|nu) \d+[a-z]?"
)
_PREPARATION_PAST = (
    r"synthesi[sz]ed|prepared|fabricated|assembled|constructed|obtained|grown|crystalli[sz]ed"
)
_PREPARATION_ACTIVE = (
    r"synthesi[sz](?:e|ed)|prepar(?:e|ed)|fabricat(?:e|ed)|assembl(?:e|ed)|"
    r"construct(?:ed)?|obtain(?:ed)?|grew|grow|crystalli[sz](?:e|ed)"
)
_PREPARATION_NOUN = r"synthes(?:is|es)|preparation|fabrication|construction|assembly|growth"
_NONASSERTED_PREPARATION = re.compile(
    r"\b(?:not|never|no|cannot|can t|couldn t|didn t|without|unsuccessful|failed|"
    r"previously|previous|earlier|already|usually|typically|commonly|generally|"
    r"traditionally|conventionally|hypothetical|hypothetically|proposed|predicted|"
    r"reported by|in the literature|has been reported|have been reported|"
    r"had been|can|could|may|might|would|should|will|aim|aims|attempt|attempts)\b"
)
_DERIVED_OR_DIFFERENT_OBJECT = re.compile(
    r"\b(?:derived|based on|carbon(?:s)?|oxides?|\w*composites?|\w*hybrids?|"
    r"using|via|over|from|cataly[sz]\w*|cataly(?:sis|tic)|"
    r"products?|substrates?|molecules?|carbonates?|amines?|alcohols?|"
    r"data|results?|measurements?|properties|isotherms?|curves?)\b"
)


def _framework_preparation_evidence(title, abstract, primary_result):
    """Find local, affirmative framework-preparation claims, without label inputs.

    Matches require a preparation noun governing the framework object, a
    first-person preparation verb with that object, or a framework subject with
    a passive preparation verb. Bridges are bounded to 18 words/180 characters
    and reject object-changing prepositions and derivative/composite objects.
    Sentences, semicolons, and explicit contrast clauses are independent: a MOF
    mention elsewhere cannot make organic-product synthesis a positive claim.
    Common framework identifiers are accepted as entities. A generic 'framework'
    alone is insufficient, as it can describe a model or a supramolecular solid.
    These conservative lexical checks are evidence flags, not a parser or a
    claim of validated accuracy; ambiguous records retain the v4 assignment.
    """
    if primary_result["is_review_article"]:
        return []
    entity = r"(?P<entity>\b(?:" + _PREPARED_FRAMEWORK + r")\b)"
    bridge = r"(?P<bridge>[^.;:!?]{0,180}?)"
    passive = re.compile(entity + bridge + r"\b(?P<action>" + _PREPARATION_PAST + r")\b")
    active = re.compile(r"\b(?P<action>" + _PREPARATION_ACTIVE + r")\b" + bridge + entity)
    nominal = re.compile(r"\b(?P<action>" + _PREPARATION_NOUN + r")\b" + bridge + entity)
    evidence = []
    denied_preparation = False
    inputs = [("title", str(title).strip(), _normalize(title))] if _normalize(title) else []
    inputs += [("abstract", raw, normalized) for raw, normalized in _abstract_sentences(abstract)]
    for source, raw, normalized in inputs:
        for clause in re.split(r";|\b(?:but|whereas|however)\b", normalized):
            for relation, pattern in (("passive", passive), ("active", active), ("nominal", nominal)):
                for match in pattern.finditer(clause):
                    link = match.group("bridge")
                    if len(re.findall(r"\b\w+\b", link)) > 18:
                        continue
                    # The preparation target must be the framework, rather than
                    # a product made using it or a derivative/composite of it.
                    if _DERIVED_OR_DIFFERENT_OBJECT.search(link):
                        continue
                    entity_suffix = clause[match.end("entity"):]
                    if re.match(r"\s*(?:\([^)]*\)\s*)?"
                                r"(?:(?!(?:was|were|is|are|has|have|and|for|which|that)\b)[a-z0-9]+\s+){0,4}"
                                r"(?:derived|based|\w*composites?|\w*hybrids?)\b", entity_suffix):
                        continue
                    prefix = clause[:match.end()]
                    if _NONASSERTED_PREPARATION.search(prefix):
                        failed_attempt = re.match(
                            r"\s*(?:was|were|is|are|have been|has been) "
                            r"(?:\w+ ){0,2}(?:unsuccessful|failed)\b", clause[match.end():])
                        if source == "abstract" and (failed_attempt or re.search(
                                r"\b(?:not|never|cannot|failed|unsuccessful)\b", prefix)):
                            denied_preparation = True
                        continue
                    suffix = clause[match.end():]
                    if re.match(r"\s*(?:(?:that|which)\s+)?(?:(?:was|were|has been|have been)\s+)?"
                                r"(?:previously|earlier|reported by|in the literature)\b", suffix):
                        continue
                    # An entirely theoretical/simulation preparation statement
                    # does not establish physical framework preparation.
                    if re.search(r"\b(?:in silico|computationally|theoretically|virtually)\b", prefix):
                        continue
                    if relation == "nominal":
                        # 'Synthesis ... using a MOF' is excluded above. Require
                        # the governing 'of' and reject a new object before it.
                        if not re.match(r"\s*(?:(?:and|,|crystal|structures?|structural|"
                                        r"characterization|characterisation|isolation|properties|"
                                        r"luminescence|magnetic|photoluminescent)\s*)*of\b", link):
                            continue
                        if re.search(r"\b(?:in|within|inside|into|on|for|with|by|containing|"
                                     r"confined|supported|embedded)\b", link.replace("in situ", "")):
                            continue
                        if source == "abstract" and not (
                                (_SELF_REPORT.search(clause[:match.start()])
                                 and re.search(r"\b(?:report|reported|present|presented|describe|"
                                               r"described|demonstrate|demonstrated|achieve|achieved)\b",
                                               clause[:match.start()]))
                                or re.search(r"\b(?:is|are|was|were|has been|have been) "
                                             r"(?:reported|described|presented)\b", clause[match.end():])):
                            continue
                    elif relation == "active":
                        if source == "title" or not _SELF_REPORT.search(clause[:match.start()]):
                            continue
                        if re.search(r"\bto\s*$", clause[:match.start()]):
                            continue
                        if re.search(r"\b(?:to|for|by|with|on|into|in)\b", link):
                            continue
                    else:
                        # Past participles preceding the entity belong to the
                        # active pattern; here the framework itself is prepared.
                        if re.search(r"\b(?:of|for|by|with|on|into|in|to|as|used|use|"
                                     r"employed|utilized|utilised)\b", link):
                            continue
                        if re.search(r"\b(?:data|results?|measurements?|properties|isotherms?|curves?)\b"
                                     r"[^.;:!?]{0,80}\b(?:about|of|for|on)\s*$",
                                     clause[:match.start("entity")]):
                            continue
                        if source == "abstract" and not re.search(
                                r"\b(?:was|were|is|are|has|have|been)\b", link):
                            continue
                    # A computational title can mention hypothetical synthesis.
                    # Only an explicit experimental abstract claim overrides it.
                    if source == "title" and (
                            primary_result["title_computational_focus"]
                            or primary_result["abstract_computational_focus"]):
                        continue
                    snippet = "{}: {}".format(source, raw)
                    if snippet not in evidence:
                        evidence.append(snippet)
    if denied_preparation and not any(item.startswith("abstract:") for item in evidence):
        # An unqualified title does not overcome an abstract reporting only a
        # failed/negated preparation. Preserve the primary-topic audit instead.
        evidence = []
    return evidence


def classify_topic(title, abstract, document_type=""):
    """Return one topic plus evidence, scores, and explicit review diagnostics.

    Inputs may be strings or spreadsheet blanks. Output contains only standard
    Python scalar values and can be expanded directly into a dataframe.
    """
    title_text = _normalize(title)
    abstract_text = _normalize(abstract)
    is_review_article, review_source, review_evidence = detect_review_article(title, abstract, document_type)
    scores = {}
    title_scores = {}
    evidence_parts = []
    category_evidence = {category: [] for category in CATEGORIES}

    for category in CATEGORIES:
        title_total = 0.0
        abstract_total = 0.0
        for group_name, title_weight, abstract_weight, pattern in _GROUPS[category]:
            for field, text_value in (("title", title_text), ("abstract", abstract_text)):
                # Retain at most three distinct phrases for an audit-friendly
                # evidence column; points still accrue only once per group.
                matches = list(dict.fromkeys(match.group(0) for match in pattern.finditer(text_value)))
                if not matches:
                    continue
                if field == "title":
                    title_total += 4.0 * title_weight
                else:
                    abstract_total += abstract_weight
                entry = "{} [{}; {}]: {}".format(category, field, group_name, ", ".join(matches[:3]))
                evidence_parts.append(entry)
                category_evidence[category].append(entry)
        title_scores[category] = min(20.0, title_total)
        scores[category] = title_scores[category] + min(6.0, abstract_total)

    signals = _contribution_signals(title_text, abstract)
    computational_focus = bool(_COMPUTATIONAL_TITLE.search(title_text))
    purpose_match = _PURPOSE_MARKER.search(title_text)
    application_purpose = bool(purpose_match and
                               _APPLICATION.search(title_text[purpose_match.end():]))
    molecular_reaction_title = any(pattern.search(title_text) for name, _, _, pattern
                                  in _GROUPS[CATEGORIES[0]] if name == "molecular reaction development")
    title_material_catalysis = bool(
        _SOLID_CATALYST.search(title_text)
        and ((_CATALYSIS.search(title_text) and
              (_CATALYTIC_OBJECTIVE.search(title_text) or molecular_reaction_title))
             or (molecular_reaction_title and _FRAMEWORK_REACTION_AGENT.search(title_text))))
    # A named organic reaction performed over a solid/MOF catalyst is an
    # application, even when the title calls the product-making step synthesis.
    # Require catalyst and solid-material evidence in the SAME abstract sentence
    # to avoid linking unrelated background mentions across the abstract.
    abstract_material_catalysis = any(
        _SOLID_CATALYST.search(sentence) and _CATALYSIS.search(sentence)
        and (_CATALYTIC_OBJECTIVE.search(sentence) or _PERFORMANCE.search(sentence))
        and (_SELF_REPORT.search(sentence) or _REPORTED_CATALYST_USE.search(sentence))
        and not re.search(r"\b(?:can|could|may|might|previously|reported by|in the literature)\b", sentence)
        for _, sentence in _abstract_sentences(abstract))
    purpose_text = title_text[purpose_match.end():] if purpose_match else ""
    purpose_is_modification = bool(_POSTSYNTHETIC_MODIFICATION.search(purpose_text))
    external_catalytic_purpose = bool(purpose_text and not purpose_is_modification and (
        _CATALYTIC_OBJECTIVE.search(purpose_text)
        or (_CATALYSIS.search(purpose_text) and _SOLID_CATALYST.search(title_text))
        or any(pattern.search(purpose_text) for name, _, _, pattern in _GROUPS[CATEGORIES[0]]
               if name == "molecular reaction development")))
    postsynthetic_framework_modification = bool(
        _POSTSYNTHETIC_MODIFICATION.search(title_text)
        and _FRAMEWORK_SUBJECT.search(title_text + " " + abstract_text)
        and not application_purpose and not external_catalytic_purpose)
    # Named groups avoid assigning routine hydrothermal conditions the same
    # status as reaction development or post-synthetic transformations.
    synthesis_focus_title = any(
        pattern.search(title_text)
        for name, _, _, pattern in _GROUPS[CATEGORIES[0]]
        if name in ("synthetic routes", "synthesis development",
                    "post-synthetic chemistry", "molecular reaction development"))
    experimental_title = bool(_EXPERIMENTAL_TITLE.search(title_text))
    computation_evidence = computational_focus or bool(signals["computation"])
    mixed_work = bool(computation_evidence and (experimental_title or signals["synthesis"]))
    focus_category = None
    focus_reason = ""

    if is_review_article:
        focus_category = CATEGORIES[1]
        focus_reason = ("User taxonomy: review articles belong to Theory & modeling, "
                        "regardless of the topic reviewed. Review identified from " + review_source + ".")
    elif computation_evidence and not mixed_work:
        focus_category = CATEGORIES[1]
        focus_reason = ("Title explicitly frames a computational/theoretical study."
                        if computational_focus else
                        "Abstract explicitly states a primary computational/theoretical investigation.")
    elif postsynthetic_framework_modification:
        focus_category = CATEGORIES[0]
        focus_reason = ("Title explicitly modifies the framework/linker post-synthetically; "
                        "a catalyst descriptor alone does not establish an external catalytic application.")
    elif signals["title_application"] or title_material_catalysis or (
            molecular_reaction_title and abstract_material_catalysis):
        focus_category = CATEGORIES[3]
        focus_reason = ("A framework or solid catalyst is used for a molecular reaction: catalytic material application."
                        if title_material_catalysis or (molecular_reaction_title and abstract_material_catalysis)
                        else "Title foregrounds an application or material performance objective.")
    elif (synthesis_focus_title and (molecular_reaction_title or _METHOD_DEVELOPMENT.search(title_text))):
        focus_category = CATEGORIES[0]
        focus_reason = "Title foregrounds synthetic-method development or a chemical transformation."
    elif (signals["abstract_method"] and not signals["structural_title"]
          and not signals["abstract_application"]):
        focus_category = CATEGORIES[0]
        focus_reason = "Abstract foregrounds synthetic route development or a post-synthetic transformation."
    elif signals["construction_structure"] and not molecular_reaction_title:
        focus_category = CATEGORIES[2]
        focus_reason = ("Synthesis/construction is coupled to structural characterization or network design; "
                        "brief supporting property measurements do not set the primary topic.")
    elif synthesis_focus_title:
        focus_category = CATEGORIES[0]
        focus_reason = "Title foregrounds a synthetic route, post-synthetic transformation, or molecular reaction."
    elif signals["abstract_method"] and not signals["abstract_application"]:
        focus_category = CATEGORIES[0]
        focus_reason = "Abstract foregrounds synthetic route development or a post-synthetic transformation."
    elif signals["abstract_application"]:
        focus_category = CATEGORIES[3]
        focus_reason = "Abstract links a stated application to performance or uptake results."
    elif computation_evidence:
        focus_category = CATEGORIES[1]
        focus_reason = "Explicit computational contribution in mixed work; no clearer alternative primary focus."

    if focus_category:
        scores[focus_category] += 30.0
        evidence_parts.append("{} [primary contribution rule; +30]: {}".format(focus_category, focus_reason))

    # Sorting is stable: exact ties follow the declared category order.
    ranked = sorted(CATEGORIES, key=lambda category: scores[category], reverse=True)
    winner, runner_up = ranked[:2]
    best_score, next_score = scores[winner], scores[runner_up]
    margin = best_score - next_score
    reasons = []
    fallback_used = False
    fallback_basis = ""
    if best_score == 0:
        if title_text or abstract_text:
            category, fallback_basis, fallback_evidence = _fallback_topic(title_text, abstract_text)
            fallback_used = True
            evidence_parts.append(fallback_evidence)
            focus_reason = "Provisional {} assignment from {}; human validation required.".format(category, fallback_basis)
            reasons.append("Low-confidence fallback: no specific topic evidence; " + fallback_basis)
        else:
            category = UNCLASSIFIED
            reasons.append("No topic-specific phrase evidence")
    else:
        category = winner
        if best_score < 4.0:
            reasons.append("Weak topic evidence (score below 4)")
        if margin < 3.0 or (next_score >= 4.0 and best_score < 1.35 * next_score):
            reasons.append("Competing categories: {} / {}".format(winner, runner_up))
        if title_scores[winner] == 0:
            reasons.append("Assigned category is supported only by abstract evidence")
        if mixed_work:
            reasons.append("Mixed computational and experimental/synthetic contributions")
        if signals["review_article"]:
            reasons.append("Review/perspective article; topic is coverage rather than a single original contribution")
        # Even a numerical winner can mask explicitly overlapping title topics.
        substantial_title_categories = [name for name in CATEGORIES if title_scores[name] >= 10.0]
        if len(substantial_title_categories) > 1:
            reasons.append("Multiple substantial topics in title: " + " / ".join(substantial_title_categories))
        if signals["construction_structure"] and synthesis_focus_title:
            reasons.append("Structural construction and synthetic-method contributions overlap")
        if signals["construction_structure"] and signals["title_application"]:
            reasons.append("Structural construction and application contributions overlap")
        if _POSTSYNTHETIC_MODIFICATION.search(title_text) and _CATALYSIS.search(title_text):
            reasons.append("Post-synthetic modification and catalysis both appear in title; confirm primary contribution")
    if is_review_article:
        # Topic overlap is expected for a review and does not change this rule.
        reasons = ([] if review_source == "Document Type" else
                   ["Review-article genre inferred from title/abstract; confirm if needed"])
    if not title_text:
        reasons.append("Missing title")
    if title_text and not abstract_text:
        reasons.append("Missing abstract; contribution inferred from title only")
    if not title_text and not abstract_text:
        reasons.append("Missing title and abstract")

    result = {
        "category": category,
        "review_needed": bool(reasons),
        "low_confidence": bool(reasons),
        "fallback_used": fallback_used,
        "fallback_basis": fallback_basis,
        "review_reason": "; ".join(reasons),
        "evidence": " | ".join(evidence_parts),
        "runner_up": runner_up if next_score > 0 else "",
        "score_margin": round(margin, 2),
        "title_computational_focus": computational_focus,
        "title_application_purpose": application_purpose,
        "primary_topic_reason": focus_reason or (
            "No specific topic evidence." if category == UNCLASSIFIED else
            "Phrase-group evidence favors {}; no explicit primary-contribution rule applies.".format(category)),
        "abstract_computational_focus": bool(signals["computation"]),
        "primary_computation_evidence": " | ".join(signals["computation"][:3]),
        "supporting_computation_evidence": " | ".join(signals["supporting_computation"][:3]),
        "has_experimental_synthesis": bool(signals["synthesis"]),
        "experimental_synthesis_evidence": " | ".join(signals["synthesis"][:3]),
        "structural_construction_evidence": " | ".join(signals["structure"][:3]),
        "mixed_experimental_computational": mixed_work,
        "is_review_or_perspective": signals["review_article"] or is_review_article,
        "is_review_article": is_review_article,
        "review_policy_applied": is_review_article,
        "review_detection_source": review_source,
        "review_detection_evidence": review_evidence,
        "classifier_version": CLASSIFIER_VERSION,
    }
    for name, key in zip(CATEGORIES, _SCORE_KEYS):
        result[key] = round(scores[name], 2)
    # Keep all v4 diagnostics untouched. The public category is a separate,
    # explicitly documented grouping; its counts must never feed this rule.
    result["primary_topic_category"] = result["category"]
    result["primary_topic_evidence"] = result["evidence"]
    framework_evidence = _framework_preparation_evidence(title, abstract, result)
    result["has_framework_synthesis"] = bool(framework_evidence)
    result["framework_synthesis_evidence"] = " | ".join(framework_evidence)
    result["taxonomy_policy"] = TAXONOMY_POLICY
    result["taxonomy_reassigned"] = bool(framework_evidence and category != CATEGORIES[0])
    result["reassignment_reason"] = ""
    result["classification_reason"] = result["primary_topic_reason"]
    if framework_evidence:
        result["category"] = CATEGORIES[0]
        result["classification_reason"] = (
            "Synthesis-inclusive taxonomy: this original paper explicitly reports "
            "preparation of a MOF/coordination framework, network, or polymer; "
            "routine preparation takes precedence over structural or application focus.")
        if result["taxonomy_reassigned"]:
            result["reassignment_reason"] = (
                "Grouped as Chemical synthesis by the synthesis-inclusive policy; "
                "the retained v4 primary topic is {}.".format(category))
    return result


def _run_sanity_checks():
    """Critical topic distinctions and invariants, not an accuracy benchmark."""
    cases = [
        ("Mechanochemical synthesis of metal organic frameworks", "", CATEGORIES[0]),
        ("Post-synthetic modification of zirconium frameworks", "", CATEGORIES[0]),
        ("Topological design of interpenetrated coordination networks", "", CATEGORIES[2]),
        ("Crystal packing and polymorphism in coordination networks", "", CATEGORIES[2]),
        ("Facile synthesis of a metal organic framework for hydrogen storage", "", CATEGORIES[3]),
        ("Density functional theory study of carbon dioxide adsorption in a metal organic framework", "", CATEGORIES[1]),
        ("Machine learning for gas adsorption and separation in MOFs", "", CATEGORIES[1]),
        ("A fluorescent framework for sensing aqueous pollutants", "DFT calculations explain the observed response.", CATEGORIES[3]),
        ("C-H functionalization by a porous framework", "", CATEGORIES[3]),
        ("Photocatalytic hydrogen production by a porous framework", "", CATEGORIES[3]),
        ("New metal organic frameworks", "Materials were characterized by single-crystal X-ray diffraction.", CATEGORIES[2]),
        ("A new material and model", "", CATEGORIES[2]),
        (None, float("nan"), UNCLASSIFIED),
    ]
    for title, abstract, expected in cases:
        result = classify_topic(title, abstract)
        assert result["primary_topic_category"] == expected, (title, expected, result)

    # Repeated abstract language cannot multiply a phrase-group score.
    once = classify_topic("New framework", "Adsorption was studied.")
    repeated = classify_topic("New framework", "Adsorption was studied. " * 30)
    assert once["score_functional_materials"] == repeated["score_functional_materials"]
    assert classify_topic("", "") ["review_needed"]
    assert classify_topic("Crystal growth and solvothermal synthesis", "")["review_needed"]
    assert classify_topic("Experimental and computational study of adsorption", "")["review_needed"]
    print("Passed {} classification cases and 4 review/invariance checks.".format(len(cases)))


if __name__ == "__main__":
    _run_sanity_checks()
