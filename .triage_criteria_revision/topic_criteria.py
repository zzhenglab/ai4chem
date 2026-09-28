"""Fixed, cumulative topic-definition sensitivity analysis (standard library only).

These criteria change the meaning of category boundaries, not triage decisions.
Only bibliographic text and the original text-derived category are inputs. The
criteria were specified without inspecting decision labels or category totals.
Evidence scores are not accuracy estimates; disputed/missing evidence is flagged.
"""

import re

from topic_classifier import (
    CATEGORIES, UNCLASSIFIED, classify_topic, detect_review_article,
    _normalize, _compile, _abstract_sentences, _contribution_signals,
    _FRAMEWORK_SUBJECT, _STRUCTURAL_DETAILS, _DIFFRACTION,
    _GROUPS, _APPLICATION, _PERFORMANCE, _SELF_REPORT,
    _COMPUTATIONAL_TITLE,
)


CRITERIA = (
    "baseline_current", "framework_scope", "property_priority", "computation_inclusive",
)
CRITERION_LABELS = {
    "baseline_current": "Current synthesis-inclusive baseline (v5)",
    "framework_scope": "Framework synthesis scope",
    "property_priority": "Property priority + strict synthesis/structure",
    "computation_inclusive": "Include primary computation in mixed studies",
}
CRITERION_DESCRIPTIONS = {
    "baseline_current": (
        "Current v5 synthesis-inclusive grouping: original affirmative preparation "
        "claims for MOFs/coordination frameworks, networks, or polymers place papers "
        "in Chemical synthesis, including routine preparation within structure and "
        "application studies. Existing primary-topic synthetic-method, molecular, "
        "and post-synthetic assignments are also retained. Reviews belong to Theory "
        "& modeling under the requested taxonomy. The independent primary-topic "
        "audit is preserved in the baseline outputs."
    ),
    "framework_scope": (
        "Starting from the current v5 grouping, distinguish making/modifying a framework from using it "
        "as a catalyst. Explicit framework-catalyzed molecular transformations or "
        "catalyst-performance contributions become Functional materials unless "
        "a framework-making/modification method is foregrounded. Reviews and "
        "primary theoretical studies remain Theory & modeling."
    ),
    "property_priority": (
        "Starting from framework scope, a functional/property endpoint foregrounded "
        "in the title or stated as a substantive study result takes priority over "
        "experimental synthesis/structure. Luminescence and magnetism in the title "
        "count here. Crystal engineering requires original construction together "
        "with new-structure or structural-design evidence; routine XRD alone is "
        "insufficient. Chemical synthesis requires an explicit preparative method "
        "or chemical-modification contribution. Unsupported assignments become "
        "Unclassified for review, rather than being inferred as theoretical. "
        "Explicit non-framework synthetic-method contributions are retained but flagged."
    ),
    "computation_inclusive": (
        "Starting from property priority, an explicit primary computational method "
        "or contribution in the title/abstract takes priority even in mixed "
        "experimental/computational studies. Supporting-only calculations do not "
        "qualify. Reviews always belong to Theory & modeling."
    ),
}

_CRITERIA_CATALYST = _compile([
    r"catalysts?", r"cataly[sz]\w*", r"catalytic\w*", r"catalysis",
])
_CRITERIA_REACTION = _compile([
    r"(?:c h|c c|c n|c o) (?:bond )?(?:activation|functionalization|functionalisation|formation|coupling)",
    r"(?:suzuki|heck|sonogashira|buchwald hartwig|knoevenagel|hantzsch|biginelli|aldol|pechmann)\w*",
    r"cross coupling", r"cycloaddition", r"esterification", r"transesterification",
    r"amidation", r"acetalization", r"acetalisation", r"alkylation", r"arylation",
    r"hydrogenation", r"oxidation", r"cyclic carbonates?", r"organic transformations?",
    r"organic reactions?", r"substrates?", r"catalytic (?:performance|activity)",
])
_CRITERIA_CATALYTIC_RESULT = _compile([
    r"selectivit\w*", r"conversion\w*", r"yield\w*", r"turnover\w*",
    r"recyclab\w*", r"reusab\w*", r"catalytic (?:performance|activity)",
])
_CRITERIA_OWN_RESULT = _compile([
    r"show\w*", r"exhibit\w*", r"demonstrat\w*", r"achiev\w*",
    r"afford\w*", r"deliver\w*", r"display\w*", r"found", r"measur\w*",
    r"observ\w*", r"investigat\w*", r"evaluat\w*", r"studied",
])
_CRITERIA_BACKGROUND = _compile([
    r"previous\w*", r"reported by", r"in the literature",
    r"could be", r"can be", r"may be", r"might be", r"potential applications?",
])
_CRITERIA_PROPERTY = _compile([
    r"luminescen\w*", r"photoluminescen\w*", r"fluorescen\w*", r"phosphorescen\w*",
    r"magnetic propert\w*", r"magnetism", r"magnetocaloric\w*", r"magnetic behavior",
    r"magnetic behaviour", r"spin crossover", r"ferroelectric\w*", r"dielectric\w*",
    r"photophysic\w*", r"optical propert\w*", r"mechanical propert\w*",
    r"electronic propert\w*", r"thermal conduct\w*", r"elastic propert\w*",
])
_CRITERIA_PROPERTY_RESULT = _compile([
    r"quantum yield", r"lifetime\w*", r"emission (?:wavelength|intensity)",
    r"magnetic susceptibility", r"magnetization", r"magnetisation", r"curie temperature",
    r"transition temperature", r"conductivit\w*", r"modulus", r"elastic constant\w*",
])
_CRITERIA_STRUCTURAL_DESIGN = _compile([
    r"topolog\w*", r"reticular\w*", r"isoreticular\w*", r"crystal engineering",
    r"crystal packing", r"supramolecular (?:assembl\w*|architectur\w*|synthons?)",
    r"interpenetrat\w*", r"polymorph\w*", r"structural diversit\w*",
    r"(?:structure|structural) directing", r"network connectivity",
    r"crystal growth", r"single crystal to single crystal", r"single crystal transformation\w*",
])
_CRITERIA_NEW_STRUCTURES = _compile([
    r"(?:new|novel|previously unreported) (?:\w+ ){0,10}(?:coordination polymer\w*|framework\w*|coordination network\w*|complex\w*|co\s?crystal\w*)",
])
_CRITERIA_CREATION_TITLE = _compile([
    r"synthes(?:is|es)", r"construction", r"self assembl\w*", r"assembly",
    r"crystal growth", r"single crystal transformation\w*",
])
_CRITERIA_MODIFICATION = _compile([
    r"post\s?synthetic (?:modification\w*|exchange\w*|transformation\w*|functionalization\w*|functionalisation\w*)",
    r"(?:linker|ligand) exchange", r"transmetalation\w*", r"transmetallation\w*",
])
_CRITERIA_SUPPORTING_TITLE = re.compile(
    r"\b(?:supported|corroborated|confirmed|rationalized|rationalised) "
    r"(?:by|using|with) (?:\w+ ){0,4}(?:dft|density functional|calculations?|simulations?)\b|"
    r"\b(?:supporting|auxiliary) (?:\w+ ){0,3}(?:dft|calculations?|simulations?)\b"
)


def _criteria_snippets(items):
    """Keep quoted source evidence compact and deterministic."""
    return " | ".join(str(item).strip() for item in items[:3] if str(item).strip())


def _criteria_framework_target(title_text):
    """Distinguish synthesis OF a framework from organic synthesis USING one."""
    match = re.search(r"\b(?:synthes(?:is|es)|preparation|fabrication|construction|growth) of (.+)", title_text)
    if not match:
        return False
    target = re.split(r"\b(?:for|using|over|by|via|with|in|on|employing|cataly[sz]\w*)\b", match.group(1), maxsplit=1)[0]
    return bool(_FRAMEWORK_SUBJECT.search(target))


def assign_topic_criteria(title, abstract, document_type="", baseline_category=None):
    """Return four topic assignments from text; no triage outcomes are accepted.

    ``baseline_category`` may reuse the current text-only v5 result. Manual
    overrides should be applied consistently by the caller after this function.
    """
    title_text = _normalize(title)
    abstract_text = _normalize(abstract)
    full_text = title_text + " " + abstract_text
    base = classify_topic(title, abstract, document_type) if baseline_category is None else None
    base_category = base["category"] if base is not None else baseline_category
    if base_category not in CATEGORIES + (UNCLASSIFIED,):
        raise ValueError("baseline_category must be an existing topic category")
    signals = _contribution_signals(title_text, abstract)
    sentences = _abstract_sentences(abstract)
    review, review_source, review_evidence = detect_review_article(title, abstract, document_type)
    if review:
        return [{
            "criterion": criterion, "category": CATEGORIES[1],
            "reason": "Review article belongs to Theory & modeling under the requested taxonomy.",
            "evidence": "{}: {}".format(review_source, review_evidence),
            "review_needed": review_source != "Document Type",
        } for criterion in CRITERIA]

    missing_text = not title_text or not abstract_text
    base_review = bool(base["review_needed"]) if base else (missing_text or base_category == UNCLASSIFIED)
    assignments = [{
        "criterion": CRITERIA[0], "category": base_category,
        "reason": base["primary_topic_reason"] if base else "Current text-derived v5 synthesis-inclusive assignment.",
        "evidence": base["evidence"] if base else "Current v5 evidence is retained in the paper audit.",
        "review_needed": base_review,
    }]

    framework = bool(_FRAMEWORK_SUBJECT.search(full_text))
    title_route = any(pattern.search(title_text) for name, _, _, pattern in _GROUPS[CATEGORIES[0]]
                      if name in ("synthetic routes", "synthesis development", "post-synthetic chemistry"))
    title_molecular = any(pattern.search(title_text) for name, _, _, pattern in _GROUPS[CATEGORIES[0]]
                          if name == "molecular reaction development")
    modification_title = bool(_CRITERIA_MODIFICATION.search(title_text))
    framework_target = _criteria_framework_target(title_text)
    framework_method_focus = framework and (
        modification_title or (title_route and (framework_target or not title_molecular))
        or (signals["abstract_method"] and not title_molecular))

    own_catalysis = [raw for raw, sentence in sentences
                    if _CRITERIA_CATALYST.search(sentence)
                    and (_CRITERIA_REACTION.search(sentence) or _CRITERIA_CATALYTIC_RESULT.search(sentence))
                    and (_SELF_REPORT.search(sentence) or _CRITERIA_OWN_RESULT.search(sentence))
                    and not _CRITERIA_BACKGROUND.search(sentence)]
    title_catalysis = bool(_CRITERIA_CATALYST.search(title_text)
                          and (_CRITERIA_REACTION.search(title_text)
                               or _CRITERIA_CATALYTIC_RESULT.search(title_text)))
    catalyst_application = framework and (title_catalysis or bool(own_catalysis))
    current = dict(assignments[-1], criterion=CRITERIA[1])
    if base_category != CATEGORIES[1] and catalyst_application and not framework_method_focus:
        current.update(
            category=CATEGORIES[3],
            reason="Framework is used as a catalyst for a molecular transformation/performance study; no foregrounded framework-making method.",
            evidence=_criteria_snippets((["Title: " + str(title)] if title_catalysis else []) + own_catalysis),
            review_needed=missing_text or bool(signals["construction_structure"]),
        )
    assignments.append(current)

    title_property = bool(_APPLICATION.search(title_text) or _CRITERIA_PROPERTY.search(title_text))
    substantive_application = [raw for raw, sentence in sentences
                               if _APPLICATION.search(sentence) and _PERFORMANCE.search(sentence)
                               and (_SELF_REPORT.search(sentence) or _CRITERIA_OWN_RESULT.search(sentence))
                               and not _CRITERIA_BACKGROUND.search(sentence)]
    substantive_property = [raw for raw, sentence in sentences
                            if _CRITERIA_PROPERTY.search(sentence)
                            and _CRITERIA_PROPERTY_RESULT.search(sentence)
                            and (_SELF_REPORT.search(sentence) or _CRITERIA_OWN_RESULT.search(sentence))
                            and not _CRITERIA_BACKGROUND.search(sentence)]
    property_focus = title_property or bool(substantive_application or substantive_property)
    new_structure = bool(_CRITERIA_NEW_STRUCTURES.search(title_text)) or any(
        _CRITERIA_NEW_STRUCTURES.search(_normalize(raw)) for raw in signals["synthesis"])
    structural_design = bool(_CRITERIA_STRUCTURAL_DESIGN.search(title_text)) or any(
        _CRITERIA_STRUCTURAL_DESIGN.search(sentence)
        and (_SELF_REPORT.search(sentence) or _CRITERIA_OWN_RESULT.search(sentence))
        and not _CRITERIA_BACKGROUND.search(sentence) for _, sentence in sentences)
    original_construction = bool(signals["synthesis"]) or bool(
        _CRITERIA_CREATION_TITLE.search(title_text) and (new_structure or structural_design))
    structural_analysis = bool(_STRUCTURAL_DETAILS.search(full_text)
                               or _CRITERIA_STRUCTURAL_DESIGN.search(title_text)
                               or (new_structure and _DIFFRACTION.search(abstract_text)))
    strict_crystal = original_construction and structural_analysis and (new_structure or structural_design)
    method_contribution = title_route or modification_title or bool(signals["abstract_method"])
    current = dict(assignments[-1], criterion=CRITERIA[2])
    if current["category"] != CATEGORIES[1] and property_focus:
        current.update(
            category=CATEGORIES[3],
            reason="A functional/property endpoint is foregrounded in the title or supported by an explicit performance/property result.",
            evidence=_criteria_snippets((["Title: " + str(title)] if title_property else [])
                                        + substantive_application + substantive_property),
            review_needed=missing_text or bool(signals["construction_structure"] or framework_method_focus),
        )
    elif current["category"] == CATEGORIES[2]:
        if strict_crystal:
            current.update(
                reason="Original construction plus new-structure/structural-design evidence satisfies the stricter crystal-engineering definition.",
                evidence=_criteria_snippets(["Title: " + str(title)] + signals["synthesis"] + signals["structure"]),
                review_needed=missing_text,
            )
        elif method_contribution:
            current.update(
                category=CATEGORIES[0],
                reason="An explicit preparative/modification contribution is present, but stricter original crystal-design evidence is incomplete.",
                evidence=_criteria_snippets(["Title: " + str(title)] + signals["method"]),
                review_needed=True,
            )
        else:
            current.update(
                category=UNCLASSIFIED,
                reason="Stricter crystal engineering needs original construction plus new-structure/structural-design evidence; available text does not establish both.",
                evidence=_criteria_snippets(["Title: " + str(title)] + signals["structure"]),
                review_needed=True,
            )
    elif current["category"] == CATEGORIES[0]:
        if method_contribution:
            current.update(
                reason=("Explicit framework preparative/modification method contribution."
                        if framework_method_focus else
                        "Explicit preparative/modification method retained; framework scope requires review."),
                evidence=_criteria_snippets(["Title: " + str(title)] + signals["method"]),
                review_needed=missing_text or not framework_method_focus,
            )
        elif strict_crystal:
            current.update(
                category=CATEGORIES[2],
                reason="Original structural construction is explicit; the text does not establish a distinct synthetic-method contribution.",
                evidence=_criteria_snippets(["Title: " + str(title)] + signals["synthesis"] + signals["structure"]),
                review_needed=missing_text,
            )
        elif title_molecular and not catalyst_application:
            current.update(
                reason="Explicit molecular reaction development retained provisionally; scope beyond framework preparation requires review.",
                evidence="Title: " + str(title), review_needed=True,
            )
        else:
            current.update(
                category=UNCLASSIFIED,
                reason="Available text does not establish a preparative-method, chemical-modification, or original structural-design contribution under the stricter definitions.",
                evidence="Title: " + str(title), review_needed=True,
            )
    assignments.append(current)

    primary_computational_title = bool(_COMPUTATIONAL_TITLE.search(title_text)
                                       and not _CRITERIA_SUPPORTING_TITLE.search(title_text))
    current = dict(assignments[-1], criterion=CRITERIA[3])
    if primary_computational_title or signals["computation"]:
        current.update(
            category=CATEGORIES[1],
            reason="Explicit primary computational method/contribution takes priority, including mixed experimental/computational studies.",
            evidence=_criteria_snippets((["Title: " + str(title)] if primary_computational_title else [])
                                        + signals["computation"]),
            review_needed=missing_text or bool(signals["synthesis"] or property_focus),
        )
    assignments.append(current)
    return assignments
