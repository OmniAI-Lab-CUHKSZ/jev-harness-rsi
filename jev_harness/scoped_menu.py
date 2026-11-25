"""Render each fragment at the point where its information is useful."""
from copy import deepcopy

from .card_library import ports
from .fragment_assembly import option


def scoped_library(library):
    result = deepcopy(library)
    result["menu_format"] = "scoped"
    return result


def is_scoped(library):
    return library.get("menu_format") == "scoped"


def retrieval_library(library):
    from .retrieval import MODE
    result = scoped_library(library)
    result["menu_retrieval"] = MODE
    result["skills"]["assembly"] = result["skills"]["assembly"].replace(
        "NONE asks the program to show all executable operations.",
        "Choose NONE if this set does not cover the next step. The program can retrieve other relevant operations.")
    return result


def specific_use(card):
    if card.get("applies_to"):
        return card["applies_to"]
    # Older goal-specific reductions have no explicit applicability annotation.
    if "compute" in card and card.get("output") == "answer" and card.get("parameters"):
        return card["description"]
    return None


def card_label(name, card):
    _, inputs, output, _ = ports(card)
    use = specific_use(card)
    text = card["description"]
    if use:
        text = "Specialized: " + text
        if use != card["description"]:
            text += " For: " + use
    if card.get("scope"):
        text += " Scope: " + card["scope"]
    if inputs:
        text += " From " + ", ".join(inputs) + "."
    return option(text + " Produces " + output + ".", name)


def result_sources(library, public, provenance):
    sources = {}
    for role, info in provenance.items():
        if role not in public:
            continue
        name = info["producer"]
        card = library["cards"][name]
        _, _, _, loads = ports(card)
        source = {"producer":name}
        if card.get("api"):
            source["api"] = card["api"]
        coverage = card.get("scope") or loads
        if coverage:
            source["scope"] = coverage
        if info.get("parameters"):
            source["parameters"] = info["parameters"]
        if info.get("source_scope"):
            source["observed_scope"] = info["source_scope"]
        sources[role] = source
    return sources


def consumer_tips(library, public, provenance):
    """Only the actual live writer can describe how to consume a role."""
    return {role:library["cards"][source["producer"]]["handoff_skill"]
            for role,source in provenance.items() if role in public
            and library["cards"].get(source["producer"],{}).get("handoff_skill")}
