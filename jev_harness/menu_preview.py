"""Render the actual initial menu and its first action page at a recorded family choice."""
import json

from .runtime import NativeNone, Trace, Uncovered
from .semantic_assembly import SemanticAssembly


class PageCaptured(Exception):
    pass


class InitialMenu(SemanticAssembly):
    def decision(self, question, evidence, candidates, skill="", policy=None, node=None, **kwargs):
        options = [c["label"] for c in candidates]+["NONE: no listed option is supported by the evidence"]
        self.pages.append({"node":node,"question":question,"skill":skill,"state":json.dumps(evidence,ensure_ascii=False,default=str),
                           "options":options,"ids":[c["value"] for c in candidates]+[None]})
        if node == "fragment/assembly":
            for choice in candidates:
                if choice["value"] == self.family:
                    return self.family
            raise NativeNone({"probabilities":[0]*len(candidates)+[1]})
        raise PageCaptured


def initial_pages(library, instruction, family=None, *, task=None):
    engine = InitialMenu(None,None)
    engine.trace,engine.pages,engine.family = Trace(),[],family
    engine.context = {"task":{"functions":[], **(task or {}), "instruction":instruction}}
    engine.initialize(library)
    try:
        engine.next_card()
    except (PageCaptured,Uncovered):
        pass
    return engine.pages


def initial_change(parent, candidate, events, functions=()):
    """Compare the opening, holding the recorded family choice fixed when it is still offered."""
    selection = next((e for e in events if e["event"]=="assembly_selection"), None)
    family = selection.get("family") if selection else None
    task = {**events[0], "functions":functions}
    pages = [initial_pages(lib,task["instruction"],family,task=task) for lib in (parent,candidate)]
    def page_at(arm,node):
        return next((p for p in pages[arm] if p["node"]==node),{})
    assembly = [page_at(i,"fragment/assembly") for i in (0,1)]
    steps = [page_at(i,"fragment/step") for i in (0,1)]
    fields = ("question","skill","state","options","ids")
    def changes(pair):
        return [field for field in fields if pair[0].get(field) != pair[1].get(field)]
    family_pages = [[p for p in arm if p["node"]=="fragment/assembly"] for arm in pages]
    family_changes = [field for field in fields if [p[field] for p in family_pages[0]] !=
                      [p[field] for p in family_pages[1]]]
    comparable = selection is not None and all(steps) and (
        family is None or all(any(p["node"]=="fragment/assembly" and family in p["ids"] for p in arm)
                              for arm in pages))
    return {"observed_parent_family":family, "family_menu_rendered":any(assembly),
            "action_menu_comparable":comparable, "changed_fields":{
                "fragment/assembly":family_changes, "fragment/step":changes(steps) if comparable else None}}
