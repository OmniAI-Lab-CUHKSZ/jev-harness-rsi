"""Train-only editing and real model probes for the six-part shared harness."""
from pathlib import Path

from .card_library import CARD_DSL
from .card_workbench import CardWorkbench
from .formal_runtime import BudgetExhausted
from .phase1_optimizer import PrefixProbeMixin, ProbeComplete, Workbench, serialized
from .runtime import is_open_generation, parse_json
from .semantic_assembly import SemanticAssembly
from .semantic_library import patch_library


PROMPT = """You are GPT-6.2 Sol Max, the offline editor of the shared execution harness for one frozen 27B.
Use only supplied Train evidence. Maximize official whole tasks solved, then minimize all online generated
tokens including thinking and failed attempts. Look for repeated G work that Jev and grounded programs
can perform accurately. At runtime the program builds every option. Native Jev can first choose a menu
family, then select an action and bind its parameters. Menu abstention widens to all executable cards.
Read Train raw tool calls, code, results, model scores, assembly_candidates, assembly_selection, card_menu,
card_failed, fragment_handoff, and fragment_reentry. Locate the first observable mistake or costly success.
You can edit six kinds: adapter, card, state, assembly, decision, handoff. Prefer a change at the responsible
layer. Use public observations and reusable small operations; preserve the request's scope and object identities.
Use the existing read/find/probe workbench. Read dsl for exact schemas and library/* or fragment/<name> for the parent.
Use {"probe":{"task":"current Train id","patch":{"objects":[...],"until":{"event":"...","max_g":2}}}}
for a provisional trial. With no until, it stops before G. A targeted probe may follow G to an actual operation
or return to Jev. Read its trace to see actual tool responses. Then return
{reason,prediction,probe_task,until?,objects:[{kind,name,value}]}. At most three named objects per revision.
After the trial, confirm or revise the candidate. Do not encode scenario IDs, known answers or whole workflows."""

DSL = """Every patch edits 1-3 named objects. Read the corresponding library/<key> documents.
adapter / application, in adapters: {login_field:'email'|'phone_number', token_path?:'access_token'}.
adapter / app__api, in adapters: {argument_map?:{local:public_arg},result_path?:'key.path',page_size?:20}.
card / name, in cards: an installed builtin with description/skill/handoff_skill or a single declared API or
calculation. Installed business cards may be replaced by declared operations. Completion belongs to the executor.
state / name, in views: {requires?:[live roles],program:'return ...'}. Result is visible to Jev and G.
assembly / family, in assembly: {description,apps?:[installed apps],cards?:[installed names],program?:'return [...]'}.
A program adds existing card names. The runner resolves missing roles, then Jev chooses among distinct grounded menus.
Every app has a default family and any rejected narrow menu can widen to the full executable set.
decision / step|assembly|quantity|libraries|directory|entity|query|genre|order:
{question?:text,skill?:text,program?:'return [option indices ...]',labels?:{songs|albums|playlists:text}}.
Only libraries takes labels. An optional program selects/reorders real option indices; see library/decision_policies.
handoff / name, in handoffs: {requires?:[roles],when?:'return bool',defer?:'return bool',skill?:text,
apis?:[public endpoint names to prefer],reads?:[installed single-read card names for G-to-Jev reentry]}.
Any program gets one argument called state. state contains request, values (public live data), bindings (producer,
inputs and bound parameters per output), completed, conditions, read_scopes and receipts. Assembly adds cards
(installed names) and eligible. Decision adds options and evidence. These are ordinary restricted Python bodies;
read code/policy_programs. Each object replaces its named definition; decision updates the supplied fields.
Probe until accepts event assembly_selection, card_menu, decision, card_executed, card_failed, adapter_login,
handoff_state, handoff_selection, parameter_bindings, api, api_action, tool_batch or fragment_reentry, plus optional exact card/family/node/app/api/url/method/guide
and max_g (default 2). api_action uses the public URL path. Count full scores only from the official rerun.
"""


class SemanticProbe(PrefixProbeMixin, SemanticAssembly):
    def set_probe_target(self, goal):
        self.goal = None
        self.max_probe_g = 0
        self.probe_g = 0
        self.checked = 0
        if goal is None:
            return
        allowed = {"assembly_selection","card_menu","decision","card_executed","card_failed","adapter_login",
                   "handoff_state","handoff_selection","parameter_bindings","api","api_action","tool_batch","fragment_reentry"}
        if not isinstance(goal, dict) or goal.get("event") not in allowed or set(goal)-{
            "event","card","family","node","app","api","url","method","guide","max_g"}:
            raise ValueError("Expected a supported probe event and optional exact public fields")
        limit = goal.get("max_g", 2)
        if type(limit) is not int or not 0 <= limit <= self.max_calls:
            raise ValueError("max_g must fit the task's model call budget")
        self.goal = {key:value for key,value in goal.items() if key != "max_g"}
        self.max_probe_g = limit

    def check_goal(self):
        if self.goal:
            for event in self.trace.events[self.checked:]:
                if all(event.get(k) == v for k, v in self.goal.items()):
                    raise ProbeComplete("Reached " + self.goal["event"])
            self.checked = len(self.trace.events)

    def _model(self, operation, **values):
        self.check_goal()
        if is_open_generation(operation, values):
            if self.probe_g >= self.max_probe_g:
                self.trace.add("probe_generation_boundary", messages=values["messages"])
                raise ProbeComplete("Generation boundary" if self.goal is None else "Reached probe generation allowance")
            self.probe_g += 1
        try:
            return SemanticAssembly._model(self, operation, **values)
        except BudgetExhausted as exc:
            raise ProbeComplete(str(exc)) from exc

    def tool_request(self, op, event, node, **params):
        self.check_goal()
        try:
            result = super().tool_request(op, event, node, **params)
        except BudgetExhausted as exc:
            raise ProbeComplete(str(exc)) from exc
        self.check_goal()
        return result

    def perform_card(self, name):
        super().perform_card(name)
        self.check_goal()

    def delegate_read(self, identifier, arguments):
        super().delegate_read(identifier, arguments)
        self.check_goal()


def revision_attempt_folder(revision):
    """On restart, keep exhausted dialogues and their usage alongside the fresh attempt."""
    folder = revision/"sol"
    retries = 0
    while (last := folder/"turn_12"/"response.txt").is_file():
        try:
            reply = parse_json(last.read_text(encoding="utf-8"))
        except ValueError:
            reply = None
        if isinstance(reply, dict) and "objects" in reply:
            break
        retries += 1
        folder = revision/f"sol_retry_{retries:02d}"
    return folder


class SemanticWorkbench(CardWorkbench):
    prompt = PROMPT
    install = staticmethod(patch_library)
    probe_type = SemanticProbe
    review_probe = True

    def __init__(self, *args):
        super().__init__(*args)
        settings = self.study.execution
        if isinstance(settings, dict) and settings.get("step_review_threshold") is not None:
            self.prompt += (f"\nAt fragment/step, a final native confidence below {settings['step_review_threshold']} "
                            "can trigger one thinking choice from the same options. A narrow menu abstention widens "
                            "first. See step_review for native and final indices, confidence, outcome and token cost. "
                            "This fixed routing also runs inside your probes. A generation anchor denotes the first "
                            "open task generation. R_review counts toward all model calls and generated tokens.")
            if settings.get("step_review_mode") == "official_adaptive":
                self.prompt += (f" The published AutoTrust service spends up to {settings['step_review_thinking_budget']} "
                                "thinking tokens. Each step_review records the fast distribution native_probabilities, "
                                "the post-thinking distribution reasoned_probabilities, and their equal-weight mean "
                                "probabilities. The highest combined score determines the action. The model event "
                                "contains the reasoning and service usage. Inspect the menu and both rankings when diagnosing choices.")
        card_format = CARD_DSL[CARD_DSL.index("An added card"):CARD_DSL.index("question and skill edit")]
        self.docs["dsl"] = (DSL + card_format + CARD_DSL[CARD_DSL.index("Supported expression operations:"):]).splitlines()
        self.docs["library/questions"].append(serialized({"assembly":self.lib["questions"]["assembly"]}))
        self.docs["library/skills"].append(serialized({"assembly":self.lib["skills"]["assembly"]}))
        for name in ("semantic_assembly","semantic_library","policy_programs"):
            self.docs["code/"+name] = (Path(__file__).parent/(name+".py")).read_text(encoding="utf-8").splitlines()

    def continue_prompt(self, prompt):
        replies = sorted(self.folder.glob("turn_*/response.txt"))
        if replies:
            last = replies[-1]
            action = parse_json(last.read_text(encoding="utf-8"))
            if "objects" in action:
                observed = self.probe(action, action.get("probe_task") or self.index[0]["task"])
                return int(last.parent.name.split("_")[-1])+1, prompt + "\nReview the saved candidate and its Train trial:\n" + serialized([action, observed])
        return super().continue_prompt(prompt)

    def work(self):
        last = self.folder/"turn_12"/"response.txt"
        if last.exists():
            action = parse_json(last.read_text(encoding="utf-8"))
            if "objects" in action:
                self.install(self.lib, action, self.sources, self.study.functions)
                self.probe(action, action.get("probe_task") or self.index[0]["task"])
                return action
        return Workbench.work(self)
