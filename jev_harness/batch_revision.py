"""One shared harness revision from a whole Train batch and its observed regressions."""
from collections import Counter
from functools import cached_property
from pathlib import Path

from .card_library import CARD_DSL
from .contrast_probes import FRESH_DECISION_LIMIT, OPEN_GENERATION_LIMIT, PROBE_OUTPUT_LIMIT
from .formal_optimizer import cleaned_text, read_events
from .history_impact import trace_sites
from .history_workbench import SCHEMA as HANDOFF_SCHEMA
from .operation_evidence import batch_operations
from .revision_evidence import (
    EDIT_SURFACES, batch_patterns, edit_history, paired_outcome_costs, request_context, trace_incidents,
)
from .semantic_library import patch_library
from .semantic_workbench import DSL
from .train_evidence import TrainEvidence, serialized
from .train_precedents import TrainPrecedents
from .runtime import is_open_generation, parse_json
from .solved_train_preview import MODE as SOLVED_FIRST_CHANGE


OBJECTIVE = """Optimize the shared harness for the fixed 27B: maximize whole tasks solved first, then minimize
all online output tokens including adaptive reasoning and retries. Look for repeated open generation that
grounded Jev decisions and executable fragments can take over on more than one goal. Trace effects on solved paths."""

INSTRUCTIONS = OBJECTIVE + """
Revise the harness offline. The program renders menus.

Localize. Read the batch results and patterns. Open the focal event and a contrasting success or near miss.
Trace available roles -> eligible -> offered -> selected and bound -> result -> consumer -> actual public effect.
Use the earliest intervention that could change the costly or incorrect outcome, even when an earlier harmless
detour is visually conspicuous. Several variants of one scenario count as one context for transfer.

At the current fault, use {"related_train":{"task":"<current ID>","event":index}} to locate earlier Train
paths with the same intermediate operation, bindings or public effect. The packet has starting examples.
Read the cited old events and their consumers. Compare what actually worked there, how much open generation
the successful path paid for, and which differences in scope or goal the shared repair must respect.
A cell's output tokens cover the entire cell. A call in source may never have run; a solved task can contain
a failed step. Repeated failures can expose a common defect. The Train catalogue includes the goals of older
recordings if another API or representation expresses the same need. A weak match can be discarded.
With no relevant past evidence, repair the present fault directly: add or edit a fragment, guide or any other
appropriate component. A historical example is optional. Keep task-specific details out of the runtime.

Choose the component whose hook really controls that behavior. The six editable surfaces describe scope and
counterevidence. Compare your best explanation with its strongest alternative at the SAME task and state.
For example a direct generated Python call bypasses compiled adapters; a decision skill cannot select a missing
option. A compiled binding fault can be repaired even when the task ultimately fails in generation.
Choose by expected recovered tasks, evidence on different goals, exposure of solved paths, then token cost.
History frequencies are descriptive. Use any subset, with no quota for diversity and no default preference for
new cards or longer handoff instructions. Modify reusable interfaces at their point of use. Check installed
operations first. Ground on public state, actor, scope and effect. Keep task IDs, answer literals and example
identities out of runtime edits. Keep ordinary fallback for uncovered cases.

Read with {"read":[{"doc":"current/<id>","start":0,"count":4}]} or {"find":{"doc":"...","text":"..."}}.
incidents/<id>, full current and historic Train traces, component sources, dsl and api/<name> are available.
Read owners/<id> at the original event index to see which hook actually supplied those arguments, menu or data.
Read dsl or the parent definition when you need an edit's exact syntax. Read raw traces when the observations
do not explain the cause. An event is evidence of exposure, not proof that a component caused a wrong answer.

Submit at most three named objects addressing one focal problem. Suggest another solved task likely to use
that component. A solved sibling that depends on the edited interface can be the most relevant case.
The program compares all current paths, gives priority to a changed solved path that exercised your edit,
and keeps that same preservation case for both patches. Different goals are useful when exposure is comparable.
When the repair interprets wording, include wording_check:{scope:"whole_task"|"same_operation",request:text,alternate?:text}.
Use a syntactically different equivalent instruction, or a different downstream goal that needs the very same
intermediate result. Preserve the objects, scope, actor and period of the tested operation. Candidate builders
can be run on both wordings at the same public state. If a filter uses terms in the request to remove public
records, also describe the same operation in ordinary alternative words in alternate. Preserve exact names when
the user's intent requires them. Check whether the previously bound value still reaches Jev on each wording.
Use structural scope for eligibility. Wording cues can prioritize candidates with the existing slot's prefer
expression. This keeps other eligible public records reachable without adding a list of aliases for one example.
Return {focus:{task,event,observable}, preserve_task:ID or null, wording_check?:object,
diagnosis:{primary:component,alternative:component,why_primary:text,distinguishing_observation:text},
reason,prediction,
history_use:[{trace:earlier Train document,event:original index,shared_step:text,scope_difference:text}],
batch_effects:[{tasks:[current task IDs],effect:"expected change or behavior to preserve"}],
objects:[{kind,name,value}]}. Cite the one or two earlier events that informed the edit, or use [] for a direct repair.
Cover all current tasks once. The program probes the focal task naturally from
the start, including open generation when reached, and the preservation task at the same library revision.
Results include actual accesses to the output and tool observations. All cases and original parent stay fixed
if you later revise. You can keep the parent with {keep_parent:true,reason:"..."} if no repair is warranted."""

REVIEW_CRITERIA = OBJECTIVE + """
Use the focal task, a solved neighbor and the batch-wide changes. A prefix stopped by its budget leaves the
unreached outcome unknown. Distinguish that from executing a wrong action or disrupting a solved path.
Use the one available full-batch evaluation to resolve a plausible gain when local evidence leaves the final
effect open. Keeping the parent is appropriate when the tested repair cannot fix the observed mechanism,
carries a credible regression, or offers no supported improvement. Report a measured move to Jev separately
from an improvement delivered by better open generation."""

REVIEW = REVIEW_CRITERIA + """
Choose whether to score this patch on the full Train batch. Inspect the actual first divergence
on current and historical Train paths, each named edit's contribution, the focal consumer and the preserved
case. Exact initial renderings, historical exposure, conditional observations and official whole-task outcomes are
different measurements. An executed reader alone establishes neither the consumer nor the task outcome.
For any cited historical path, check its actual replay and the first change. Explain which old generated work
the edit could replace or which old capability it preserves. A replay cannot measure the new whole-task cost.
focus_history also compares earlier cases at the exact event YOU chose, including cases absent from the draft.
If one exposes a shared fix or a conflicting scope, use it in the revision. A direct repair can stand when the
old behavior has a different purpose.
You can read/find the trial documents, including complete public probe traces and all archived exposure details.
Reply {use:"draft"|"parent",reason}, or {use:"revise",reason,prediction,diagnosis,batch_effects,history_use,objects:[...]}
for a specific repair to the SAME focal problem. Use the original task and preservation case. Give the
revised patch its own diagnosis and expected effects across this batch, especially if you switch the
component or drop an operation. If the draft has no wording_check, you may supply one for a revised binding. Revised objects
replace the draft against the original parent. Its observations are collected within the remaining budget.
An unresolved past trace may still have a comparable opening menu, conditional on its recorded family choice.
Use this only for those menus. If an outcome was not observed, say so."""

FINAL_REVIEW = REVIEW_CRITERIA + '\nReturn {use:"draft"|"revised"|"parent",reason}. Compare each version using its own measured evidence.'

SOLVED_REVIEW = """
Read historical_choice for each tested patch. The program checks one successful earlier Train recording per
distinct task and stops at its first difference. Compare each listed choice change with the old goal and
grounded action. Consider changed tool arguments, handoffs and generation inputs as possible interference.
An equal choice establishes only that page's result. At a comparable recorded decision, a drifted parent
menu still receives a check of the candidate against the successful recording's bound action. Inspect
recording_comparison.candidate_menu_matches_parent: when true the tested menu was already in the parent.
When false, a differing choice identifies a risk on the candidate menu; the live parent choice may be unknown.
An earlier incompatible action, missing grounding or missing reply leaves the subsequent outcome unknown.
Use read/find on the details document for any task, including a task folded into the counts. Use this
evidence to decide whether to retain, revise or submit the patch. Whole-task outcomes require execution."""


def revised_patch(draft, review):
    # The two interventions share their cases, not their predictions.
    patch = {k:review[k] for k in ("reason","prediction","objects","diagnosis","batch_effects","wording_check") if k in review}
    patch.update({k:draft[k] for k in ("focus","preserve_task","wording_check") if k in draft})
    if "diagnosis" in draft:
        patch["initial_diagnosis"] = draft["diagnosis"]
    if "history_use" in review or "history_use" in draft:
        patch["history_use"] = review.get("history_use", draft.get("history_use"))
    return patch


def validate_batch_effects(groups, tasks):
    if not isinstance(groups,list) or not all(isinstance(group,dict) and isinstance(group.get("tasks"),list)
            and group["tasks"] and all(isinstance(t,str) for t in group["tasks"])
            and isinstance(group.get("effect"),str) and group["effect"].strip() for group in groups):
        raise ValueError('batch_effects needs groups such as {"tasks":["<current task ID>"],"effect":"<expected behavior>"}. Cover each current task once.')
    counts = Counter(task for group in groups for task in group["tasks"])
    if counts != Counter(list(tasks)):
        raise ValueError(f"Cover each current task once. Missing: {sorted(set(tasks)-counts.keys())}; "
                         f"unexpected or repeated: {sorted(t for t,n in counts.items() if t not in tasks or n>1)}")


def task_summary(item, events):
    errors, endpoints = [], Counter()
    fallback = Counter(calls=0, truncated=0, parse_errors=0, executed_cells=0, cells_with_errors=0)
    for i,event in enumerate(events):
        kind = event["event"]
        message = None
        if kind == "model" and is_open_generation(event["operation"],event["request"]):
            fallback["calls"] += 1
            fallback["truncated"] += int(bool(event["response"].get("truncated")))
        elif kind == "generation_parse_error":
            fallback["parse_errors"] += 1
        elif kind == "tool_batch" and event.get("node") == "generation_agent":
            fallback["executed_cells"] += 1
            fallback["cells_with_errors"] += int("Execution failed." in event["response"].get("output",""))
        if kind in ("card_failed","execution_error","program_error","generation_parse_error"):
            message = event.get("reason") or event.get("error") or kind
        elif kind == "tool_batch":
            result = event["response"].get("output","")
            if "Execution failed." in result:
                message = result
        elif kind == "api_action":
            endpoints[(event["method"],event["url"].partition("?")[0])] += 1
        if message:
            errors.append({"event":i,"kind":kind,"message":cleaned_text(message)[-220:]})
    row = {k:item[k] for k in ("task","instruction","success","output_tokens","executed_cards","handoffs","modes")}
    row["errors"] = errors[:3]
    row["open_generation"] = dict(fallback)
    row["public_endpoint_examples"] = [f"{method} {path}" for (method,path),_ in endpoints.most_common(6)]
    if "last_measured_round" in item:
        row["last_measured_round"] = item["last_measured_round"]
    return row


class BatchRevisionWorkbench(TrainEvidence):
    install = staticmethod(patch_library)

    def __init__(self,*args):
        super().__init__(*args)
        self.current_events = {task:read_events(self.paths["current/"+task]) for task in sorted(self.tasks)}
        for trace in getattr(self.study, "prior_train_entries", ()):
            path = self.study.root / trace
            self.paths.setdefault(path.parts[-4]+"/"+path.parts[-3], path)
        current_files = {self.paths["current/"+task].resolve() for task in self.tasks}
        self.past_paths = {doc:path for doc,path in self.paths.items() if not doc.startswith("current/")
                           and path.resolve() not in current_files}
        self.docs["trace_directory"] = sorted(self.paths)
        self.archived = [doc for doc in self.paths if doc.rsplit("/",1)[-1] not in self.tasks]
        self.sites = {}
        self.add_last_round(self.folder,self.previous)
        self.incidents = {task:trace_incidents(events) for task,events in self.current_events.items()}
        for task, rows in self.incidents.items():
            self.docs["incidents/"+task] = [serialized(r) for r in rows]
            events = self.current_events[task]
            self.docs["owners/"+task] = [serialized(request_context(events,i,self.lib)) for i in range(len(events))]
        operations,_ = batch_operations(self.current_events)
        self.current_operations = operations
        self.docs["batch/operations"] = [serialized({k:v for k,v in g.items() if k!="sites"}) for g in operations]
        for task in self.tasks:
            self.docs["operations/"+task] = [serialized({"api":g["api"],"operation":g["operation"],**site})
                for g in operations for site in g["sites"].get(task,())]
        for name in ("semantic_assembly","semantic_library","policy_programs","semantic_handoff"):
            self.docs["code/"+name] = Path(__file__).with_name(name+".py").read_text(encoding="utf-8").splitlines()
        schema = DSL.split("handoff / name,")[0] + HANDOFF_SCHEMA + "\nControl handoff: {requires?,when?,defer?,apis?,reads?}. Default: {skill}.\n"
        schema += "Programs receive state with request, values, bindings, completed, conditions, read_scopes and receipts.\n"
        schema += ("State views can specify at:'parameter' to project only into a declared slot decision. Default at:'state' serves action menus and handoff. "
                   "Public datetime and active_card are available to projections. decision/card_parameter accepts skill or program over grounded option indices.\n")
        self.docs["dsl"] = (schema + CARD_DSL[CARD_DSL.index("An added card"):]).splitlines()
        if self.lib.get("menu_retrieval"):
            self.docs["code/retrieval"] = Path(__file__).with_name("retrieval.py").read_text(encoding="utf-8").splitlines()
            self.docs["dsl"].append("The program retrieves family, step, parameter and guide menus before Jev chooses. Inspect retrieval_menu and the actual decision page. Distinguish an excluded or unretrieved target from one offered and rejected. Improve the appropriate description, typed interface, state view, binding or decision skill. See code/retrieval.")
        if self.lib.get("menu_format") == "scoped":
            self.docs["code/scoped_menu"] = Path(__file__).with_name("scoped_menu.py").read_text(encoding="utf-8").splitlines()
            self.docs["dsl"] = [s for s in self.docs["dsl"] if "when this card has executed or failed" not in s] + [
                "Card fields: scope? describes actual coverage; applies_to? marks a specific goal. A short description is shown in the menu.",
                "Binding sees skill and a slot skill. Only the current writer of a live role contributes consumption notes. See code/scoped_menu.",
                "Reuse or replace an existing name when possible. To retire a redundant declared card, specialized handoff, view or family, set its value to null. Update any explicit references in the same patch. Each name counts once."]

    @cached_property
    def precedents(self):
        past = ((doc, read_events(path)) for doc,path in self.past_paths.items())
        index = TrainPrecedents(self.current_events, self.current_operations, past)
        self.docs["history/train_catalogue"] = [serialized(r) for r in index.records.values()]
        return index

    def lookup(self, action):
        if "related_train" in action:
            query = action["related_train"]
            return self.precedents.related(query["task"], query["event"])
        return super().lookup(action)

    def check_history_use(self, reply):
        sources = reply.get("history_use", [])
        example = ('{"trace":"<indexed earlier Train document>","event":0,'
                   '"shared_step":"<operation shared by both tasks>","scope_difference":"<difference in goal or scope>"}')
        if not isinstance(sources, list):
            raise ValueError(f"history_use must be a list. Example entry: {example}")
        for index, source in enumerate(sources):
            required = ("trace","event","shared_step","scope_difference")
            missing = [k for k in required if not isinstance(source,dict) or k not in source]
            where = f"history_use[{index}]"
            if missing:
                raise ValueError(f"{where} is missing {', '.join(missing)}. Example entry: {example}")
            record = self.precedents.records.get(source["trace"]) if isinstance(source["trace"],str) else None
            i = source["event"]
            if record is None or type(i) is not int or not 0 <= i < record["events"]:
                raise ValueError(f"{where}.trace and .event must identify a real earlier Train event. Example entry: {example}")
            if not all(isinstance(source.get(k), str) and source[k].strip() for k in ("shared_step", "scope_difference")):
                raise ValueError(f"{where}.shared_step and .scope_difference need nonempty descriptions. Example entry: {example}")

    def trace_data(self,doc):
        if doc not in self.sites:
            self.sites[doc] = trace_sites(read_events(self.paths[doc]))
        return self.sites[doc]

    @cached_property
    def retained_rounds(self):
        """Expose local evidence from earlier unsubmitted rounds on this Train batch."""
        from formal_run import read
        round_dir = self.folder.parent
        visit = round_dir.parent
        earlier = sorted(p for p in visit.glob(visit.name+"r[0-9][0-9]") if p.name < round_dir.name)
        _, separator, source = visit.name.partition("_revisit")
        if separator and source.isdigit():
            original = visit.parent/("b"+source)
            earlier = sorted(original.glob(original.name+"r[0-9][0-9]")) + earlier
        summaries = []
        for previous in earlier:
            decision = previous/"retained_parent.json"
            if not decision.is_file():
                continue
            info = read(decision)
            documents = []
            for which in ("draft", "revised"):
                trial = previous/"sol_scoped"/("trial_"+which)
                for name, filename, field in (("edits", "tested_patch.json", "objects"),
                                              ("observations", "local_evidence.json", "cases"),
                                              ("old_choices", "historical_choice.json", "tasks")):
                    file = trial/filename
                    if file.is_file() and field in (data := read(file)):
                        doc = f"retained/{previous.name}/{which}/{name}"
                        self.docs[doc] = [serialized(item) for item in data[field]]
                        documents.append(doc)
            summaries.append({"visit":previous.parent.name, "round":info["round"], "parent":info["version"],
                              "reason":info["reason"], "whole_candidate_scored":False, "documents":documents})
        return summaries

    def packet(self):
        tasks = [task_summary(item,self.current_events[item["task"]]) for item in self.index]
        families = {}
        for row in tasks:
            group = families.setdefault(row["task"].rsplit("_",1)[0],{"tasks":[],"solved":0,"output_tokens":0})
            group["tasks"].append(row["task"])
            group["solved"] += int(row["success"])
            group["output_tokens"] += row["output_tokens"]
        for row in tasks:
            if "last_measured_round" in row:
                detail = row["last_measured_round"]
                self.docs["changes/"+row["task"]] = [serialized(detail)]
                row["last_measured_round"] = {k:detail[k] for k in ("success_before_after","total_generated_tokens_before_after")}
                if "confirmation" in detail:
                    row["last_measured_round"]["confirmation_parent_candidate"] = detail["confirmation"]
        surfaces = {kind:{"controls":meaning,"check":check,"read":[d for d in docs if d in self.docs]}
                    for kind,(meaning,check,docs) in EDIT_SURFACES.items()}
        patterns = batch_patterns(tasks,self.incidents)
        self.docs["batch/patterns"] = [serialized(p) for p in patterns]
        past = self.precedents
        retained = self.retained_rounds
        return {**({"earlier_retained_train_rounds":retained,
                    "next_round":"Inspect the previous edits and observations. Try another cause or a repair "
                                 "for the observed limitation. You may keep the parent again if warranted."} if retained else {}),
                "parent_batch":{"tasks":len(tasks),"solved":sum(row["success"] for row in tasks),
                                "output_tokens":sum(row["output_tokens"] for row in tasks)},
                "scenarios":families,"tasks":tasks,"editable_components":surfaces,
                "last_round_cost_by_paired_outcome":paired_outcome_costs(
                    (row["last_measured_round"]["success_before_after"],
                     row["last_measured_round"]["total_generated_tokens_before_after"])
                    for row in tasks if "last_measured_round" in row),
                "observed_patterns":patterns[:12],
                "historical_comparisons":{"recordings":len(past.records),
                    "distinct_tasks":len({r["task"] for r in past.records.values()}),
                    "examples":past.suggestions(tasks), "catalogue":"history/train_catalogue"},
                "editor_history":edit_history(self.previous),"prior_train_traces":self.archived,
                "previous_impacts":"history",
                "execution":self.study.execution,
                "documents":[name for name in self.docs if not name.startswith(("api/","fragment/"))]}

    def ask(self, instructions, payload, folder, parse):
        original = folder
        attempt = 2
        while (folder/"turn_01").exists():
            folder = original/f"retry_{attempt:02d}"
            attempt += 1
        account = self.study.solver()
        thread = account.start(folder,"Use only the supplied public Train evidence. Return the requested JSON.")
        prompt = (instructions+'\nYou may reply with {"read":[{"doc":"current/<id>","start":0,"count":4}]} '
                  'or {"find":{"doc":"...","text":"..."}} or {"related_train":{"task":"...","event":0}} before your final action. '
                  'You can combine these evidence requests. The reply lists returned results, pending pages or '
                  'documents, and unexecuted items with a correction. Up to six read documents are returned per reply. '
                  'Resubmit the exact requests in pending when needed. Trial documents use the names in observations.\n'
                  +serialized(payload))
        for turn in range(1,11):
            if turn>=6:
                prompt += "\nSubmit the requested decision now. The final turn is reserved for your decision."
            if turn==10:
                prompt += "\nThis is the final reply. Use the evidence just returned. Only a final JSON action can be processed."
            if turn==10:
                prompt += '\nReturn action_json containing the complete serialized final action.'
                envelope = {"type":"object","additionalProperties":False,"properties":{"action_json":{"type":"string"}},
                            "required":["action_json"]}
                answer = account.turn(thread,prompt,folder/f"turn_{turn:02d}",schema=envelope)
            else:
                answer = account.turn(thread,prompt,folder/f"turn_{turn:02d}")
            try:
                reply = parse_json(answer)
                if turn==10:
                    reply = parse_json(reply["action_json"])
                    return parse(reply)
                if "read" in reply or "find" in reply or "related_train" in reply:
                    prompt = serialized(self.evidence_reply(reply))
                    continue
                return parse(reply)
            except (ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
                prompt = f"The reader or installer reported {exc}. Correct the lookup or return the requested final action."
        raise ValueError("Sol did not return a valid whole-batch revision")

    def validate_draft(self, reply):
        if reply.get("keep_parent") is True and reply.get("reason") and not reply.get("objects"):
            return reply
        validate_batch_effects(reply.get("batch_effects"),self.tasks)
        focus = reply.get("focus")
        if not isinstance(focus,dict) or not {"task","event","observable"}.issubset(focus):
            raise ValueError('focus needs three fields, for example {"task":"<current task ID>","event":1,"observable":"<downstream effect>"}.')
        task,index = focus["task"],focus["event"]
        if task not in self.tasks or type(index) is not int or not 0<=index<len(self.current_events[task]) or not focus.get("observable"):
            raise ValueError("The focus needs a current task, its real event index and an observable downstream result")
        if self.current_events[task][index]["event"] in ("task","summary"):
            raise ValueError("Identify an execution, decision or observation in the trace")
        preserve = reply.get("preserve_task")
        others = self.tasks-{task}
        solved = {item["task"] for item in self.index if item["success"] and item["task"] in others}
        if others and preserve not in (solved or others):
            raise ValueError("Choose another current task that was solved, if available")
        if not others and preserve is not None:
            raise ValueError("This batch has just the focal task")
        self.check_history_use(reply)
        self.install(self.lib,reply,self.sources,self.study.functions)
        return reply

    def draft(self):
        from formal_run import read, save
        path = self.folder/"focus_draft.json"
        if path.exists():
            return read(path)
        packet = self.packet()
        packet["local_27b_budget"] = {"fresh_judgments_per_round":FRESH_DECISION_LIMIT,
            "draft_max_judgments":FRESH_DECISION_LIMIT//2,"open_generations":OPEN_GENERATION_LIMIT,
            "output_tokens":PROBE_OUTPUT_LIMIT}
        choice_mode = self.study.settings.get("optimizer",{}).get("historical_choice")
        if choice_mode == SOLVED_FIRST_CHANGE:
            packet["historical_choice_preview"] = (
                "The program checks the first change on each previously solved earlier Train task for both "
                "the draft and any revision. Compare old actions against the candidate menu even when the parent "
                "already differs at that recorded decision. The report identifies menus shared with the parent. Matching "
                "inputs share a response. Historical failures are available for diagnosis in the Train catalogue.")
        elif choice_mode:
            packet["historical_choice_preview"] = (
                "One extra 27B choice is available on a successful earlier Train task at its first changed "
                "decision. Cite a relevant contrast in history_use to prioritize it. The draft review receives "
                "the recorded selection and the new choice on that page.")
        save(self.folder/"index.json",packet)
        result = self.ask(INSTRUCTIONS,packet,self.folder/"focus",self.validate_draft)
        save(path,result)
        return result

    def work(self,worker=None):
        from formal_run import read, save
        from .batch_revision_trials import preview_batch
        path = self.folder/"focus_proposal.json"
        if path.exists():
            return read(path)
        draft = self.draft()
        if draft.get("keep_parent"):
            save(self.folder/"local_evidence.json",{"cost":{},"trials":[]})
            save(path,draft)
            return draft
        worker = worker or self.study.workers[0]
        trials = []
        patches = {"draft":draft}
        def preview(key,prior=None):
            obs = preview_batch(self,patches[key],worker,key,prior)
            if "preservation" in obs:
                patches[key] = {**patches[key],"preserve_task":obs["preservation"]["task"]}
            trials.append({"version":key,"cases":obs["cases"],"batch":obs["batch"],
                           "historical_impact":obs["historical_impact"][key],"stage_cost":obs["stage_cost"],
                           "path_comparison":obs.get("path_comparison"),"preservation":obs.get("preservation"),
                           "wording":obs.get("wording"),
                           "documents":obs.get("documents",[])})
            if "historical_choice" in obs:
                trials[-1]["historical_choice"] = obs["historical_choice"]
            save(self.folder/"local_evidence.json",{"trials":trials,"cost":obs["cost"]})
            return obs["cost"]
        cost = preview("draft")
        draft = patches["draft"]
        def decision(reply):
            choice = reply.get("use")
            if not reply.get("reason"):
                raise ValueError('Provide the reason, for example {"use":"parent","reason":"<observed tradeoff>"}.')
            if choice in (*patches,"parent") and not reply.get("objects"):
                return reply
            if choice=="revise" and "revised" not in patches and (cost.get("fresh_decisions",0)<FRESH_DECISION_LIMIT or cost.get("open_generations",0)<OPEN_GENERATION_LIMIT):
                if "batch_effects" in reply:
                    validate_batch_effects(reply["batch_effects"],self.tasks)
                self.check_history_use(reply)
                self.install(self.lib,reply,self.sources,self.study.functions)
                if reply["objects"] != draft["objects"]:
                    return reply
            raise ValueError("Choose a tested version or parent. Only the first review can propose revised objects")
        payload = {"parent":self.packet(),"draft":draft,"observations":trials,
                   "local_27b_remaining":{"fresh_judgments":FRESH_DECISION_LIMIT-cost.get("fresh_decisions",0),
                   "open_generations":OPEN_GENERATION_LIMIT-cost.get("open_generations",0),
                   "output_tokens":PROBE_OUTPUT_LIMIT-cost.get("output_tokens",0)}}
        review_path = self.folder/"review_choice.json"
        if review_path.exists():
            review = read(review_path)
        else:
            instructions = REVIEW
            if trials[0].get("historical_choice", {}).get("mode") == SOLVED_FIRST_CHANGE:
                instructions += SOLVED_REVIEW
            elif "historical_choice" in trials[0]:
                instructions += ("\nInspect historical_choice on the draft when its status is observed. "
                                 "Compare the old and new choices with the old goal. Explain whether the new choice "
                                 "preserves the next step, offers useful reuse, or introduces interference. "
                                 "You can revise the relevant components. This is a single page observation; "
                                 "whole-task success and any subsequent revised menu remain unmeasured.")
            review = self.ask(instructions,payload,self.folder/"review",decision)
            save(review_path,review)
        if review["use"] == "revise":
            patches["revised"] = revised_patch(draft,review)
            cost = preview("revised",cost)
            instructions = FINAL_REVIEW
            if trials[-1].get("historical_choice", {}).get("mode") == SOLVED_FIRST_CHANGE:
                instructions += SOLVED_REVIEW
            review = self.ask(instructions,
                              {"focus":draft["focus"],"patches":patches,"trials":trials},self.folder/"final_review",decision)
        chosen = review["use"]
        if chosen=="parent":
            proposal = {"keep_parent":True,"reason":review["reason"],"focus":draft["focus"]}
        else:
            proposal = {**patches[chosen],"review_reason":review["reason"],"measured_preview":chosen}
            history = next(t["historical_impact"] for t in trials if t["version"]==chosen)
            save(self.folder.parent/"optimizer_impact.json",history)
        save(path,proposal)
        return proposal
