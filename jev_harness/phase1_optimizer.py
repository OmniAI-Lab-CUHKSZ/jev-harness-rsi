"""Offline Train evidence and patch application for the public semantic harness."""
import re

from .formal_runtime import BudgetExhausted
from .formal_optimizer import api_context
from .train_evidence import TrainEvidence, leading_choices, read_page, serialized, visible_event
from .phase1 import PhaseOneExecutor, patch_library
from .staged_entry import StagedExecutor, completion
from .composable_v2 import routed
from .runtime import STEP_REVIEW, Trace, is_open_generation, parse_json
from .sol_account import AccountError
from .transport import WorkerUnavailable


PROMPT = """You are GPT-6.2 Sol Max, offline editor of a shared AppWorld harness for one frozen 27B.
Optimize the nine whole tasks: maximize number solved first, then minimize all generated output tokens,
including reasoning, failed attempts and local generation. Prefer reusable pieces, actual state-based Jev
choices and a partial compiled prefix when coverage is incomplete. A forward native choice uses zero output
tokens but costs a model call. A task can use an existing compiled fragment, a native option or residual
27B generation. The program alone builds candidate lists and questions. No task-time self-compilation.
Distinguish no installed ability, recall misses, rejected candidates, wrong selections, fields/bindings,
and an actual tool-side failure after a correct choice. Check actual calls, not just generated code.
Your role is to make ONE executable patch for this round from Train evidence. Source examples are not runtime
predicates: do not code task IDs, example identities, credentials or known answers. Prefer parametrized,
cross-goal operations. Successful but costly traces are also useful. Inspect the original records as needed.

Reply with {"read":[{"doc":"document key","start":0,"count":6,"offset":0}, ...]}
or {"find":{"doc":"document key","text":"literal text","start":0}}. Trace start is the zero-based event.
For long lines, use the returned next offset. Documents can be fetched through to the end. All fields of
the observable Train trajectory are available; credentials are redacted and private grading details excluded.
To try a candidate from a fresh Train state, optionally return {"probe":{"task":"one current task ID",
"patch":{...candidate...}}}. This runs REAL native choices and dependencies up to the first generation.
For a final proposal return {"reason":"specific diagnostic evidence","prediction":"observable event and expected
whole-task change","probe_task":"one current task ID","objects":[{"kind":"...","name":"...","value":...}]}.
Edit one to three NAMED objects total. Existing unrelated keys of a replaced object must be retained.

Editable object formats:
  menu: value={cues:[OR regexes over original instruction], after_roles?:[...], after_paths?:[...],
    question:string, skill?:string, options:[{fragment:installed_name,label:string,parameters?:named_parameter_source,
    inputs?:{role:expression},captures?:{role:{pattern:single_capture,mode:"one"|"many"}}}]}.
    Matching public cues or state can open a menu. For a lexical-only menu, a read retained by the
    coarse app filter can also be offered on a paraphrase. State-dependent menus keep their activation rules.
    Only options with genuinely reachable dependencies are shown.
    Group comparable alternatives. Runtime adds NONE and pages long lists of real options.
    These are scoped partial-read menus. Pick sources for an unfinished subgoal (e.g. which person's
    records, and what entity, direction, status and date range). They do not declare complete operations.
  fragment: value={definition:{inputs:[roles],output:role,app:string,source_tasks:[Train IDs],
    effect:string,steps:[...],result:expression,finish?:"answer"|"action",effect_kind?:"read"|"write"|"internal",
    slots?:{input_role:{question?:string,candidates:definition,skill?:string,requires?:[public_read_roles]}}},
    bindings:{private_key:expression},route?:{...}}.
    A private binding used by a step is namespaced automatically. A fragment's optional discovery route
    may reference only that fragment's output. Independently editable selection behavior belongs to menus.
    Put the public scope, data source, preconditions and effect in the fragment itself. A complete
    fragment needs only its own finish and input definitions: the program automatically lists it for
    native selection, no duplicate menu or exact-instruction discovery route. Infer existing enum
    domains from literal alternatives in old captures; for new ones specify slots directly. Candidates
    can be lists of label/value pairs or a projection of observed public records. New standalone
    partial reads can declare effect_kind:"read" without a dedicated menu; existing scoped menus get
    first chance. Credentials are dependencies. Keep writes scoped to the requested live objects.
  selector: name=complete or read; value={question?:string,skill?:string}. Candidate membership is
    determined by installed capabilities and public state, not by these prompts.
  route: value={patterns:[all regexes],exclude:[],after_roles?:[],after_paths?:[],goals:[roles],
    inputs?:{},captures?:{},complete:false}. Existing complete routes can describe completion, but
    matching the wording never runs a complete operation without Jev. A partial route supplies bindings
    or state cues; do not memorize sentences. Source-task IDs are offline provenance only.
  interface: value={question:expression,evidence:expression,candidates:definition,skill?:string,policy?:{}}
  decision_skill or generation_skill: a scoped prompt string. Menu skill name is menu_<menu_name>.
  handoff: scoped G prompt string for a completed output role or general.
  tool_context: value={apis:[public app__api names], cues?:[instruction regexes], after_roles?:[roles]}.
    At least one cue or role is needed. Prioritize public tool descriptions for residual generation
    without removing the built-in documentation tools or increasing the shortlist size.
  state: name=public output role; value={project:expression,description:string}. In a projection item is
    the observed role value and vars contains the public runtime values. The projection enriches G handoff.
  routing: name=existing local menu; value={defer?:[regex],minimum_probability?:number}. Basic
    app filtering keeps transitive providers, or the full pool without an explicit app; it is shared code.
    Improve wrong selections through fragment descriptions, candidate semantics or scoped questions.
  recovery: existing fragment name, value={skill:string,policy:{...}}. Generation instructions for
    an uncovered fragment are handed to the residual agent; no generation during assembly.
  program: name=discovery or handoff, value=a function body in the supplied policy language.
  No access to grader, dataset, budgets or compiler edits. Read docs 'dsl', 'code/staged_entry', 'code/phase1' or
    'code/policy_programs' when needed. Do not return patches to the optimizer itself."""


class ProbeComplete(Exception):
    pass


class PrefixProbeMixin:
    def _model(self, operation, **values):
        if is_open_generation(operation, values):
            raise ProbeComplete("Generation boundary")
        return super()._model(operation, **values)

    def run_prefix(self, task_id, lib, folder, seed, experiment_name):
        folder.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(folder/"trace.jsonl")
        self.seed, self.done = seed, False
        try:
            task = self.environment.request("prepare", task_id=task_id, seed=seed,
                experiment_name=experiment_name,
                **self.prepare_options())
            self.context={"task":task,"vars":{},"generated":{},"history":[]}
            self.trace.add("task",task_id=task_id,instruction=task["instruction"])
            try:
                self.execute_task(lib)
                end = "completed prefix"
            except (ProbeComplete,BudgetExhausted) as exc:
                end = str(exc)
            events = self.trace.events
            card_mode = "cards" in lib
            return {"task":task_id,"stop":end,"completed":self.done,
                    "fragments":self.completed_cards if card_mode else self.completed_roles,
                    "native_calls":sum(e["event"]=="model" and e["operation"]=="decide" for e in events),
                    "generated_calls":sum(e["event"]=="model" and is_open_generation(e["operation"], e["request"]) for e in events),
                    "step_review_calls":sum(e["event"]=="model" and e["request"].get("purpose")==STEP_REVIEW for e in events),
                    "output_tokens":sum(e["response"].get("output_tokens",0) for e in events if e["event"]=="model"),
                    "evidence":[visible_event(e) for e in events if e["event"] in
                    ("stage_prefilter","stage_candidates","stage_choice","stage_binding","stage_deferred","stage_enter",
                     "stage_completed","stage_handoff","fragment_recall","fragment_choice","fragment_choice_used",
                     "choice_used","fragment_failure","composition_missing","fallback","card_menu",
                     "card_executed","card_failed","assembly_candidates","assembly_selection","assembly_widen",
                     "adapter_login","program_error","fragment_handoff","fragment_reentry","step_review","handoff_selection")]}
        finally:
            self.trace.close()


class PrefixProbe(PrefixProbeMixin, StagedExecutor):
    pass


class LegacyPrefixProbe(PrefixProbeMixin, PhaseOneExecutor):
    pass


class Workbench(TrainEvidence):
    review_probe = False
    trial_reminder = ""
    def opening_evidence(self):
        return {}
    def catalog(self):
        defs = routed(self.lib)
        return [{"name":n,"app":f["app"],"effect":f.get("effect",""),"inputs":f["inputs"],
                 "output":f["output"],"finish":completion(n,f,defs)} for n,f in self.lib["workflow"]["fragments"].items()]

    prompt = PROMPT
    install = staticmethod(patch_library)
    probe_type = PrefixProbe

    def continue_prompt(self,prompt):
        return 1,prompt

    def __init__(self, study, lib, rows, history, folder, sources):
        super().__init__(study, lib, rows, history, folder, sources)
        self.calls = max((int(p.name[6:]) for p in folder.glob("probe_*") if p.name[6:].isdigit()),default=0)
        self.last_probed = None

    def probe_key(self, proposal, task):
        return (proposal["objects"],task,proposal.get("until")) if self.review_probe else (proposal["objects"],task)

    def probe(self, proposal, task):
        from formal_run import save
        if task not in self.tasks:
            raise ValueError("Probe requires one of the current Train tasks")
        candidate,_ = self.install(self.lib,proposal,self.sources,self.study.functions)
        self.calls += 1
        folder = self.folder/f"probe_{self.calls:02d}"
        worker = self.study.workers[0]
        engine = self.probe_type(worker.worker("model"),worker.worker("environment"),**self.study.execution)
        if hasattr(engine, "set_probe_target"):
            engine.set_probe_target(proposal.get("until"))
        label="_".join(("phase1_probe",self.study.root.name,*folder.relative_to(self.study.root).parts,task))
        result = engine.run_prefix(task,candidate,folder,self.study.seed,label)
        save(folder/"result.json",result)
        self.last_probed=self.probe_key(proposal,task)
        return result

    def work(self):
        from formal_run import save
        account = self.study.solver()
        thread = account.start(self.folder, "Use only supplied public Train observations. Return JSON. The runner executes proposals.")
        contents = {"batch":self.index,"available_fragments":self.catalog(),
            "menus":self.lib.get("menus",{}),"prior_outcomes":[{k:r.get(k) for k in ("round","candidate","accepted","prediction","parent_metrics","candidate_metrics")} for r in self.previous[-5:]],
            "documents":[k for k in self.docs if not k.startswith("api/")], "traces":list(self.paths),
            "relevant_api_schemas":api_context(self.study.functions,self.index,cap=22),
            "api_lookup":"Read or search api_names, then read api/spotify__show_playlist (or api/spotify.show_playlist) for the full schema. events/<task_id> maps each raw event to actual requests and stdout excerpts."}
        contents.update(self.opening_evidence())
        save(self.folder/"index.json",contents)
        first_turn,prompt=self.continue_prompt(self.prompt+"\n"+serialized(contents))
        for turn in range(first_turn,13):
            if turn == 8 and self.trial_reminder and not self.calls:
                prompt += "\n" + self.trial_reminder
            if self.review_probe and turn == 12:
                prompt += ("\nThis is the final interaction (12 of 12). Return the complete final patch now as "
                           "{reason,prediction,probe_task,until?,objects:[{kind,name,value}]}. Confirm or revise "
                           "using the evidence already returned. A read, find or probe request cannot be "
                           "answered after this reply.")
            raw=account.turn(thread,prompt,self.folder/f"turn_{turn:02d}")
            try:
                answer=parse_json(raw)
                if "objects" in answer:
                    self.install(self.lib,answer,self.sources,self.study.functions)
                    probe_task=answer.get("probe_task") or self.index[0]["task"]
                    if self.last_probed!=self.probe_key(answer,probe_task):
                        observed = self.probe(answer,probe_task)
                        if self.review_probe and turn < 12:
                            prompt = "Review this actual Train trial. Confirm by resubmitting the same objects or revise them.\n" + serialized(observed)
                            continue
                    return answer
                if "probe" in answer:
                    q=answer["probe"]
                    out=self.probe(q["patch"],q["task"])
                elif "find" in answer:
                    out=self.find_evidence(answer["find"])
                else:
                    out=[]
                    for q in answer["read"][:5]:
                        if isinstance(q,str):
                            q={"doc":q}
                        try:
                            out.append({"doc":q["doc"],**read_page(self.lines(q["doc"]),q.get("start",0),q.get("count",6),q.get("offset",0))})
                        except (KeyError, TypeError, ValueError) as exc:
                            out.append({"doc":q.get("doc"),"error":str(exc),"lookup":"Use api_names or trace_directory"})
                prompt=serialized(out)
            except (KeyError, ValueError, TypeError, AttributeError, IndexError, re.error) as exc:
                prompt="The installer or evidence reader returned "+str(exc)+". Correct the request from the same parent."
            if turn>=9:
                prompt+="\nReturn the final valid patch using the evidence you have."
        raise ValueError("Sol did not return an installable patch within the allotted dialogue")


def propose_one(study, library, rows, history, folder, sources):
    from formal_run import save
    bench = Workbench
    attempt_folder = folder/"sol"
    if "assembly" in library and "cards" in library:
        from .semantic_workbench import SemanticWorkbench, revision_attempt_folder
        options = study.settings.get("optimizer", {})
        bench,attempt_folder = SemanticWorkbench,revision_attempt_folder(folder)
        if options.get("batch_revision"):
            from .batch_revision import BatchRevisionWorkbench
            bench = BatchRevisionWorkbench
            attempt_folder = folder/"sol_scoped"
        elif options.get("history_preview"):
            from .history_workbench import HistoryWorkbench
            bench = HistoryWorkbench
        elif options.get("fragment_trials"):
            from .fragment_trials import FragmentTrialWorkbench
            bench = FragmentTrialWorkbench
        elif options.get("same_batch_feedback_from") is not None:
            # A visit such as b06_revisit03 uses the rules in effect after batch 6.
            scheduled_batch = int(folder.parent.name.partition("_")[0].removeprefix("b"))
            if scheduled_batch >= options["same_batch_feedback_from"]:
                from .batch_feedback import BatchFeedbackWorkbench
                bench = BatchFeedbackWorkbench
    elif "cards" in library:
        from .card_workbench import CardWorkbench
        bench = CardWorkbench
    reuse_trials = study.settings.get("optimizer",{}).get("batch_revision") is True
    for attempt in range(2):
        target=attempt_folder if not attempt or reuse_trials else attempt_folder.with_name(attempt_folder.name+"_reconnect")
        work=bench(study,library,rows,history,target,sources)
        try:
            return work.work()
        except AccountError as exc:
            save(target/"account_error.json",{"error":str(exc)})
            study.close_account()
            if attempt or re.search(r"quota|usage limit|rate.?limit|sign.?in|unauthori[sz]ed",str(exc),re.I):
                raise
        except WorkerUnavailable as exc:
            if not reuse_trials or attempt:
                raise
            save(target/"probe_service_error.json",{"error":str(exc)})
            for service in ("model","environment"):
                study.workers[0].close_worker(service)
