import time

from .card_assembly import CardAssembly
from .card_library import ACTIVE_QUESTIONS, CARD_DSL, CARD_PROMPT, patch_cards, ports
from .phase1_optimizer import PrefixProbeMixin, Workbench, leading_choices, serialized
from .formal_optimizer import read_events
from .runtime import parse_json
from .transport import WorkerError, WorkerUnavailable


class CardProbe(PrefixProbeMixin, CardAssembly):
    pass


class CardWorkbench(Workbench):
    prompt = CARD_PROMPT
    install = staticmethod(patch_cards)
    probe_type = CardProbe

    def __init__(self,*args):
        super().__init__(*args)
        self.docs["dsl"] = CARD_DSL.splitlines()
        self.docs["library/questions"] = [serialized({k:self.lib["questions"][k]}) for k in ACTIVE_QUESTIONS]
        self.docs["library/skills"] = [serialized({k:self.lib["skills"].get(k,"")}) for k in (*ACTIVE_QUESTIONS,"residual")]

    def catalog(self):
        return [{"name":name,"app":ports(card)[0],"inputs":ports(card)[1],"output":ports(card)[2],
                 "reads":ports(card)[3],"description":card["description"],"skill":card["skill"]}
                for name,card in self.lib["cards"].items()]

    def probe(self, proposal, task):
        from formal_run import save
        for attempt in range(3):
            try:
                result=super().probe(proposal,task)
                folder=self.folder/f"probe_{self.calls:02d}"
                doc="probe/"+folder.name
                self.paths[doc]=folder/"trace.jsonl"
                self.docs["trace_directory"]=sorted(self.paths)
                result["trace_document"]=doc
                result["native_choices"]=[{"event":i,"top":leading_choices(e)} for i,e in enumerate(
                    read_events(self.paths[doc])) if e["event"]=="model" and e["operation"]=="decide"]
                save(folder/"result.json",result)
                return result
            except (WorkerUnavailable,WorkerError) as exc:
                save(self.folder/f"probe_{self.calls:02d}"/"infrastructure.json",{"error":str(exc)})
                worker=self.study.workers[0]
                worker.close_worker("environment")
                worker.close_worker("model")
                if attempt==2:
                    raise
                time.sleep(2)

    def continue_prompt(self,prompt):
        replies=sorted(self.folder.glob("turn_*/response.txt"))
        if not replies:
            return 1,prompt
        last=replies[-1]
        turn=int(last.parent.name.split("_")[-1])
        action=parse_json(last.read_text(encoding="utf-8"))
        resume={"last_saved_action":action}
        if "probe" in action:
            query=action["probe"]
            resume["actual_prefix_result"]=self.probe(query["patch"],query["task"])
        return turn+1,prompt+"\nContinue the interrupted Train revision from this saved action and observation:\n"+serialized(resume)

    def work(self):
        # A transport outage during the final probe can leave an already submitted patch on disk.
        replies=sorted(self.folder.glob("turn_*/response.txt"))
        if replies:
            action=parse_json(replies[-1].read_text(encoding="utf-8"))
            if "objects" in action:
                self.install(self.lib,action,self.sources,self.study.functions)
                self.probe(action,action.get("probe_task") or self.index[0]["task"])
                return action
        return super().work()
