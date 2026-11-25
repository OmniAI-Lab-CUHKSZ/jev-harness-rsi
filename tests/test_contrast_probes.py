from collections import Counter
import unittest

from jev_harness.contrast_probes import MeteredModel, recorded_input
from jev_harness.phase1_optimizer import ProbeComplete
from jev_harness.runtime import STEP_REVIEW


class Respond:
    def __init__(self):
        self.calls = []

    def request(self,op,**values):
        self.calls.append((op,values))
        return {"output_tokens":values.get("max_tokens",0),"reasoning_tokens":0,"backend_requests":1}


class ContrastProbeTests(unittest.TestCase):
    def test_twenty_four_fresh_choices_are_shared_and_cached_choices_are_free(self):
        remote,cost = Respond(),Counter()
        draft = MeteredModel(remote,{},cost,12,False)
        for number in range(12):
            draft.request("decide",question=str(number))
        revised = MeteredModel(remote,{},cost,24,False)
        for number in range(12,24):
            revised.request("decide",question=str(number))
        revised.request("decide",question="23")
        with self.assertRaises(ProbeComplete):
            revised.request("decide",question="24")
        self.assertEqual((len(remote.calls),cost["cached_decisions"],cost["output_tokens"]),(24,1,0))

    def test_changed_evidence_or_skill_requires_a_new_decision(self):
        values = {"question":"Which source?","options":["Notes","Files"],"state":{"available":2},"skill":"Use live inputs"}
        event = {"operation":"decide","request":{**values,"purpose":STEP_REVIEW},"response":{"thinking_used":True,"index":0}}
        remote,cost = Respond(),Counter()
        model = MeteredModel(remote,{recorded_input(event):event["response"]},cost,2,False)
        self.assertEqual(model.request("decide",**values)["index"],0)
        model.request("decide",**{**values,"state":{"available":3}})
        model.request("decide",**{**values,"skill":"Resolve the actor"})
        self.assertEqual((cost["cached_decisions"],cost["fresh_decisions"],len(remote.calls)),(1,2,2))

    def test_generation_and_thinking_share_the_remaining_budget(self):
        remote,cost = Respond(),Counter(output_tokens=4016,open_generations=2)
        model = MeteredModel(remote,{},cost,2,True)
        with self.assertRaises(ProbeComplete):
            model.request("decide",thinking="auto",think_budget=256)
        model.request("generate",max_tokens=1024)
        self.assertEqual((remote.calls[0][1]["max_tokens"],cost["output_tokens"],cost["open_generations"]),(80,4096,3))
        with self.assertRaises(ProbeComplete):
            model.request("generate",max_tokens=1024)
