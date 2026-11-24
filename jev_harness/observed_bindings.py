"""Observe parameter binding and actual carried-state reads in isolated AppWorld probes."""
import json

from .card_library import ports


class ObserveBindings:
    def perform_card(self, name):
        completed = len(self.completed_cards)
        try:
            return super().perform_card(name)
        finally:
            if len(self.completed_cards) > completed:
                role = ports(self.registry["cards"][name])[2]
                info = self.provenance.get(role, {})
                if info.get("producer") == name and info.get("parameters"):
                    self.trace.add("trial_card_bindings", card=name, parameters=info["parameters"])

    def prime_generated_session(self, values):
        bound = super().prime_generated_session(values)
        self.observe_values = bool(bound)
        return bound

    def tool_request(self, op, event, node, **params):
        if event != "tool_batch" or node != "generation_agent" or not getattr(self, "observe_values", False):
            return super().tool_request(op, event, node, **params)
        # Isolated natural probe only. Keep the original generated program and its visible stdout intact.
        setup = self.environment.request("execute", permit_finish=False, code=(
            "jev_probe_keys = []\n"
            "class JevProbeValues(dict):\n"
            "    def __getitem__(self, key):\n"
            "        value = dict.__getitem__(self, key)\n"
            "        jev_probe_keys.append(key)\n"
            "        return value\n"
            "    def get(self, key, *default):\n"
            "        if key in self: jev_probe_keys.append(key)\n"
            "        return dict.get(self, key, *default)\n"
            "if type(compiled).__name__ != 'JevProbeValues':\n"
            "    compiled = JevProbeValues(compiled)\n"
            "    compiled_progress['live_values'] = compiled"))
        if "Execution failed." in str(setup.get("output","")):
            self.observe_values = False
            self.trace.add("generated_read_observation", available=False)
            return super().tool_request(op, event, node, **params)
        # SemanticProbe can stop immediately after the tool result; collect actual accesses then too.
        try:
            return super().tool_request(op, event, node, **params)
        finally:
            report = self.environment.request("execute", permit_finish=False,
                code="import json\nprint(json.dumps(list(dict.fromkeys(jev_probe_keys))))")
            try:
                keys = json.loads(report["output"])
            except (KeyError, ValueError):
                keys = None
            self.trace.add("generated_read_observation", available=isinstance(keys,list), roles=keys)
