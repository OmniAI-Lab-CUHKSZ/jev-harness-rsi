"""Assemble independent reads, transformations and effects on live AppWorld records."""
from itertools import combinations
import json
import re

from .expressions import resolve
from .formal_runtime import FormalExecutor
from .runtime import MAX_REAL_OPTIONS, NativeNone, Uncovered, parse_json


# Each operation consumes the current typed data. No task or scenario dispatches an operation.
LIBRARIES = {"songs": "directly saved songs", "albums": "songs in all saved albums",
             "playlists": "songs in all playlists in the user's playlist library"}
FIELDS = {"play_count": "number of plays or listens", "like_count": "number of likes",
          "release_date": "release date", "duration": "duration", "rating": "rating",
          "review_count": "number of reviews"}
SORT_ROWS = {"op": "sorted", "args": [{"ref": "rows"}, {"ref": "field"}, {"ref": "descending"}]}
DATA_OPERATIONS = {
    "ordered_records": SORT_ROWS,
    "title_text": {"op": "join", "args": [{"map": {"source": {"ref": "rows"},
                                                     "value": {"ref": "item.title"}}}, ", "]},
    "unique_ids": {"op": "unique", "args": [{"ref": "ids"}]},
    "record_count": {"op": "length", "args": [{"ref": "ids"}]},
}
AUTH_APPS = {"library": "spotify", "playlist": "spotify", "album": "spotify", "directory": "file_system"}


def option(label, value):
    return {"label": label, "value": value}


def normalized(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def literal_names(text):
    quoted = re.findall(r"""["“]([^"”\n]+)["”]""", text)
    capitals = re.findall(r"\b[A-Z][\w'-]*(?:(?:[:,-]?\s+)(?:(?:of|the|and|in|at)\s+)?[A-Z][\w'-]*)+", text)
    return list(dict.fromkeys(x.strip() for x in quoted + capitals if x.strip()))


def put_ids(rows):
    ids = []
    for row in rows:
        key = row.get("song_id", row.get("id")) if isinstance(row, dict) else row
        if type(key) is not int:
            raise Uncovered("Song membership did not identify a live song")
        ids.append(key)
    return list(dict.fromkeys(ids))


class FragmentAssembly(FormalExecutor):
    def pick_page(self, part, pick, **mode):
        local = [option(row["label"], i) for i, row in enumerate(part)]
        return part[pick(local, **mode)]

    def choose(self, name, choices, *, widen_on_none=False, **facts):
        if not choices:
            raise Uncovered("No grounded options for " + name)
        evidence = {"request": self.request, **facts}
        question = self.registry["questions"][name]
        skill = self.registry["skills"].get(name, "")

        review = name == "step" and self.step_review_threshold is not None
        def pick(items, **kwargs):
            return self.decision(question, evidence, items, skill=skill, node="fragment/" + name, **kwargs)

        # A returned value may be a list. Use a page-local index to preserve its identity.
        while len(choices) > MAX_REAL_OPTIONS:
            winners = []
            closest = None
            for start in range(0, len(choices), MAX_REAL_OPTIONS):
                part = choices[start:start + MAX_REAL_OPTIONS]
                try:
                    winner = self.pick_page(part, pick)
                except NativeNone as exc:
                    if review and not widen_on_none:
                        confidence = max(exc.response["probabilities"])
                        if confidence < self.step_review_threshold and (closest is None or confidence < closest[0]):
                            closest = (confidence, part, exc.response)
                except Uncovered as exc:
                    if "uncovered option" not in str(exc):
                        raise
                else:
                    winners.append(winner)
            choices = winners
            if not choices:
                # All pages abstained. Revisit the least certain page, keeping its original option indices.
                if closest is not None:
                    _, part, response = closest
                    return self.pick_page(part, pick, review_step=True, native_response=response)["value"]
                raise Uncovered("No grounded choice for " + name)
        return pick(choices, review_step=True, widen_on_none=widen_on_none) if review else pick(choices)

    def used(self, name, input_role, output_role, **details):
        self.trace.add("assembled_fragment", name=name, consumes=input_role, provides=output_role,
                       **details)
        self.sequence.append(name)

    def apply_data(self, port, **values):
        piece = self.registry.get("pieces", {}).get(port)
        result = resolve(piece["result"] if piece else DATA_OPERATIONS[port], values)
        if piece:
            self.trace.add("extracted_piece", name=piece["name"], port=port,
                           inputs=list(values), size=len(result) if isinstance(result, list) else None)
        return result

    def token(self, app):
        if app not in self.tokens:
            passwords = self.call("supervisor", "show_account_passwords", {}, "session")
            credentials = next((row for row in passwords if row["account_name"] == app), None)
            if credentials is None:
                raise Uncovered("No account for " + app)
            session = self.call(app, "login", {"username": self.context["task"]["supervisor"]["email"],
                                "password": credentials["password"]}, "session")
            self.tokens[app] = session["access_token"]
        return self.tokens[app]

    def pages(self, api, **arguments):
        records = []
        page = 0
        while True:
            values = self.call("spotify", api, {**arguments, "page_index": page, "page_limit": 20}, api)
            if not isinstance(values, list):
                raise Uncovered("Unexpected list response for " + api)
            records.extend(values)
            if len(values) < 20:
                return records
            page += 1

    def members(self, record, kind):
        items = record.get("song_ids", record.get("songs"))
        if items is None:
            args = {kind + "_id": record[kind + "_id"]}
            if kind == "playlist":
                args["access_token"] = self.token("spotify")
            detail = self.call("spotify", "show_" + kind, args, "collection_members")
            items = detail.get("song_ids", detail.get("songs"))
        if not isinstance(items, list):
            raise Uncovered("Missing membership for " + kind)
        return put_ids(items)

    def libraries(self):
        labels = self.registry.get("collections", LIBRARIES)
        masks = [list(group) for size in range(1, len(labels) + 1)
                 for group in combinations(labels, size)]
        selected = self.choose("libraries", [option(" AND ".join(labels[x] for x in mask), mask)
                                            for mask in masks])
        ids, groups = [], []
        for source in selected:
            kind = {"songs": "song", "albums": "album", "playlists": "playlist"}[source]
            records = self.pages("show_" + kind + "_library", access_token=self.token("spotify"))
            if kind == "song":
                ids.extend(put_ids(records))
                self.seed_rows.update({r["song_id"]: r for r in records})
            else:
                for r in records:
                    children = self.members(r, kind)
                    groups.append({"kind": kind, "id": r[kind + "_id"], "title": r["title"],
                                   "song_ids": children})
                    ids.extend(children)
            self.used("read_" + kind + "_library", "account", "collection_members",
                      source=source, records=len(records))
        self.public["collections"] = groups
        self.public["scope"] = [LIBRARIES[s] for s in selected]
        return ids

    def named(self, records, kind):
        if not records:
            raise Uncovered("No live matches for the named " + kind)
        request = " " + normalized(self.request) + " "
        exact = [r for r in records if " " + normalized(r["title"]) + " " in request]
        options = exact or records
        i = self.choose("entity", [option(r["title"], i) for i, r in enumerate(options)], kind=kind)
        return options[i]

    def search_named_album(self):
        spans = literal_names(self.request)
        query = None
        if spans:
            try:
                query = self.choose("query", [option(s, s) for s in spans], entity="public catalog album")
            except Uncovered:
                pass
        if query is None:
            # G dispatches a single missing public read. Only the actual API response supplies records to D.
            command = self.generate_message(
                "Produce the arguments for the next public Spotify search_albums read. Resolve the "
                "album name from the request. Return only JSON with one string field, query. Leave all "
                "selection, ranking, playback and submission for later steps.", purpose="G_bridge")
            payload = parse_json(command["text"].split("</think>")[-1])
            query = payload.get("query")
            if not isinstance(query, str) or not normalized(query):
                raise Uncovered("The generated read did not provide an album query")
            self.trace.add("bridge_call", app="spotify", api="search_albums", query=query)
        records = self.pages("search_albums", query=query)
        self.used("search_catalog_albums", "public_query", "album_records", records=len(records))
        return self.named(records, "album")

    def read_source(self):
        apps = set()
        if re.search(r"\b(spotify|songs?|albums?|playlists?|music|tracks?)\b", self.request, re.I):
            apps.add("spotify")
        if re.search(r"\b(file(?:[ _]system)?|files|director(?:y|ies)|folders?)\b|~/", self.request, re.I):
            apps.add("file_system")
        names = [n for n in self.registry["source"] if not apps or AUTH_APPS[n] in apps]
        mode = self.choose("source", [option(self.registry["source"][n], n) for n in names])
        self.read_mode(mode)

    def read_mode(self, mode):
        if mode == "directory":
            paths = list(dict.fromkeys(x.rstrip(".,;") for x in re.findall(r"""(?:~/|(?<!\w)/)[^\s"'\x60,;]+""", self.request)))
            path = self.choose("directory", [option(x, x) for x in paths])
            data = self.call("file_system", "show_directory",
                             {"directory_path": path, "recursive": True, "access_token": self.token("file_system")},
                             "directory_entries")
            self.public.update(source_directory=path, directory_entries=data)
            self.used("read_directory", "source_path", "directory_entries", source=path)
            return
        if mode == "library":
            ids = self.libraries()
        else:
            if mode == "playlist":
                entities = self.pages("show_playlist_library", access_token=self.token("spotify"))
                record = self.named(entities, "playlist")
            else:
                record = self.search_named_album()
            ids = self.members(record, mode)
            self.public["scope"] = [mode, record["title"]]
            self.public["collections"] = [{"kind": mode, "id": record[mode + "_id"],
                                            "title": record["title"], "song_ids": ids}]
            self.used("read_" + mode + "_members", mode + "_reference", "song_ids", count=len(ids))
        self.public["song_ids"] = self.apply_data("unique_ids", ids=ids)
        self.used("distinct_song_ids", "collection_members", "song_ids", count=len(self.public["song_ids"]))

    def generated_source(self):
        """G supplies a missing read call; record identity and subsequent actions stay in the outer loop."""
        self.choose("missing_data", [option("A scoped set of Spotify song IDs for an installed operation "
                                           "such as filtering, ranking, returning titles or playing.", "song_ids")])
        docs = {}
        for item in self.context["task"]["functions"]:
            fn = item.get("function", item)
            name = fn["name"]
            if not name.startswith(("spotify__search_", "spotify__show_")):
                continue
            if not any(noun in name for noun in ("song", "playlist", "album")):
                continue
            docs[name] = fn["parameters"]
        signatures = {name: sorted(spec.get("properties", {})) for name, spec in docs.items()}
        response = self._model("generate", messages=[
            {"role": "system", "content": "Perform the missing source read in simulated AppWorld. "
             "Return JSON {\"api\":\"spotify__...\",\"arguments\":{...}} for ONE public read that yields "
             "the specified collection or its member songs. The harness supplies authentication and "
             "pagination. Do not rank, play, submit, or invent entity IDs. Prefer the current user's "
             "library for an owned playlist and the public catalog for an arbitrary album."},
            {"role": "user", "content": json.dumps({"request": self.request, "public_reads": signatures})}],
            thinking=self.generation_thinking, purpose="G_bridge", max_tokens=self.generation_tokens)
        call = parse_json(response["text"])
        name, args = call.get("api"), call.get("arguments")
        if name not in docs or not isinstance(args, dict):
            raise Uncovered("The open source step did not identify an available public read")
        spec = docs[name]
        args = {k: v for k, v in args.items() if k not in ("access_token", "page_index", "page_limit")}
        if set(args) - spec.get("properties", {}).keys():
            raise Uncovered("The open read used an unknown argument")
        if "access_token" in spec.get("properties", {}):
            args["access_token"] = self.token("spotify")
        api = name.split("__", 1)[1]
        self.trace.add("bridge_call", app="spotify", api=api)
        values = self.pages(api, **args) if "page_index" in spec.get("properties", {}) else self.call(
            "spotify", api, args, "open_source_read")
        records = values if isinstance(values, list) else [values]
        if not records or not all(isinstance(r, dict) for r in records):
            raise Uncovered("No typed records returned by the source read")
        if all("song_id" in r for r in records):
            ids = put_ids(records)
            self.seed_rows.update({r["song_id"]: r for r in records})
            scope = [api, {k:v for k,v in args.items() if k != "access_token"}]
        else:
            kind = next((kind for kind in ("playlist", "album") if all(
                kind + "_id" in r and "title" in r for r in records)), None)
            if kind is None:
                raise Uncovered("Read yielded no identifiable song collection")
            record = self.named(records, kind)
            ids = self.members(record, kind)
            scope = [kind, record["title"]]
            self.public["collections"] = [{"kind": kind, "id": record[kind + "_id"],
                                           "title": record["title"], "song_ids": ids}]
        self.public.update(scope=scope, song_ids=self.apply_data("unique_ids", ids=ids))
        self.used("read_missing_source", "generated_public_read", "song_ids", count=len(ids))
        self.trace.add("fragment_reentry", scope=scope, count=len(ids), via="public_read")

    def hydrate(self, fields):
        ids = self.public["song_ids"]
        for sid in ids:
            if not fields.issubset(self.seed_rows.get(sid, {})):
                r = self.call("spotify", "show_song", {"song_id": sid}, "song_attributes")
                if not isinstance(r, dict) or not fields.issubset(r):
                    raise Uncovered("Missing requested fields in live song response")
                self.seed_rows[sid] = r
        rows = [self.seed_rows[sid] for sid in ids]
        self.public["song_records"] = rows
        self.used("read_song_attributes", "song_ids", "song_records", count=len(rows), fields=sorted(fields))
        return rows

    def filter_genre(self):
        rows = self.hydrate({"genre"})
        genres = sorted({r["genre"] for r in rows if isinstance(r["genre"], str) and r["genre"]})
        exact = [g for g in genres if " " + normalized(g) + " " in " " + normalized(self.request) + " "]
        chosen = self.choose("genre", [option(g, g) for g in exact or genres])
        subset = [r for r in rows if r["genre"] == chosen]
        self.criteria["genre"] = chosen
        self.set_subset(subset)
        self.used("filter_genre", "song_records", "song_subset", genre=chosen, count=len(subset))

    def set_subset(self, songs):
        self.public["song_records"] = songs
        self.public["song_ids"] = [r["song_id"] for r in songs]
        self.public["selection"] = dict(self.criteria)

    def rank(self):
        orders = [option(f"{desc}, {direction}", (k, descending)) for k, desc in FIELDS.items()
                  for descending, direction in ((True, "highest or latest first"), (False, "lowest or earliest first"))]
        field, descending = self.choose("order", orders)
        n = len(self.public["song_ids"])
        numbers = sorted({1} | {int(s) for s in re.findall(r"\b[1-9]\d*\b", self.request)})
        choices = [option("One song" if k == 1 else f"The first {k} songs", k) for k in numbers]
        choices.append(option("All matching songs in ranked order", "all"))
        limit = self.choose("limit", choices, records=n)
        rows = self.hydrate({field, "title"})
        if any(r[field] is None for r in rows):
            raise Uncovered("Some songs have no value for the requested ranking")
        order = self.apply_data("ordered_records", rows=rows, field=field, descending=descending)
        subset = order if limit == "all" else order[:limit]
        self.criteria.update(field=field, descending=descending, limit=limit)
        self.set_subset(subset)
        self.used("rank_records", "song_records", "ranked_songs", field=field, limit=limit, count=len(subset))

    def choose_next(self):
        ids = self.public["song_ids"]
        available = list(self.registry["operations"])
        if "genre" in self.criteria or "field" in self.criteria or not ids:
            available.remove("filter_genre")
        if "field" in self.criteria or not ids:
            available.remove("rank")
        if len(ids) != 1:
            available.remove("play")
        return self.choose("next", [option(self.registry["operations"][op], op) for op in available],
                           scope=self.public["scope"], completed=self.sequence, conditions=self.criteria,
                           current_entity="song", current_record_count=len(ids))

    def perform(self, op):
        if op == "rank":
            self.rank()
        elif op == "filter_genre":
            self.filter_genre()
        elif op == "play":
            sid = self.public["song_ids"][0]
            self.call("spotify", "play_music", {"song_id": sid, "access_token": self.token("spotify")}, "play")
            self.used("play_song", "one_song_id", "playback", song_id=sid)
            self.call("supervisor", "complete_task", {}, "submit_effect")
        else:
            if op == "count":
                answer = self.apply_data("record_count", ids=self.public["song_ids"])
                self.used("count_distinct", "song_ids", "number", count=answer)
            elif op == "titles":
                answer = self.apply_data("title_text", rows=self.hydrate({"title"}))
                self.used("project_titles", "song_records", "text")
            else:
                raise Uncovered("Unsupported operation " + op)
            self.call("supervisor", "complete_task", {"answer": answer}, "submit_answer")

    def assemble_available(self):
        if "song_ids" not in self.public:
            raise Uncovered("No installed next operation for these records")
        while not self.done:
            self.perform(self.choose_next())

    def code(self, code, node=None, permit_finish=True):
        result = super().code(code, node, permit_finish)
        if self.done or node != "generation_agent" or not re.search(r"\bfragment_resume\s*=", code):
            return result
        # Read the value from the executed Python cell. Natural-language suggestions do not enter a menu.
        transfer = super().code(
            "import json\nprint(json.dumps(fragment_resume))\nfragment_resume = None",
            "read_returned_data", permit_finish=False)
        try:
            state = json.loads(transfer["output"].strip())
            if not isinstance(state, dict) or not isinstance(state.get("scope"), str) or not state["scope"].strip():
                raise Uncovered("No described scope returned")
            if not isinstance(state.get("song_ids"), list):
                raise Uncovered("No list of observed song IDs returned")
            ids = put_ids(state["song_ids"])
            identity = (state["scope"], tuple(ids))
            if identity in self.resumed:
                raise Uncovered("This returned scope was already processed")
            self.resumed.add(identity)
            self.public = {"scope": [state["scope"]], "song_ids": ids}
            self.criteria = {}
            self.trace.add("fragment_reentry", scope=state["scope"], count=len(ids))
            self.assemble_available()
        except (Uncovered, KeyError, IndexError, TypeError, ValueError) as error:
            self.trace.add("resume_uncovered", reason=str(error))
            if self.public:
                self.prime_generated_session(self.public)
                self.residual_progress.update(variables=list(self.public), scope=self.public.get("scope"),
                                              completed=self.sequence, unresolved=str(error))
        return result

    def execute_task(self, library):
        self.registry, self.request = library, self.context["task"]["instruction"]
        self.public, self.tokens, self.seed_rows, self.criteria = {}, {}, {}, {}
        self.sequence = []
        self.resumed = set()
        try:
            try:
                self.read_source()
            except Uncovered:
                if "missing_data" not in self.registry["questions"]:
                    raise
                self.generated_source()
            self.assemble_available()
        except (Uncovered, KeyError, IndexError, TypeError, ValueError) as error:
            reason = str(error)
            self.trace.add("fragment_handoff", reason=reason, available=list(self.public))
            if self.public:
                self.prime_generated_session(self.public)
            self.residual_progress = {"variables": list(self.public), "scope": self.public.get("scope"),
                                      "selection": self.criteria, "completed": self.sequence, "unresolved": reason}
            self.agent(self.registry["skills"]["residual"] +
                       "\nWhen a tool read or a missing calculation gives you a new, correctly scoped set "
                       "of live song IDs and ranking, counting, returning titles or playback remains, assign "
                       "fragment_resume = {'scope': 'describe the actual scope', 'song_ids': ids} and end "
                       "that cell. The program can resume on the returned data. Otherwise use the tools "
                       "to finish the remaining task.", self.residual_progress)

    def summarize(self, *args):
        result = super().summarize(*args)
        result["fragments"] = self.sequence
        result["bridges"] = sum(e["event"] == "bridge_call" for e in self.trace.events)
        result["generated_reentries"] = sum(e["event"] == "fragment_reentry" and e.get("via") != "public_read"
                                            for e in self.trace.events)
        result["public_read_reentries"] = sum(e["event"] == "fragment_reentry" and e.get("via") == "public_read"
                                             for e in self.trace.events)
        result["extracted_pieces"] = [e["port"] for e in self.trace.events if e["event"] == "extracted_piece"]
        result["fallback"] = [e["reason"] for e in self.trace.events if e["event"] == "fragment_handoff"]
        return result
