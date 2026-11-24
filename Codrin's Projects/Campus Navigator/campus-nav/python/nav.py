"""Navigation by behaviour cloning from the recorded walks.

Demonstrations. Every saved walk is cut into legs between waypoints
(walks.legs). Along each leg, every QUARTER metre walked becomes one example

    state  = (leg from -> to, turns made so far on the leg, metres since the waypoint)
    action = the next direction (forward to the waypoint / turn left / turn
             right / turn around) and the metres until it

The policy is k-nearest-neighbours on those examples: for the current state
it takes the K closest examples (same leg, same number of turns made, nearest
distance) from all demonstrations, of both walkers, and copies their
majority action and their average distance to it. With one demo it repeats
that walk; with several it averages where each of you turned.

Following along. Distance comes from the step counter; turns from the
heading tracker. A turn is a landmark: when you make the turn the policy
expects, the distance is snapped to where the demonstrations made it, so
distance errors don't pile up over a route. A turn the other way is flagged
("expected a right turn"). When the distance passes the end of a leg (the
demos' average length), the navigator moves on to the next leg. "I'm at X"
puts you at a waypoint at any time and plans again from there.

Routes chain demonstrated legs (breadth-first over waypoints), in the
direction they were walked.
"""
from collections import Counter, deque

import numpy as np

STEP_M = 0.25              # one example per this many metres along a leg
K = 5                      # neighbours
NOW_M = 3.0                # a turn closer than this is "turn now"
ARRIVE_M = 1.5             # within this of a waypoint: arrived


def _maneuvers(leg: dict) -> tuple:
    """[(metres from the leg start, 'left' | 'right' | 'around-left' | 'around-right')], length."""
    out, d, i = [], 0.0, 0
    items = leg["items"]
    while i < len(items):
        it = items[i]
        if it["do"] == "forward":
            d += it["m"]
            i += 1
            continue
        n = 1
        while i + n < len(items) and items[i + n]["do"] == it["do"] and items[i + n].get("turn") == it.get("turn"):
            n += 1
        out.append((d, it["do"] if n == 1 else f"around-{it['do']}"))
        i += n
    return out, d


class Policy:
    def __init__(self):
        self.examples = {}       # (from, to) -> list of (turns_done, metres, action, metres_to_action)
        self.demos = {}          # (from, to) -> list of (maneuvers, length, walker)

    def fit(self, walk_summaries: list):
        examples, demos = {}, {}            # built aside, swapped in at the end (the worker thread reads them)
        for w in walk_summaries:
            for leg in w["legs"]:
                key = (leg["from"], leg["to"])
                man, length = _maneuvers(leg)
                if length < 1.0:
                    continue
                demos.setdefault(key, []).append((man, length, w.get("walker", "?")))
                ex = examples.setdefault(key, [])
                for d in np.arange(0.0, length + STEP_M, STEP_M):
                    done = sum(1 for m, _ in man if m <= d)
                    nxt = next(((m, a) for m, a in man if m > d), None)
                    if nxt is None:
                        ex.append((done, float(d), "arrive", max(length - d, 0.0)))
                    else:
                        ex.append((done, float(d), nxt[1], nxt[0] - d))
        self.examples, self.demos = examples, demos
        return self

    @property
    def legs(self):
        return sorted(self.demos)

    def predict(self, key, done: int, d: float):
        """-> (action, metres to it) for this leg, turns done, metres walked."""
        ex = self.examples.get(key)
        if not ex:
            return None, None
        same = [e for e in ex if e[0] == done] or [e for e in ex if e[0] == max(x[0] for x in ex)]
        near = sorted(same, key=lambda e: abs(e[1] - d))[:K]
        action = Counter(e[2] for e in near).most_common(1)[0][0]
        dist = [e[3] - (d - e[1]) for e in near if e[2] == action]     # shifted to the current distance
        return action, max(0.0, float(np.mean(dist)))

    def turn_at(self, key, index: int):
        """Where the demos made their index-th turn on this leg (mean metres), and which way."""
        spots = [man[index] for man, _, _ in self.demos.get(key, []) if index < len(man)]
        if not spots:
            return None, None
        return float(np.mean([m for m, _ in spots])), Counter(a for _, a in spots).most_common(1)[0][0]

    def n_turns(self, key) -> int:
        counts = [len(man) for man, _, _ in self.demos.get(key, [])]
        return Counter(counts).most_common(1)[0][0] if counts else 0

    def length(self, key) -> float:
        return float(np.mean([length for _, length, _ in self.demos.get(key, [])] or [0.0]))

    def route(self, start: str, goal: str):
        """Waypoints from start to goal over demonstrated legs (fewest legs), or None."""
        if start == goal:
            return [start]
        prev, queue = {start: None}, deque([start])
        while queue:
            here = queue.popleft()
            for a, b in self.demos:
                if a == here and b not in prev:
                    prev[b] = here
                    if b == goal:
                        path = [b]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    queue.append(b)
        return None

    def reachable(self, start: str) -> list:
        return [w for w in {b for _, b in self.demos} if w != start and self.route(start, w)]


class Navigator:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.active = False
        self.path = []
        self.message = ""

    def start(self, here: str, goal: str, distance: float) -> str | None:
        path = self.policy.route(here, goal)
        if not path or len(path) < 2:
            return f"No recorded walk goes from {here} to {goal}"
        self.active, self.path, self.goal = True, path, goal
        self.message = ""
        self._begin_leg(0, distance)
        return None

    def stop(self):
        self.active = False

    def at(self, here: str, distance: float) -> str | None:
        """You told it where you are."""
        if not self.active:
            return None
        if here == self.goal:
            self.path, self.leg_i = [here], 0
            self.arrived = True
            return None
        return self.start(here, self.goal, distance)

    def _begin_leg(self, i: int, distance: float):
        self.leg_i = i
        self.d0 = distance                # step counter distance at the leg start
        self.done = 0                     # turns made on this leg
        self.arrived = False

    @property
    def key(self):
        return (self.path[self.leg_i], self.path[self.leg_i + 1]) if self.leg_i + 1 < len(self.path) else None

    def on_turn(self, angle: int, distance: float):
        if not self.active or self.arrived or self.key is None:
            return
        side = "left" if angle > 0 else "right"
        kind = side if abs(angle) < 180 else f"around-{side}"
        at, expected = self.policy.turn_at(self.key, self.done)
        if expected is None:
            self.message = f"Unexpected {_word(kind).lower()}: this stretch has no more turns"
            return
        if expected.split("-")[-1] != side and not (expected.startswith("around") and kind.startswith("around")):
            self.message = f"Expected {_word(expected).lower()}, you made a {_word(kind).lower()}. Wrong way?"
            return
        self.done += 1
        self.d0 = distance - at           # landmark: snap to where the demos turned
        self.message = ""

    def update(self, distance: float) -> dict | None:
        if not self.active:
            return None
        if self.arrived or self.key is None:
            return {"arrived": True, "goal": self.goal, "path": self.path, "text": f"You've arrived at {self.goal}"}
        d = distance - self.d0
        action, metres = self.policy.predict(self.key, self.done, d)
        if action == "arrive" and metres <= ARRIVE_M:
            if self.leg_i + 2 >= len(self.path):
                self.arrived = True
                return self.update(distance)
            self._begin_leg(self.leg_i + 1, self.d0 + self.policy.length(self.key))   # the leg ends where the demos' did
            return self.update(distance)
        to = self.path[self.leg_i + 1]
        if action == "arrive":
            text, arrow = f"Go forward {metres:.0f} m to {to}", "forward"
        elif metres <= NOW_M:
            text, arrow = _word(action), action.split("-")[-1]
        else:
            text, arrow = f"Go forward {metres:.0f} m, then {_word(action).lower()}", "forward"
        then = action.split("-")[-1] if action != "arrive" else "forward"
        return {"arrived": False, "goal": self.goal, "path": self.path, "leg": self.leg_i,
                "from": self.path[self.leg_i], "to": to, "action": action, "metres": round(metres, 1),
                "text": text, "arrow": arrow, "then": then, "walked": round(d, 1),
                "leg_length": round(self.policy.length(self.key), 1), "message": self.message}


def _word(action: str) -> str:
    return {"left": "Turn left", "right": "Turn right", "around-left": "Turn around (left)",
            "around-right": "Turn around (right)", "arrive": "Arrive"}.get(action, action)
