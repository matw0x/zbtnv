from collections import defaultdict, deque
from collections.abc import Sequence

from meshguard.detection.detector import Detection
from meshguard.domain import AnomalySignal


class SignalGate:
    """Turns per-window detections into incident-level signals.

    A service votes anomalous when at least ``min_votes`` of its last ``votes_window``
    windows exceed the threshold. The first vote opens an incident and emits a signal;
    while the incident is open, further anomalous services (typically the blast radius
    of the same root cause) join it silently. The incident closes once every member has
    stayed normal for ``quiet_windows`` consecutive windows.
    """

    def __init__(
        self, *, votes_window: int = 3, min_votes: int = 2, quiet_windows: int = 6
    ) -> None:
        if not 0 < min_votes <= votes_window:
            raise ValueError("min_votes must be in (0, votes_window]")
        self.min_votes = min_votes
        self.quiet_windows = quiet_windows
        self._votes: defaultdict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=votes_window))
        self._members: dict[str, int] = {}

    @property
    def incident_open(self) -> bool:
        return bool(self._members)

    @property
    def members(self) -> frozenset[str]:
        return frozenset(self._members)

    def observe(self, detections: Sequence[Detection]) -> AnomalySignal | None:
        anomalous = sorted(
            (d for d in detections if d.is_anomaly), key=lambda d: d.score, reverse=True
        )
        flagged = {d.window.service for d in anomalous}
        for detection in detections:
            service = detection.window.service
            self._votes[service].append(service in flagged)
            if service in self._members:
                self._members[service] = 0 if service in flagged else self._members[service] + 1
        if self._members and all(q >= self.quiet_windows for q in self._members.values()):
            self._members.clear()

        voted = [d for d in anomalous if sum(self._votes[d.window.service]) >= self.min_votes]
        if not voted:
            return None
        if self._members:
            for detection in voted:
                self._members.setdefault(detection.window.service, 0)
            return None

        primary = voted[0]
        correlated = tuple(d.window.service for d in anomalous if d is not primary)
        for service in (primary.window.service, *correlated):
            self._members[service] = 0
        return AnomalySignal(
            service=primary.window.service,
            window_start=primary.window.start,
            window_end=primary.window.end,
            score=primary.score,
            threshold=primary.threshold,
            deviations=primary.deviations,
            correlated_services=correlated,
        )
