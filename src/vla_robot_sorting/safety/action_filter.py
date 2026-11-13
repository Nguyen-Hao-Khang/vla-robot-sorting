"""Runtime safety filter between a policy and the arm (ADR 0004).

Every action a policy produces passes ``ActionSafetyFilter.filter`` before it reaches the
real arm or the simulator. An action is a joint-position target for the arm joints, optionally
followed by a gripper command. The filter applies, in order:

1. format: the action has the configured length and only finite values, else it is rejected;
2. joint limits: a target inside the limit band is accepted; up to ``clamp_tolerance`` past
   the band it is clamped; further out it is rejected;
3. gripper range: the gripper command is clamped to its range;
4. rate limits: the commanded joint velocity and its change per step are limited to the
   configured velocity and acceleration (the target is moved closer to the current state);
   a jump larger than ``reject_step`` from the measured state is rejected outright;
5. workspace: the tool point of the (possibly limited) target must stay inside the
   workspace box, and every joint origin above ``min_link_z``, else it is rejected.

A rejected action is replaced by "hold": the arm keeps its measured position. After
``max_consecutive_rejections`` rejections in a row the filter latches STOPPED and passes no
action until ``reset`` is called. ``check_watchdog`` stops the arm when the policy has been
silent for longer than ``watchdog_timeout_s``; ``emergency_stop`` latches STOPPED at once.
The hardware emergency stop stays in reach regardless (ADR 0004, item 5).

Every decision is recorded, so interventions can be counted per episode
(docs/evaluation-protocol.md).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import panthera

Vec = tuple[float, float, float]


class Status(str, Enum):
    PASS = "pass"
    LIMITED = "limited"  # accepted after clamping or rate limiting
    REJECTED = "rejected"  # replaced by hold
    STOPPED = "stopped"  # latched stop, no motion until reset


@dataclass(frozen=True)
class SafetyConfig:
    lower: tuple[float, ...]  # joint limits (rad), arm joints only
    upper: tuple[float, ...]
    margin: float = 0.05  # rad kept free inside every joint limit
    clamp_tolerance: float = 0.1  # rad past the band that is still clamped instead of rejected
    max_velocity: float = 1.0  # rad/s per joint
    max_acceleration: float = 4.0  # rad/s^2 per joint
    reject_step: float = 0.35  # rad; larger jumps from the measured state are rejected
    control_dt: float = 1.0 / 15.0  # s between actions
    workspace_min: Vec = (0.20, -0.32, 0.0)  # tool point box in the base frame (m)
    workspace_max: Vec = (0.65, 0.32, 0.40)
    min_link_z: float = 0.05  # m; joints 2-6 stay above the mounting surface
    gripper_range: tuple[float, float] | None = None  # (min, max) of the gripper command
    watchdog_timeout_s: float = 0.5
    max_consecutive_rejections: int = 5

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper) or not self.lower:
            raise ValueError("lower and upper must have the same, non-zero length")
        if any(lo + 2 * self.margin >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("every joint needs lower + 2 * margin < upper")
        positive = (self.max_velocity, self.max_acceleration, self.reject_step, self.control_dt)
        if min(positive) <= 0 or self.watchdog_timeout_s <= 0:
            raise ValueError("rates, step, dt and timeout must be positive")
        if any(a >= b for a, b in zip(self.workspace_min, self.workspace_max, strict=True)):
            raise ValueError("workspace_min must be below workspace_max on every axis")

    @property
    def dof(self) -> int:
        return len(self.lower)

    @classmethod
    def load(cls, path: Path, profile: str) -> SafetyConfig:
        """Read one profile ("real", "simulation", ...) from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))[profile]
        tuples = ("lower", "upper", "workspace_min", "workspace_max", "gripper_range")
        return cls(**{k: tuple(v) if k in tuples and v is not None else v for k, v in data.items()})


@dataclass(frozen=True)
class Decision:
    status: Status
    action: tuple[float, ...] | None  # what to send; None means hold the measured position
    reasons: tuple[str, ...] = ()
    time_s: float = 0.0


@dataclass
class ActionSafetyFilter:
    config: SafetyConfig
    link_points: Callable[[Sequence[float]], list[Vec]] = panthera.link_points
    decisions: list[Decision] = field(default_factory=list)
    _stopped: str | None = None
    _rejections: int = 0
    _last_command: tuple[float, ...] | None = None
    _last_velocity: tuple[float, ...] | None = None
    _last_action_time: float | None = None

    # ------------------------------------------------------------------ state
    @property
    def stopped(self) -> bool:
        return self._stopped is not None

    def reset(self) -> None:
        """Re-arm after a stop; called by the operator, never by the policy."""
        self._stopped = None
        self._rejections = 0
        self._last_command = None
        self._last_velocity = None
        self._last_action_time = None

    def emergency_stop(self, reason: str = "emergency stop", time_s: float = 0.0) -> Decision:
        self._stopped = reason
        return self._record(Status.STOPPED, None, (reason,), time_s)

    def check_watchdog(self, time_s: float) -> Decision | None:
        """Stop when no action arrived for longer than the timeout; call from the control loop."""
        if self.stopped or self._last_action_time is None:
            return None
        if time_s - self._last_action_time > self.config.watchdog_timeout_s:
            return self.emergency_stop(
                f"watchdog: no action for {time_s - self._last_action_time:.2f} s", time_s
            )
        return None

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Status}
        for d in self.decisions:
            out[d.status.value] += 1
        return out

    @property
    def interventions(self) -> int:
        """Actions changed or replaced by the filter (limited, rejected or stopped)."""
        return sum(d.status is not Status.PASS for d in self.decisions)

    # ------------------------------------------------------------------ filtering
    def filter(
        self, action: Sequence[float], measured: Sequence[float], time_s: float = 0.0
    ) -> Decision:
        cfg = self.config
        if self.stopped:
            return self._record(Status.STOPPED, None, (f"stopped: {self._stopped}",), time_s)
        self._last_action_time = time_s
        n = cfg.dof
        expected = n + (1 if cfg.gripper_range is not None else 0)
        values = [float(v) for v in action]
        if len(values) != expected or not all(math.isfinite(v) for v in values):
            return self._reject(
                f"malformed action (length {len(values)}, expected {expected})", time_s
            )
        if len(measured) < n or not all(math.isfinite(float(v)) for v in measured[:n]):
            return self._reject("malformed measured state", time_s)
        q_now = [float(v) for v in measured[:n]]
        target = values[:n]
        reasons: list[str] = []

        # 2. joint limits
        for i, (lo, hi) in enumerate(zip(cfg.lower, cfg.upper, strict=True)):
            lo_band, hi_band = lo + cfg.margin, hi - cfg.margin
            if (
                target[i] < lo_band - cfg.clamp_tolerance
                or target[i] > hi_band + cfg.clamp_tolerance
            ):
                return self._reject(
                    f"joint {i + 1} target {target[i]:.3f} rad outside [{lo:.2f}, {hi:.2f}]", time_s
                )
            if not lo_band <= target[i] <= hi_band:
                target[i] = min(max(target[i], lo_band), hi_band)
                reasons.append(f"joint {i + 1} clamped to its limit band")

        # 3. gripper
        gripper: list[float] = []
        if cfg.gripper_range is not None:
            g_lo, g_hi = cfg.gripper_range
            g = values[n]
            if not g_lo <= g <= g_hi:
                g = min(max(g, g_lo), g_hi)
                reasons.append("gripper clamped to its range")
            gripper = [g]

        # 4. rate limits
        steps = [t - q for t, q in zip(target, q_now, strict=True)]
        worst = max(abs(s) for s in steps)
        if worst > cfg.reject_step:
            return self._reject(f"jump of {worst:.3f} rad from the measured state", time_s)
        dt = cfg.control_dt
        v_prev = self._last_velocity or tuple(0.0 for _ in range(n))
        velocity = []
        limited = False
        for i, s in enumerate(steps):
            v = s / dt
            v_lo = max(-cfg.max_velocity, v_prev[i] - cfg.max_acceleration * dt)
            v_hi = min(cfg.max_velocity, v_prev[i] + cfg.max_acceleration * dt)
            if not v_lo <= v <= v_hi:
                v = min(max(v, v_lo), v_hi)
                limited = True
            velocity.append(v)
        if limited:
            target = [q + v * dt for q, v in zip(q_now, velocity, strict=True)]
            reasons.append("velocity or acceleration limited")

        # 5. workspace (checked on the command actually sent)
        points = self.link_points(target)
        tool = points[-1]
        for axis, (low, value, high) in enumerate(
            zip(cfg.workspace_min, tool, cfg.workspace_max, strict=True)
        ):
            if not low <= value <= high:
                return self._reject(
                    f"tool {'xyz'[axis]} = {value:.3f} m outside [{low:.3f}, {high:.3f}]", time_s
                )
        low_link = min(p[2] for p in points[1:-1])
        if low_link < cfg.min_link_z:
            return self._reject(f"a joint would go down to z = {low_link:.3f} m", time_s)

        self._rejections = 0
        self._last_velocity = tuple(velocity)
        command = tuple(target + gripper)
        self._last_command = command
        return self._record(
            Status.LIMITED if reasons else Status.PASS, command, tuple(reasons), time_s
        )

    # ------------------------------------------------------------------ helpers
    def _reject(self, reason: str, time_s: float) -> Decision:
        self._rejections += 1
        self._last_velocity = None  # the arm holds, so it starts from rest next time
        if self._rejections >= self.config.max_consecutive_rejections:
            self._stopped = f"{self._rejections} rejected actions in a row; last: {reason}"
            return self._record(Status.STOPPED, None, (self._stopped,), time_s)
        return self._record(Status.REJECTED, None, (reason,), time_s)

    def _record(
        self, status: Status, action: tuple[float, ...] | None, reasons: tuple[str, ...], t: float
    ) -> Decision:
        decision = Decision(status, action, reasons, t)
        self.decisions.append(decision)
        return decision
