"""Demonstration recorder (raw format), task sampling and the policy socket protocol (no GPU)."""

from __future__ import annotations

import importlib.util
import json
import random
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/twin"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/twin/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rec = _load("twin_record")
bridge = _load("policy_bridge")
conv = _load("lerobot_convert")
ts = rec.ts
TABLE = 0.74


def _xyz(level):
    return {
        n: [x, y, TABLE + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()
    }


def test_instruction_templates() -> None:
    rng = random.Random(0)
    text = rec.instruction_for("red cube", "yellow tray", "tray", rng)
    assert "red cube" in text and "yellow tray" in text
    assert rec.instruction_for("ball", "staging pad", "pad", rng).split()[0] in ("put", "move")


def test_sample_task_respects_capacity_location_and_graspability() -> None:
    level = ts.LEVELS[7]
    xyz = _xyz(level)
    xyz["tennis ball"] = [ts.DEEP_BIN.xy[0] - 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.028]
    xyz["volleyball"] = [ts.DEEP_BIN.xy[0] + 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.028]
    rng = random.Random(3)
    seen = {rec.sample_task(level, xyz, TABLE, rng) for _ in range(300)}
    assert all(task[0] not in ("tennis ball", "volleyball") for task in seen)  # stuck in the bin
    assert all(task[1] != "deep bin" for task in seen)  # full
    assert ("water bottle", "wide tray") in seen
    l1 = ts.LEVELS[1]
    tray = l1.containers[0]
    full = {"red cube": [tray.xy[0] - 0.06, tray.xy[1], TABLE + 0.02]}
    rng2 = random.Random(0)
    assert (
        rec.sample_task(l1, full, TABLE, rng2) is None
        or rec.sample_task(l1, full, TABLE, rng2)[1] != tray.name
    )


def test_derive_actions_is_next_state() -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    actions = rec.derive_actions(states)
    assert np.array_equal(actions[:3], states[1:]) and np.array_equal(actions[3], states[3])
    assert rec.derive_actions(np.zeros((0, 3), np.float32)).shape == (0, 3)


class FakeTask:
    def __init__(self):
        self.steps = 0

    def _step_scene(self):
        self.steps += 1


def test_frame_recorder_samples_at_fixed_sim_rate() -> None:
    counter = {"n": 0}

    def grab():
        counter["n"] += 1
        return np.full(7, counter["n"], np.float32), {"wrist": np.zeros((4, 4, 3), np.uint8)}

    task = FakeTask()
    recorder = rec.FrameRecorder(grab, step_dt=1 / 250, fps=15)
    recorder.attach(task)
    for _ in range(50):
        task._step_scene()  # inactive: nothing recorded
    assert recorder.states == []
    recorder.start()
    for _ in range(250):  # one simulated second
        task._step_scene()
    recorder.stop()
    states, actions, times = recorder.arrays()
    assert len(states) == 16 and len(recorder.images["wrist"]) == 16  # t=0 plus 15 samples
    assert np.allclose(np.diff(times), 1 / 15, atol=1 / 250 + 1e-6)
    assert np.array_equal(actions[:-1], states[1:])


def _write(root, index, frames=5, success=True, cams=("wrist", "workspace")):
    states = np.random.default_rng(index).normal(size=(frames, 7)).astype(np.float32)
    images = {c: [np.zeros((24, 32, 3), np.uint8) for _ in range(frames)] for c in cams}
    saved = []

    def save(path, rgb):
        path.write_bytes(b"jpg")
        saved.append(path)

    folder = rec.write_episode(
        root,
        index,
        states,
        rec.derive_actions(states),
        np.arange(frames) / 15,
        images,
        {"task": "put the red cube in the yellow tray", "fps": 15, "success": success},
        save,
    )
    return folder, saved


def test_write_episode_layout_and_meta(tmp_path) -> None:
    folder, saved = _write(tmp_path, 0)
    meta = json.loads((folder / "meta.json").read_text())
    assert (
        meta["frames"] == 5 and meta["state_dim"] == 7 and meta["cameras"] == ["workspace", "wrist"]
    )
    assert meta["format"] == rec.RAW_FORMAT_VERSION and len(saved) == 10
    assert np.load(folder / "states.npy").shape == (5, 7)
    with pytest.raises(FileExistsError):
        _write(tmp_path, 0)
    assert rec.next_episode_index(tmp_path) == 1


def test_converter_selects_episodes_and_checks_frames(tmp_path) -> None:
    _write(tmp_path, 0)
    _write(tmp_path, 1, success=False)
    assert [f.name for f in conv.episode_folders(tmp_path, False)] == ["episode_000000"]
    assert len(conv.episode_folders(tmp_path, True)) == 2
    _meta, states, actions, frames = conv.load_episode(tmp_path / "episode_000000")
    assert states.shape == actions.shape == (5, 7) and len(frames["wrist"]) == 5
    (tmp_path / "episode_000000" / "wrist" / "000004.jpg").unlink()
    with pytest.raises(ValueError):
        conv.load_episode(tmp_path / "episode_000000")
    feats = conv.build_features(7, ["workspace", "wrist"], (240, 320), True)
    assert feats["observation.state"]["names"][-1] == "gripper"
    assert feats["observation.images.wrist"]["dtype"] == "video" and feats[
        "observation.images.wrist"
    ]["shape"] == (240, 320, 3)


def test_policy_message_framing_and_guards() -> None:
    a, b = socket.socketpair()
    bridge.send_message(a, {"state": [1.0, 2.0], "images": {}})
    assert bridge.recv_message(b) == {"state": [1.0, 2.0], "images": {}}
    a.close()
    with pytest.raises(ConnectionError):
        bridge.recv_message(b)
    b.close()
    with pytest.raises(ValueError):
        bridge.parse_address("10.0.0.5:5555")
    assert bridge.parse_address("127.0.0.1:5555") == ("127.0.0.1", 5555)
    assert bridge.parse_camera_map("wrist=camera1, workspace=camera2") == {
        "wrist": "camera1",
        "workspace": "camera2",
    }
    with pytest.raises(ValueError):
        bridge.parse_camera_map("wrist")


def test_policy_server_reset_act_and_error_replies() -> None:
    class FakeRunner:
        policy_type, image_keys = "fake", []

        def reset(self):
            pass

        def act(self, state, images, task):
            if task == "boom":
                raise RuntimeError("bad input")
            return [state[0] * 2]

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=bridge.serve, args=(FakeRunner(), port), daemon=True).start()
    sock = None
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            threading.Event().wait(0.05)
    bridge.send_message(sock, {"reset": True})
    assert bridge.recv_message(sock) == {"ok": True}
    bridge.send_message(sock, {"state": [1.5], "images": {}, "task": "go"})
    assert bridge.recv_message(sock) == {"action": [3.0]}
    bridge.send_message(sock, {"state": [1.0], "images": {}, "task": "boom"})
    assert "bad input" in bridge.recv_message(sock)["error"]
    sock.close()
