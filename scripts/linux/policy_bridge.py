#!/usr/bin/env python3
# ruff: noqa: E501
"""Run a trained LeRobot policy (ACT / SmolVLA) against the digital twin over a local socket.

The policy and the simulator live in different Python environments (LeRobot vs RoboTwin), so
the policy runs as a small server and the twin is the client, exactly as a real robot would be:
  server (LeRobot env):  python policy_bridge.py serve --checkpoint outputs/train/act/checkpoints/last/pretrained_model
  client (twin console): command 'policy <English instruction>' with --policy-server 127.0.0.1:5555

Protocol (localhost only): each message is a 4-byte big-endian length followed by UTF-8 JSON.
Request  {"state": [D floats], "images": {"wrist": <base64 JPEG>, ...}, "task": "..."}
Response {"action": [D floats]} or {"error": "..."}; {"reset": true} clears the policy queue.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import sys
from pathlib import Path

MAX_MESSAGE = 32 * 1024 * 1024


def send_message(sock: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> dict:
    (size,) = struct.unpack(">I", _recv_exact(sock, 4))
    if size > MAX_MESSAGE:
        raise ValueError(f"message too large ({size} bytes)")
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


def parse_address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the policy server must be on localhost")
    return host, int(port)


class PolicyClient:
    """Twin side: encode an observation, receive one joint-position action."""

    def __init__(self, address: str, timeout_s: float = 30.0) -> None:
        self.sock = socket.create_connection(parse_address(address), timeout=timeout_s)

    def reset(self) -> None:
        send_message(self.sock, {"reset": True})
        recv_message(self.sock)

    def act(self, state, images: dict, task: str) -> list[float]:
        import cv2

        encoded = {}
        for name, rgb in images.items():
            ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for camera {name}")
            encoded[name] = base64.b64encode(jpeg.tobytes()).decode()
        send_message(
            self.sock, {"state": [float(v) for v in state], "images": encoded, "task": task}
        )
        reply = recv_message(self.sock)
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return [float(v) for v in reply["action"]]

    def close(self) -> None:
        self.sock.close()


class PolicyRunner:
    """Server side (LeRobot env): load a checkpoint and map observations to actions."""

    def __init__(
        self, checkpoint: Path, device: str = "auto", camera_map: dict[str, str] | None = None
    ) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.device = device
        self.device = torch.device(device)
        self.policy = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
        # SmolVLA pads "empty_camera_*" inputs itself; camera_map maps twin camera -> policy camera.
        self.image_keys = [
            k
            for k in cfg.input_features
            if k.startswith("observation.images.") and "empty_camera" not in k
        ]
        self.camera_map = camera_map or {}
        self.policy_type = cfg.type

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state: list[float], images: dict[str, bytes], task: str) -> list[float]:
        import io

        import numpy as np
        from lerobot.utils.control_utils import predict_action
        from PIL import Image

        observation = {"observation.state": np.asarray(state, np.float32)}
        renamed = {self.camera_map.get(name, name): data for name, data in images.items()}
        for key in self.image_keys:
            camera = key.removeprefix("observation.images.")
            if camera not in renamed:
                raise KeyError(f"missing camera '{camera}' (policy expects {self.image_keys})")
            with Image.open(io.BytesIO(renamed[camera])) as img:
                observation[key] = np.asarray(img.convert("RGB"))
        action = predict_action(
            observation, self.policy, self.device, self.pre, self.post, False, task=task
        )
        return [float(v) for v in action.reshape(-1).tolist()]


def parse_camera_map(text: str) -> dict[str, str]:
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    if any(len(p) != 2 or not p[0].strip() or not p[1].strip() for p in pairs):
        raise ValueError("camera map must look like wrist=camera1,workspace=camera2")
    return {a.strip(): b.strip() for a, b in pairs}


def serve(runner: PolicyRunner, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        print(
            f"POLICY_SERVER_READY type={runner.policy_type} cameras={runner.image_keys} port={port}",
            flush=True,
        )
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        request = recv_message(conn)
                    except (ConnectionError, struct.error):
                        break
                    try:
                        if request.get("reset"):
                            runner.reset()
                            send_message(conn, {"ok": True})
                            continue
                        images = {
                            k: base64.b64decode(v) for k, v in request.get("images", {}).items()
                        }
                        action = runner.act(request["state"], images, request.get("task", ""))
                        send_message(conn, {"action": action})
                    except Exception as exc:  # report to the client, keep serving
                        send_message(conn, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve a LeRobot checkpoint (run in the LeRobot env)")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--port", type=int, default=5555)
    s.add_argument("--device", default="auto")
    s.add_argument(
        "--camera-map",
        default="",
        help="twin=policy camera names, e.g. wrist=camera1,workspace=camera2 (SmolVLA fine-tuned from smolvla_base)",
    )
    args = parser.parse_args(argv)
    serve(PolicyRunner(args.checkpoint, args.device, parse_camera_map(args.camera_map)), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
