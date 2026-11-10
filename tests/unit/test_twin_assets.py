"""Procedural textured meshes for look-alike objects (geometry checks; textures need OpenCV)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/twin"))
spec = importlib.util.spec_from_file_location("twin_assets", ROOT / "tools/twin/twin_assets.py")
ta = importlib.util.module_from_spec(spec)
sys.modules["twin_assets"] = ta
spec.loader.exec_module(ta)
ts = importlib.import_module("twin_scene")


def parse(obj_text: str):
    verts, uvs, faces = [], [], []
    for line in obj_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(v) for v in parts[1:4]])
        elif parts[0] == "vt":
            uvs.append([float(v) for v in parts[1:3]])
        elif parts[0] == "f":
            faces.append([tuple(int(i) - 1 for i in p.split("/")) for p in parts[1:4]])
    return np.array(verts), np.array(uvs), faces


def outward_ratio(verts, faces) -> float:
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
    good = total = 0
    for face in faces:
        a, b, c = (verts[i[0]] for i in face)
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12:
            continue
        total += 1
        good += np.dot(normal, (a + b + c) / 3 - centre) > 0
    return good / total


def test_box_faces_point_outward_and_text_is_upright() -> None:
    hx, hy, hz = ts.CARTON_HALF
    verts, uvs, faces = parse(ta.box_obj(hx, hy, hz, "c"))
    assert np.allclose(np.abs(verts).max(axis=0), [hx, hy, hz])
    assert outward_ratio(verts, faces) == 1.0
    # front face (normal -y): u grows with +x and v grows with +z -> readable from outside
    front = faces[:2]
    idx = {i for f in front for i in f}
    for vi, ti in idx:
        x, _, z = verts[vi]
        u, v = uvs[ti]
        assert (u > 0.5) == (x > 0) and (v > 0.5) == (z > 0)


def test_sphere_and_bottle_meshes_closed_outward_with_valid_uvs() -> None:
    verts, uvs, faces = parse(ta.sphere_obj(0.024, "s"))
    assert np.allclose(np.linalg.norm(verts, axis=1), 0.024, atol=1e-6)
    assert outward_ratio(verts, faces) > 0.99
    profile = ta.bottle_profile(ts.BOTTLE_R, ts.BOTTLE_HALF_H)
    zs = [z for _, z in profile]
    assert (
        zs == sorted(zs)
        and np.isclose(zs[0], -ts.BOTTLE_HALF_H)
        and np.isclose(zs[-1], ts.BOTTLE_HALF_H)
    )
    assert max(r for r, _ in profile) == ts.BOTTLE_R
    verts, uvs, faces = parse(ta.lathe_obj(profile, "b"))
    # outward-oriented closed surface: positive signed volume close to the lathe volume
    volume = (
        sum(np.dot(verts[f[0][0]], np.cross(verts[f[1][0]], verts[f[2][0]])) for f in faces) / 6
    )
    expected = sum(
        np.pi * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3
        for (r0, z0), (r1, z1) in itertools.pairwise(profile)
    )
    assert volume > 0 and abs(volume - expected) / expected < 0.02
    assert uvs.min() >= 0 and uvs.max() <= 1


def test_bottle_collision_matches_visual_height() -> None:
    parts = ts.bottle_collision(ts.BOTTLE_R, ts.BOTTLE_HALF_H)
    bottom = min(c - h for c, _, h in parts)
    top = max(c + h for c, _, h in parts)
    assert np.isclose(bottom, -ts.BOTTLE_HALF_H) and np.isclose(top, ts.BOTTLE_HALF_H)
    assert max(r for _, r, _ in parts) == ts.BOTTLE_R


def test_write_asset_creates_and_reuses_files(tmp_path) -> None:
    pytest.importorskip("cv2")
    obj = ta.write_asset("carton", "soy_milk", ts.CARTON_HALF, tmp_path)
    assert obj.is_file() and (obj.parent / f"{obj.stem}.mtl").is_file()
    assert len(list(obj.parent.glob("*.png"))) == 6
    assert ta.write_asset("carton", "soy_milk", ts.CARTON_HALF, tmp_path) == obj
    for kind, design, dims in (("sphere", "tennis", (0.024,)), ("bottle", "juice", (0.021, 0.065))):
        path = ta.write_asset(kind, design, dims, tmp_path)
        assert path.is_file() and "map_Kd" in (path.parent / f"{path.stem}.mtl").read_text()


def test_look_alike_textures_differ() -> None:
    pytest.importorskip("cv2")
    fresh = ta.carton_faces("fresh_milk", 90, 76, 200)["front"]
    soy = ta.carton_faces("soy_milk", 90, 76, 200)["front"]
    one = ta.carton_faces("box_1", 100, 100, 100)["front"]
    two = ta.carton_faces("box_2", 100, 100, 100)["front"]
    assert np.abs(fresh.astype(int) - soy.astype(int)).mean() > 20
    assert np.abs(one.astype(int) - two.astype(int)).mean() > 2
    assert (
        np.abs(
            ta.ball_texture("tennis").astype(int) - ta.ball_texture("volleyball").astype(int)
        ).mean()
        > 30
    )


def test_unknown_designs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ta.write_asset("teapot", "x", (0.1,), tmp_path)
