# ruff: noqa: E501
"""Procedural textured meshes for the twin's household-like objects (no external assets).

Every object gets an OBJ mesh with UV coordinates plus PNG textures drawn with OpenCV:
  carton  - brick-shaped drink cartons (fresh milk / soy milk) and cardboard boxes printed "1"/"2"
  sphere  - tennis ball (yellow-green with a white seam) and mini volleyball (coloured panels)
  bottle  - lathe-shaped PET bottles (water / orange juice / soft drink) with a label band, neck and cap
All designs are generic (no real brands). Meshes are written in the object frame used by
twin_scene: origin at mid-height on the vertical axis, +z up. Files are cached per design;
the generator only needs numpy + OpenCV (both present in the RoboTwin env).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

PX_PER_M = 4000  # texture resolution on box faces (4.5 cm -> 180 px)

# ----------------------------------------------------------------- texture painters (RGB uint8)


def _canvas(
    w: int, h: int, rgb: tuple[int, int, int], noise: float = 0.0, seed: int = 0
) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[:] = rgb
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, (h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def _text(
    img: np.ndarray,
    text: str,
    center: tuple[float, float],
    height_px: float,
    color,
    thickness: int | None = None,
) -> None:
    """Draw text centred at ``center`` with a cap height of about ``height_px`` pixels."""
    import cv2

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = height_px / 22.0
    thick = thickness or max(1, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = round(center[0] - tw / 2)
    y = round(center[1] + th / 2)
    cv2.putText(img, text, (x, y), font, scale, tuple(int(c) for c in color), thick, cv2.LINE_AA)


def _fit_text(img: np.ndarray, text: str, center, max_w: float, max_h: float, color) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    scale = min(max_w / tw, max_h / th)
    _text(img, text, center, th * scale, color, max(1, round(scale * 2)))


def carton_faces(design: str, w: int, d: int, h: int) -> dict[str, np.ndarray]:
    """Face textures (front/back are w x h, left/right d x h, top/bottom w x d)."""
    import cv2

    if design in ("fresh_milk", "soy_milk"):
        fresh = design == "fresh_milk"
        bg = (245, 247, 250) if fresh else (236, 222, 180)
        band = (30, 90, 200) if fresh else (60, 140, 50)
        title = "FRESH MILK" if fresh else "SOY MILK"
        front = _canvas(w, h, bg)
        cv2.rectangle(front, (0, int(h * 0.68)), (w, h), band, -1)
        cv2.rectangle(front, (0, 0), (w, int(h * 0.06)), band, -1)
        if fresh:  # cow spots
            rng = np.random.default_rng(7)
            for _ in range(7):
                cx, cy = rng.integers(0, w), rng.integers(int(h * 0.08), int(h * 0.42))
                cv2.ellipse(
                    front,
                    (int(cx), int(cy)),
                    (int(w * 0.09), int(w * 0.06)),
                    float(rng.integers(0, 180)),
                    0,
                    360,
                    (25, 25, 25),
                    -1,
                )
        else:  # soybeans
            for i in range(5):
                cv2.circle(
                    front,
                    (int(w * (0.18 + 0.16 * i)), int(h * 0.3)),
                    max(2, int(w * 0.06)),
                    (95, 160, 60),
                    -1,
                )
                cv2.circle(
                    front,
                    (int(w * (0.18 + 0.16 * i)), int(h * 0.3)),
                    max(1, int(w * 0.03)),
                    (200, 220, 120),
                    -1,
                )
        _fit_text(front, title, (w / 2, h * 0.55), w * 0.8, h * 0.12, band)
        _fit_text(front, "180 ml", (w / 2, h * 0.84), w * 0.5, h * 0.07, (255, 255, 255))
        side = _canvas(d, h, bg)
        cv2.rectangle(side, (0, int(h * 0.68)), (d, h), band, -1)
        for i in range(8):
            y = int(h * (0.12 + 0.06 * i))
            cv2.line(
                side,
                (int(d * 0.12), y),
                (int(d * (0.88 - 0.05 * (i % 3))), y),
                (120, 120, 120),
                max(1, h // 160),
            )
        top = _canvas(w, d, bg)
        cv2.circle(top, (int(w * 0.75), int(d * 0.5)), max(2, int(d * 0.12)), (190, 190, 190), -1)
        bottom = _canvas(w, d, tuple(int(c * 0.85) for c in bg))
        return {
            "front": front,
            "back": front.copy(),
            "left": side,
            "right": side,
            "top": top,
            "bottom": bottom,
        }
    if design in ("box_1", "box_2"):
        digit = design[-1]
        kraft = (196, 150, 98)
        faces = {}
        for name, (fw, fh) in {
            "front": (w, h),
            "back": (w, h),
            "left": (d, h),
            "right": (d, h),
        }.items():
            face = _canvas(fw, fh, kraft, noise=6, seed=hash(name) % 100)
            cv2.rectangle(face, (0, 0), (fw - 1, fh - 1), (150, 110, 70), max(1, fw // 60))
            _fit_text(face, digit, (fw / 2, fh * 0.5), fw * 0.6, fh * 0.62, (20, 20, 20))
            faces[name] = face
        top = _canvas(w, d, kraft, noise=6, seed=11)
        cv2.rectangle(top, (int(w * 0.4), 0), (int(w * 0.6), d), (215, 185, 135), -1)  # tape
        _fit_text(top, digit, (w * 0.2, d * 0.5), w * 0.25, d * 0.5, (20, 20, 20))
        faces["top"] = top
        faces["bottom"] = _canvas(w, d, (170, 128, 82), noise=5, seed=12)
        return faces
    raise ValueError(f"unknown carton design {design!r}")


def ball_texture(design: str, w: int = 512, h: int = 256) -> np.ndarray:
    """Equirectangular texture: u = longitude, v = latitude (row 0 = north pole)."""
    import cv2

    lon = np.linspace(0, 2 * math.pi, w, endpoint=False)[None, :]
    lat = np.linspace(math.pi / 2, -math.pi / 2, h)[:, None]
    if design == "tennis":
        img = _canvas(w, h, (205, 225, 60), noise=10, seed=3)
        seam_lat = 0.62 * np.sin(2 * lon)
        mask = np.abs(lat - seam_lat) < 0.05
        img[mask] = (245, 245, 235)
        return img
    if design == "volleyball":
        colors = [(245, 245, 245), (240, 200, 30), (30, 80, 190)]
        img = np.zeros((h, w, 3), np.uint8)
        sector = (lon // (2 * math.pi / 3)).astype(int)  # 3 panel groups
        band = np.clip(((lat + math.pi / 2) / math.pi * 3).astype(int), 0, 2)
        idx = (sector + band) % 3
        for i, c in enumerate(colors):
            img[np.broadcast_to(idx == i, (h, w))] = c
        seams = (np.abs(lon % (2 * math.pi / 3)) < 0.02) | (
            np.abs(np.abs(lat) - math.pi / 6) < 0.015
        )
        img[np.broadcast_to(seams, (h, w))] = (60, 60, 60)
        return cv2.GaussianBlur(img, (3, 3), 0)
    raise ValueError(f"unknown ball design {design!r}")


def bottle_texture(
    design: str, profile: list[tuple[float, float]], w: int = 512, h: int = 512
) -> np.ndarray:
    """Texture over the whole lathe height (row 0 = top of the cap)."""
    import cv2

    z0, z1 = profile[0][1], profile[-1][1]

    def row(z: float) -> int:
        return round((z1 - z) / (z1 - z0) * (h - 1))

    palette = {  # body, cap, label band, label text, title
        "water": ((190, 225, 245), (40, 110, 220), (255, 255, 255), (30, 90, 200), "PURE WATER"),
        "juice": ((245, 150, 30), (210, 40, 40), (255, 230, 120), (200, 60, 10), "ORANGE JUICE"),
        "soda": ((90, 40, 20), (200, 30, 30), (200, 30, 30), (255, 255, 255), "SOFT DRINK"),
    }
    body, cap, label, text_color, title = palette[design]
    img = _canvas(w, h, body)
    cap_bottom = row(profile[-3][1])
    img[:cap_bottom] = cap
    lz0, lz1 = z0 + 0.25 * (z1 - z0), z0 + 0.55 * (z1 - z0)
    cv2.rectangle(img, (0, row(lz1)), (w, row(lz0)), label, -1)
    for k in range(2):  # twice around the bottle
        _fit_text(
            img,
            title,
            (w * (0.25 + 0.5 * k), (row(lz1) + row(lz0)) / 2),
            w * 0.3,
            (row(lz0) - row(lz1)) * 0.5,
            text_color,
        )
    for k in range(6):  # ribs on the lower body
        y = row(z0 + (0.06 + 0.03 * k) * (z1 - z0))
        cv2.line(img, (0, y), (w, y), tuple(int(c * 0.9) for c in body), 2)
    return img


# ----------------------------------------------------------------- meshes (OBJ with UVs)


def box_obj(hx: float, hy: float, hz: float, stem: str) -> str:
    """Six quads, one material per face (<stem>_<face>), UVs upright when seen from outside."""
    faces = {
        "front": ((0, -1, 0), (0, 0, 1)),
        "back": ((0, 1, 0), (0, 0, 1)),
        "right": ((1, 0, 0), (0, 0, 1)),
        "left": ((-1, 0, 0), (0, 0, 1)),
        "top": ((0, 0, 1), (0, 1, 0)),
        "bottom": ((0, 0, -1), (0, -1, 0)),
    }
    half = np.array([hx, hy, hz])
    lines = [f"mtllib {stem}.mtl"]
    vi = 1
    for name, (n, up) in faces.items():
        n, up = np.array(n, float), np.array(up, float)
        right = np.cross(up, n)
        c = n * half
        hr, hu = float(np.abs(right) @ half), float(np.abs(up) @ half)
        corners = [
            c - right * hr - up * hu,
            c + right * hr - up * hu,
            c + right * hr + up * hu,
            c - right * hr + up * hu,
        ]
        lines.append(f"usemtl {stem}_{name}")
        for p in corners:
            lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        for uv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            lines.append(f"vt {uv[0]} {uv[1]}")
        lines.append(f"f {vi}/{vi} {vi + 1}/{vi + 1} {vi + 2}/{vi + 2}")
        lines.append(f"f {vi}/{vi} {vi + 2}/{vi + 2} {vi + 3}/{vi + 3}")
        vi += 4
    return "\n".join(lines) + "\n"


def lathe_obj(profile: list[tuple[float, float]], stem: str, segments: int = 32) -> str:
    """Surface of revolution of (radius, z) points (bottom to top) with end caps; u = angle, v = z."""
    z0, z1 = profile[0][1], profile[-1][1]
    lines = [f"mtllib {stem}.mtl", f"usemtl {stem}_skin"]
    verts, uvs, faces = [], [], []
    for r, z in profile:
        for j in range(segments + 1):
            a = 2 * math.pi * j / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append((j / segments, (z - z0) / (z1 - z0)))
    ring = segments + 1
    for k in range(len(profile) - 1):
        for j in range(segments):
            a, b = k * ring + j, k * ring + j + 1
            c, d = (k + 1) * ring + j + 1, (k + 1) * ring + j
            faces += [(a, b, c), (a, c, d)]
    for z_index, sign in ((0, -1), (len(profile) - 1, 1)):  # flat end caps
        r, z = profile[z_index]
        if r <= 1e-9:
            continue
        center = len(verts)
        verts.append((0.0, 0.0, z))
        uvs.append((0.5, 0.0 if sign < 0 else 1.0))
        base = z_index * ring
        for j in range(segments):
            p, q = base + j, base + j + 1
            faces.append((center, p, q) if sign > 0 else (center, q, p))
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def sphere_obj(radius: float, stem: str, rings: int = 16, segments: int = 32) -> str:
    profile = [
        (radius * math.cos(lat), radius * math.sin(lat))
        for lat in (math.pi * (k / rings - 0.5) for k in range(rings + 1))
    ]
    return lathe_obj(profile, stem, segments)


def bottle_profile(radius: float, half_height: float) -> list[tuple[float, float]]:
    """Body, shoulder, neck and cap of a small PET bottle (bottom to top), origin at mid-height."""
    H = 2 * half_height
    z = lambda f: -half_height + f * H  # noqa: E731
    neck = radius * 0.55
    return [
        (radius * 0.92, z(0.0)),
        (radius, z(0.03)),
        (radius, z(0.62)),
        (radius * 0.8, z(0.72)),
        (neck, z(0.80)),
        (neck, z(0.86)),
        (neck * 1.12, z(0.86)),
        (neck * 1.12, z(1.0)),
    ]


# ----------------------------------------------------------------- files


def _save_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]))


def _mtl(stem_textures: dict[str, str]) -> str:
    return "".join(
        f"newmtl {name}\nKd 1 1 1\nmap_Kd {png}\n\n" for name, png in stem_textures.items()
    )


def write_asset(kind: str, design: str, dims: tuple[float, ...], cache_dir: Path) -> Path:
    """Write (or reuse) the OBJ/MTL/PNG files for one design; returns the OBJ path."""
    stem = f"{kind}_{design}_" + "_".join(str(round(v * 1000)) for v in dims)
    folder = cache_dir / "assets" / stem
    obj_path = folder / f"{stem}.obj"
    if obj_path.is_file():
        return obj_path
    folder.mkdir(parents=True, exist_ok=True)
    textures: dict[str, str] = {}
    if kind == "carton":
        hx, hy, hz = dims
        size = {
            "w": round(2 * hx * PX_PER_M),
            "d": round(2 * hy * PX_PER_M),
            "h": round(2 * hz * PX_PER_M),
        }
        for face, img in carton_faces(design, size["w"], size["d"], size["h"]).items():
            png = f"{stem}_{face}.png"
            _save_png(folder / png, img)
            textures[f"{stem}_{face}"] = png
        obj = box_obj(hx, hy, hz, stem)
    elif kind == "sphere":
        (radius,) = dims
        png = f"{stem}.png"
        _save_png(folder / png, ball_texture(design))
        textures[f"{stem}_skin"] = png
        obj = sphere_obj(radius, stem)
    elif kind == "bottle":
        radius, half_height = dims
        profile = bottle_profile(radius, half_height)
        png = f"{stem}.png"
        _save_png(folder / png, bottle_texture(design, profile))
        textures[f"{stem}_skin"] = png
        obj = lathe_obj(profile, stem)
    else:
        raise ValueError(f"unknown asset kind {kind!r}")
    (folder / f"{stem}.mtl").write_text(_mtl(textures))
    obj_path.write_text(obj)
    return obj_path


PAINTERS: dict[str, Callable] = {
    "carton": carton_faces,
    "sphere": ball_texture,
    "bottle": bottle_texture,
}
