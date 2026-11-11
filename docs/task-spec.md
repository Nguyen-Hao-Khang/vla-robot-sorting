# Task specification

Status: draft. The team freezes this specification before generating the first training dataset. Any later change requires a new dataset version.

## 1. Scene

- A table in front of the robot, matched to the real table once the robot arrives.
- Three fixed containers: left, centre, right.
- Two to four objects per scene, placed at random in the pick area without overlap.

## 2. Objects

| Attribute | Values |
|---|---|
| Shape | cube, cylinder, rectangular block |
| Colour | red, green, blue, yellow |
| Size | 4–6 cm edge, within the gripper opening |

Tetrahedra are excluded ([architecture](architecture.md#5-known-simulator-limits)).

## 3. Commands

A command names one object by colour, shape or both, and one container. Examples:

- "gắp khối màu đỏ bỏ vào khay bên trái"
- "put the blue cylinder in the middle tray"
- "cho cái hình trụ vàng vào khay phải"

The command must identify exactly one object in the scene. The scene generator checks this.

## 4. Success criteria

An episode succeeds when, at its end:

1. the named object rests in the named container;
2. no other object has moved into any container;
3. the safety layer has not stopped the robot.

Time limit: 60 seconds per episode.
