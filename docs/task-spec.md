# Task specification

- Version: 1.2
- Status: frozen on 12 November 2026. Changes require a new version and a new dataset version.
- 1.1 (12 November 2026, before any dataset was recorded): section 5.3 names the language-model template file.
- 1.2 (12 November 2026, before any dataset was recorded): the held-out test commands are written and reviewed by the team instead of people outside it (section 5.3).
- Implementation: `tools/twin/sorting_task.py`

## 1. Sorting criteria

Sorting partitions a set of objects: under a chosen criterion, every object belongs to exactly one class. The project considers five families of criteria and evaluates them at three levels.

| Family | Example classes | Observable by camera | Level |
|---|---|---|---|
| Geometry | shape (cube, cuboid, cylinder, triangular prism); size (small, large) | Yes | **Required** |
| Optical and surface | colour (red, yellow, green, blue) | Yes | **Required** |
| Symbols and identifiers | printed label ("fresh milk" vs "soy milk"), printed number | Yes, if legible at 320×240 | Extension (simulation) |
| Semantics | drinks, writing tools, toys | Indirectly, through appearance | Extension (simulation) |
| Physical and material | mass, hardness, material | No; only appearance cues | Discussed in the report, not evaluated |

A vision-language-action policy sees only RGB images, so mass, hardness and internal material are not observable. The report discusses this limit instead of measuring it.

## 2. Objects

### 2.1 Required set: wooden blocks

| Attribute | Values |
|---|---|
| Shape | cube, cuboid, cylinder, triangular prism |
| Colour | red, yellow, green, blue |
| Size | small, large |

This gives 32 object types. Simulated dimensions (`OBJECT_DIMS` in `sorting_task.py`) keep every object within the gripper opening: small objects about 3 cm across, large objects 4–5 cm. The real set is a commercial wooden building-block kit; its shapes and colours must be checked against this table on delivery, and any block outside it is excluded.

### 2.2 Extension set: household objects

Fresh-milk and soy-milk cartons, water and soft-drink bottles, pens of several colours and types. The simulator already contains textured cartons, bottles and balls (scene levels 6 and 7).

## 3. Containers

Each scene contains 2 or 3 containers drawn at random. Each container has three attributes, and no two containers in a scene share all three:

| Attribute | Values |
|---|---|
| Colour | white, grey, black, brown (none shared with object colours) |
| Outline | square, rectangle, round |
| Depth | shallow (2 cm walls), deep (6 cm walls) |

Containers stand in a row in front of the robot: left and right for two containers, left, middle and right for three. Each holds two objects.

## 4. Scenes

- 2 to 4 objects with distinct (shape, colour, size) triples.
- Objects are placed at random in the pick area, x ∈ [−0.22, 0.22] m, y ∈ [−0.07, 0.01] m, at least 11 cm apart, with random yaw.
- The scene seed determines objects, containers and positions, so every scene can be rebuilt exactly.

## 5. Commands

A command contains one object reference and one container reference, in English or Vietnamese.

### 5.1 Object references

| Form | Meaning | Correct picks | Example |
|---|---|---|---|
| Object command | Names one object with the smallest attribute set that is unique in the scene | Exactly that object | "put the red cylinder in the round tray" |
| Class command | Names one value of one criterion (colour, shape or size) | Any object of that class still on the table | "put a red object in the left tray" |

Object commands test grounding of a specific object. Class commands test sorting: repeating class commands until no member remains on the table sorts the scene under that criterion.

### 5.2 Container references

The container is named by one attribute that is unique among the containers in the scene: position (left, middle, right), colour, outline (square, rectangular, round) or depth (shallow, deep). Deep containers are called "bin" / "thùng", shallow ones "tray" / "khay".

### 5.3 Sources of commands

| Source | Use |
|---|---|
| Templates in `sorting_task.py` (4 object and 3 class templates per language) | Training data, baseline |
| Language-model templates in `configs/command_templates.json` (about 40 per language and form) | Training data |
| Held-out test commands in `configs/test_commands_v1.csv`: one free command per evaluation episode, drafted in varied styles and edited and approved row by row by the team; none reproduces a training template | Test set only |

Examples generated from the templates:

- "move a cylinder to the square tray"
- "pick up the red object and place it in the deep tray"
- "gắp một khối trụ bỏ vào khay xám"
- "đặt khối hộp chữ nhật lớn vào thùng bên phải"

## 6. Success criteria

An episode succeeds when, at its end:

1. an object that the command allows (one object for object commands, any class member for class commands) rests in the named container;
2. no other object has moved into a container;
3. the safety layer has not stopped the robot.

Time limit: 60 seconds per episode.

A sorting run issues class commands for one criterion until the table holds no member of any class. Its score is the fraction of objects that end in the container of their class.
