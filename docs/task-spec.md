# Task specification

- Version: 1.5
- Status: frozen on 12 November 2026. Changes require a new version and a new dataset version.
- 1.1 (12 November 2026, before any dataset was recorded): section 5.3 names the language-model template file.
- 1.2 (12 November 2026, before any dataset was recorded): the held-out test commands are written and reviewed by the team instead of people outside it (section 5.3).
- 1.3 (13 November 2026, after an 80-episode pilot recording of blocks only): household objects that differ by printed identity and semantic category enter the simulated task and the training data (sections 1, 2.2, 4, 5.1).
- 1.4 (13 November 2026): a third family models the real 30-piece wooden kit, whose colours and shapes differ from the block set (sections 2.1, 4). The block set, its evaluation scenes and its test commands are unchanged.
- 1.5 (13 November 2026, before the first recording with the kit and household families): kit sizes corrected from a closer reading of the product photo and half cylinders added (section 2.1.1); household bottles come in two sizes, are grasped by the neck and cap, and drink packages are full on the real arm; simulated bottles stay light with a grippier surface, decided after a smoke test on the same day and before any recording (sections 2.2, 4). Six rows of the household test commands were re-derived for the changed scenes (section 5.3).
- Implementation: `tools/twin/sorting_task.py`

## 1. Sorting criteria

Sorting partitions a set of objects: under a chosen criterion, every object belongs to exactly one class. The project considers five families of criteria and evaluates them at three levels.

| Family | Example classes | Observable by camera | Level |
|---|---|---|---|
| Geometry | shape (cube, cuboid, cylinder, triangular prism); size (small, large) | Yes | **Required** |
| Optical and surface | colour (red, yellow, green, blue) | Yes | **Required** |
| Symbols and identifiers | printed label ("fresh milk" vs "soy milk"), printed number ("box 1" vs "box 2") | Yes, if legible at 320×240 | **Extension**: simulation, real robot if time permits |
| Semantics | milk carton, bottle, numbered box | Indirectly, through appearance | **Extension**: simulation, real robot if time permits |
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

### 2.1.1 Real kit

The real arm sorts pieces of a commercial 30-piece wooden kit sold for shape and colour recognition. Its pieces do not cover the block set of section 2.1: cubes, cylinders and large pieces are mostly unpainted, the painted pieces come in six colours, and cuboids and triangular prisms come in two sizes. Read from the product photo (30 pieces):

| Shape | In the kit | Simulated colours | Simulated size |
|---|---|---|---|
| cube, small | 4 natural wood | natural wood | 3 cm |
| cylinder, small | 4 natural wood | natural wood | 3 cm diameter, 3 cm tall |
| cuboid, small | 2 each blue, red, green, purple | red, orange, yellow, green, blue, purple | 3 × 1.5 × 3 cm, standing |
| cuboid, large | 2 green, 2 natural wood | green, natural wood | 6 × 3 × 1.5 cm bar, lying |
| triangular prism, small | 2 orange, 2 yellow | red, orange, yellow, green, blue, purple | 4.2 cm sides, 3 cm tall |
| triangular prism, large | 2 natural wood | natural wood | 5.2 cm sides, 3 cm tall |
| half cylinder, small | 1 orange, 1 yellow | red, orange, yellow, green, blue, purple | 4 cm across, 3 cm thick, 2 cm tall, on its flat face |
| arch | 2 natural wood | not simulated | — |

Small cuboids, small prisms and half cylinders take all six colours in simulation, although the kit paints each of them in only two to four; the extra combinations add variety to the training data and never occur on the real table. The kit's prisms are right-angled and lie on their base, so the real gripper closes on the triangular faces 3 cm apart; the simulated prism is equilateral and stands on a triangular face, and the large one is cut to 5.2 cm sides so that its grasp width (4.5 cm) stays within the range the simulated gripper holds reliably. Arches (6 × 3 × 3 cm with a half-round cut-out) are hard to grasp from above and are left out of sorting runs. Sizes are estimates and are replaced by measured values when the kit arrives. Colour words: natural wood ("gỗ mộc"), orange ("cam"), purple ("tím"); half cylinder is "khối bán trụ".

The block set stays the main simulated family: it varies shape, colour and size independently, which the kit cannot, and its evaluation scenes and test commands are fixed.

### 2.2 Extension set: household objects

Six printed packages in three semantic categories. Each category holds a look-alike pair that only the print tells apart.

| Item | Category | Vietnamese | Simulated size |
|---|---|---|---|
| fresh milk carton | milk carton | hộp sữa tươi | 4.5 × 3.8 × 10 cm |
| soy milk carton | milk carton | hộp sữa đậu nành | 4.5 × 3.8 × 10 cm |
| box number 1 | numbered box | hộp số 1 | 5 cm cube |
| box number 2 | numbered box | hộp số 2 | 5 cm cube |
| water bottle | bottle | chai nước suối | small: 5.8 cm diameter, 16 cm tall; large: 6.5 cm diameter, 20.5 cm tall |
| soft drink bottle | bottle | chai nước ngọt | same two sizes |

Bottles are wider than the gripper opens (5.8 cm), so they are grasped from above by the neck and cap, which are about 3 cm across on every common bottle (`BOTTLE_CAP_R` in `twin_scene.py`). Each bottle in a scene is small or large; in a quarter of the scenes that hold a bottle, the same bottle also appears in its other size, so size alone decides which bottle a command means. Drink packages are full on the real arm (team decision): 0.20 kg per carton, about 0.37 kg per small and 0.53 kg per large bottle; the numbered boxes weigh 0.05 kg. In simulation, bottles weigh 0.12 and 0.15 kg and their surface has friction 2.0: with full-bottle masses every scripted grasp by the cap slipped out of the tilted gripper (13 November 2026). Whether the real gripper holds full bottles by the cap is tested on the arm.

The simulated packages carry generic prints (no brands, `tools/twin/twin_assets.py`). The real ones must keep the same look-alike pairs (two milk cartons of the same size, two boxes differing only by a printed number, two kinds of bottle in the same two sizes); cartons and boxes must be at most 4.5 cm wide where the gripper closes. Bottle sizes and masses are estimates for 350 ml and 500 ml bottles and are replaced by measured values when the bottles are bought.

## 3. Containers

Each scene contains 2 or 3 containers drawn at random. Each container has three attributes, and no two containers in a scene share all three:

| Attribute | Values |
|---|---|
| Colour | white, grey, black, brown (none shared with object colours) |
| Outline | square, rectangle, round |
| Depth | shallow (2 cm walls), deep (6 cm walls) |

Containers stand in a row in front of the robot: left and right for two containers, left, middle and right for three. Each holds two objects.

## 4. Scenes

A scene holds objects of one family: blocks (section 2.1), real-kit pieces (section 2.1.1) or household items (section 2.2). Household scenes always contain at least one look-alike pair, so the object's identity, not only its category, decides which object a command means. When recording training data, 40% of the layouts hold blocks, 30% real-kit pieces (`--kit-share`) and 30% household items (`--household-share`).


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

In household scenes the same two forms apply with other attributes. Object commands name the item ("put the soy milk carton in the left tray") or, when it is the only one of its category, the category ("put the bottle in the deep bin"). When the same bottle stands there in both sizes, the size is added ("put the large water bottle in the deep bin", "chai nước suối lớn"). Class commands name a category, and either item of the category is a correct pick ("put a milk carton in the round tray").

### 5.2 Container references

The container is named by one attribute that is unique among the containers in the scene: position (left, middle, right), colour, outline (square, rectangular, round) or depth (shallow, deep). Deep containers are called "bin" / "thùng", shallow ones "tray" / "khay".

### 5.3 Sources of commands

| Source | Use |
|---|---|
| Templates in `sorting_task.py` (4 object and 3 class templates per language) | Training data, baseline |
| Language-model templates in `configs/command_templates.json` (about 40 per language and form) | Training data |
| Held-out test commands in `configs/test_commands_v1.csv`: one free command per evaluation episode, drafted in varied styles and edited and approved row by row by the team; none reproduces a training template | Test set only |
| Held-out household commands in `configs/test_commands_household_v1.csv` (60 rows, seed 10), made the same way | Test set only |

Bottle sizes (version 1.5) are drawn from a separate random stream, so household scenes keep their items, positions and containers. Six of the 60 household scenes gained the other size of a bottle; their scene and allowed-object columns were re-derived. Five commands stayed correct as written; one (seed 1000005) now targets the small soft drink bottle, was reworded and waits for a new review.

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
