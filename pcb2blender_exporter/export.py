import pcbnew, tempfile, shutil, struct, re
from pathlib import Path
from zipfile import ZipFile, ZipInfo
from dataclasses import dataclass, field

PCB = "pcb.wrl"
COMPONENTS = "components"
BOARDS = "boards"
BOUNDS = "bounds"
STACKED = "stacked_"

@dataclass
class StackedBoard:
    name: str
    offset: list[float]

@dataclass
class BoardDef:
    name: str
    bounds: list[float]
    stacked_boards: list[StackedBoard] = field(default_factory=list)

def export_pcb3d(filepath, boarddefs):
    init_tempdir()
    
    wrl_path = get_temppath(PCB)
    components_path = get_temppath(COMPONENTS)
    pcbnew.ExportVRML(wrl_path, 0.001, True, True, components_path, 0.0, 0.0)

    with ZipFile(filepath, mode="w") as file:
        # always ensure the BOARDS and COMPONENTS directories are created
        file.writestr(BOARDS + "/", "")
        file.writestr(COMPONENTS + "/", "")

        for boarddef in boarddefs.values():
            subdir = Path(BOARDS) / boarddef.name

            file.writestr(str(subdir / BOUNDS), struct.pack("!ffff", *boarddef.bounds))

            for stacked in boarddef.stacked_boards:
                file.writestr(
                    str(subdir / (STACKED + stacked.name)),
                    struct.pack("!fff", *stacked.offset)
                )

        file.write(wrl_path, PCB)
        for path in components_path.glob("**/*.wrl"):
            file.write(path, str(Path(COMPONENTS) / path.name))

def get_boarddefs(board):
    boarddefs = {}
    ignored = []

    tls = {}
    brs = {}
    stacks = {}
    for drawing in board.GetDrawings():
        if drawing.Type() == pcbnew.PCB_TEXT_T:
            text_obj = drawing.Cast()
            pos = text_obj.GetPosition()
            pos = (pos.x / 1000000.0, pos.y / 1000000.0)

            text = text_obj.GetText()
            add_to = None
            if text.startswith("PCB3D_TL_"):
                add_to = tls
            elif text.startswith("PCB3D_BR_"):
                add_to = brs
            elif text.startswith("PCB3D_STACK_"):
                add_to = stacks
            if text.startswith("PCB3D_"):
                if add_to != None and not text in add_to:
                    add_to[text] = pos
                else:
                    ignored.append(text)

    for tl_str in tls.copy():
        name = tl_str[9:]
        br_str = "PCB3D_BR_" + name
        if br_str in brs:
            tl_pos = tls.pop(tl_str)
            br_pos = brs.pop(br_str)

            boarddef = BoardDef(
                sanitized(name),
                (tl_pos[0], tl_pos[1], br_pos[0] - tl_pos[0], br_pos[1] - tl_pos[1])
            )
            boarddefs[boarddef.name] = boarddef

    for stack_str in stacks.copy():
        try:
            other, onto, target, z_offset = stack_str[12:].split("_")
            z_offset = float(z_offset)
        except ValueError:
            continue

        if onto != "ONTO":
            continue
        
        other_name = sanitized(other)
        target_name = sanitized(target)

        if not other_name in set(boarddefs) | {"FPNL"} or not target_name in boarddefs:
            continue

        stack_pos = stacks.pop(stack_str)
        target_pos = boarddefs[target_name].bounds[:2]
        stacked = StackedBoard(
            other_name,
            (stack_pos[0] - target_pos[0], stack_pos[1] - target_pos[1], z_offset)
        )
        boarddefs[target_name].stacked_boards.append(stacked)

    ignored += list(tls.keys()) + list(brs.keys()) + list(stacks.keys())

    return boarddefs, ignored

def sanitized(name):
    return re.sub("[\W]+", "_", name)

def get_tempdir():
    return Path(tempfile.gettempdir()) / "pcb2blender_tmp"

def get_temppath(filename):
    return get_tempdir() / filename

def init_tempdir():
    tempdir = get_tempdir()
    if tempdir.exists():
        shutil.rmtree(tempdir)
    tempdir.mkdir()
