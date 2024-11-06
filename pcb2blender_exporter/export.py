import re, shutil, struct, tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import pcbnew
from pcbnew import DRILL_MARKS_NO_DRILL_SHAPE, PLOT_CONTROLLER, PLOT_FORMAT_SVG, ToMM

PCB = "pcb.wrl"
COMPONENTS = "components"
LAYERS = "layers"
LAYERS_BOUNDS = "bounds"
LAYERS_STACKUP = "stackup"
BOARDS = "boards"
BOUNDS = "bounds"
STACKED = "stacked_"
PADS = "pads"

INCLUDED_LAYERS = (
    "F_Cu", "B_Cu", "F_Paste", "B_Paste", "F_SilkS", "B_SilkS", "F_Mask", "B_Mask"
)

SVG_MARGIN = 1.0 # mm

@dataclass
class StackedBoard:
    name: str
    offset: Tuple[float, float, float]

@dataclass
class BoardDef:
    name: str
    bounds: Tuple[float, float, float, float]
    stacked_boards: List[StackedBoard] = field(default_factory=list)

class KiCadColor(IntEnum):
    CUSTOM = 0
    GREEN  = 1
    RED    = 2
    BLUE   = 3
    PURPLE = 4
    BLACK  = 5
    WHITE  = 6
    YELLOW = 7

class SurfaceFinish(IntEnum):
    HASL = 0
    ENIG = 1
    NONE = 2

SURFACE_FINISH_MAP = {
    "ENIG": SurfaceFinish.ENIG,
    "ENEPIG": SurfaceFinish.ENIG,
    "Hard gold": SurfaceFinish.ENIG,
    "HT_OSP": SurfaceFinish.NONE,
    "OSP": SurfaceFinish.NONE,
}

@dataclass
class Stackup:
    thickness_mm: float = 1.6
    mask_color: KiCadColor = KiCadColor.GREEN
    mask_color_custom: Tuple[int, int, int] = (0, 0, 0)
    silks_color: KiCadColor = KiCadColor.WHITE
    silks_color_custom: Tuple[int, int, int] = (0, 0, 0)
    surface_finish: SurfaceFinish = SurfaceFinish.HASL

    def pack(self) -> bytes:
        return struct.pack("!fbBBBbBBBb",
            self.thickness_mm,
            self.mask_color, *self.mask_color_custom,
            self.silks_color, *self.silks_color_custom,
            self.surface_finish,
        )

def export_pcb3d(filepath, boarddefs):
    init_tempdir()

    wrl_path = get_temppath(PCB)
    components_path = get_temppath(COMPONENTS)
    pcbnew.ExportVRML(wrl_path, 0.001, True, True, components_path, 0.0, 0.0)

    layers_path = get_temppath(LAYERS)
    board = pcbnew.GetBoard()
    box = board.ComputeBoundingBox(aBoardEdgesOnly=True)
    bounds = (
        ToMM(box.GetLeft()) - SVG_MARGIN, ToMM(box.GetTop()) - SVG_MARGIN,
        ToMM(box.GetRight() - box.GetLeft()) + SVG_MARGIN * 2,
        ToMM(box.GetBottom() - box.GetTop()) + SVG_MARGIN * 2,
    )
    export_layers(board, bounds, layers_path)

    with ZipFile(filepath, mode="w", compression=ZIP_DEFLATED) as file:
        # always ensure the COMPONENTS, LAYERS and BOARDS directories are created
        file.writestr(f"{COMPONENTS}/", "")
        file.writestr(f"{LAYERS}/", "")
        file.writestr(f"{BOARDS}/", "")

        file.write(wrl_path, PCB)
        for path in components_path.glob("**/*.wrl"):
            file.write(path, f"{COMPONENTS}/{path.name}")

        for path in layers_path.glob("**/*.svg"):
            file.write(path, f"{LAYERS}/{path.name}")
        file.writestr(f"{LAYERS}/{LAYERS_BOUNDS}", struct.pack("!ffff", *bounds))
        file.writestr(f"{LAYERS}/{LAYERS_STACKUP}", get_stackup(board).pack())

        for boarddef in boarddefs.values():
            subdir = f"{BOARDS}/{boarddef.name}"
            file.writestr(f"{subdir}/{BOUNDS}", struct.pack("!ffff", *boarddef.bounds))

            for stacked in boarddef.stacked_boards:
                file.writestr(
                    f"{subdir}/{STACKED}{stacked.name}",
                    struct.pack("!fff", *stacked.offset)
                )

        for i, footprint in enumerate(board.Footprints()):
            has_model = len(footprint.Models()) > 0
            is_tht_or_smd = bool(
                footprint.GetAttributes() & (pcbnew.FP_THROUGH_HOLE | pcbnew.FP_SMD))
            value = footprint.GetValue()
            reference = footprint.GetReference()
            for j, pad in enumerate(footprint.Pads()):
                name = sanitized(f"{value}_{reference}_{i}_{j}")
                is_flipped = pad.IsFlipped()
                has_paste = pad.IsOnLayer(pcbnew.B_Paste if is_flipped else pcbnew.F_Paste)
                data = struct.pack(
                    "!ff????BBffffBff",
                    *map(ToMM, pad.GetPosition()),
                    is_flipped,
                    has_model,
                    is_tht_or_smd,
                    has_paste,
                    pad.GetAttribute(),
                    pad.GetShape(),
                    *map(ToMM, pad.GetSize()),
                    pad.GetOrientation().AsRadians(),
                    pad.GetRoundRectRadiusRatio(),
                    pad.GetDrillShape(),
                    *map(ToMM, pad.GetDrillSize()),
                )
                file.writestr(f"{PADS}/{name}", data)

def get_boarddefs(board):
    boarddefs = {}
    ignored = []

    tls = {}
    brs = {}
    stacks = {}
    for drawing in board.GetDrawings():
        if drawing.Type() == pcbnew.PCB_TEXT_T:
            text_obj = drawing.Cast()
            text = text_obj.GetText()

            if not text.startswith("PCB3D_"):
                continue

            pos = tuple(map(ToMM, text_obj.GetPosition()))
            if text.startswith("PCB3D_TL_"):
                tls.setdefault(text, pos)
            elif text.startswith("PCB3D_BR_"):
                brs.setdefault(text, pos)
            elif text.startswith("PCB3D_STACK_"):
                stacks.setdefault(text, pos)
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

        if other_name not in set(boarddefs) | {"FPNL"} or target_name not in boarddefs:
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

def get_stackup(board):
    stackup = Stackup()

    tmp_path = get_temppath("pcb2blender_tmp.kicad_pcb")
    pcbnew.SaveBoard(str(tmp_path), board, aSkipSettings=True)
    content = tmp_path.read_text(encoding="utf-8")

    if not (match := stackup_regex.search(content)):
        return stackup
    stackup_content = match.group(0)

    if matches := stackup_thickness_regex.finditer(stackup_content):
        stackup.thickness_mm = sum(float(match.group(1)) for match in matches)

    if match := stackup_mask_regex.search(stackup_content):
        stackup.mask_color, stackup.mask_color_custom = parse_kicad_color(match.group(1))

    if match := stackup_silks_regex.search(stackup_content):
        stackup.silks_color, stackup.silks_color_custom = parse_kicad_color(match.group(1))

    if match := stackup_copper_finish_regex.search(stackup_content):
        stackup.surface_finish = SURFACE_FINISH_MAP.get(match.group(1), SurfaceFinish.HASL)

    return stackup

def parse_kicad_color(string):
    if string[0] == "#":
        return KiCadColor.CUSTOM, hex2rgb(string[1:7])
    else:
        return KiCadColor[string.upper()], (0, 0, 0)

def export_layers(board, bounds, output_directory: Path):
    plot_controller = PLOT_CONTROLLER(board)
    plot_options = plot_controller.GetPlotOptions()
    plot_options.SetOutputDirectory(output_directory)

    plot_options.SetPlotFrameRef(False)
    plot_options.SetAutoScale(False)
    plot_options.SetScale(1)
    plot_options.SetMirror(False)
    plot_options.SetUseGerberAttributes(True)
    plot_options.SetDrillMarksType(DRILL_MARKS_NO_DRILL_SHAPE)

    for layer in INCLUDED_LAYERS:
        plot_controller.SetLayer(getattr(pcbnew, layer))
        plot_controller.OpenPlotfile(layer, PLOT_FORMAT_SVG, "")
        plot_controller.PlotLayer()
        filepath = Path(plot_controller.GetPlotFileName())
        plot_controller.ClosePlot()
        filepath = filepath.replace(filepath.parent / f"{layer}.svg")

        content = filepath.read_text(encoding="utf-8")
        width  = f"{bounds[2]:.6f}mm"
        height = f"{bounds[3]:.6f}mm"
        viewBox = " ".join(f"{value:.6f}" for value in bounds)
        content = svg_header_regex.sub(svg_header_sub.format(width, height, viewBox), content)
        filepath.write_text(content, encoding="utf-8")

def sanitized(name):
    return re.sub(r"[\W]+", "_", name)

def get_tempdir():
    return Path(tempfile.gettempdir()) / "pcb2blender_tmp"

def get_temppath(filename):
    return get_tempdir() / filename

def init_tempdir():
    tempdir = get_tempdir()
    if tempdir.exists():
        try:
            shutil.rmtree(tempdir)
        except OSError:
            try:
                # try to delete all files first
                for file in tempdir.glob("**/*"):
                    if file.is_file():
                        file.unlink()
                shutil.rmtree(tempdir)
            except OSError:
                # if this still doesn't work, fuck it
                return
    tempdir.mkdir()

def hex2rgb(hex_string):
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )

svg_header_regex = re.compile(
    r"<svg([^>]*)width=\"[^\"]*\"[^>]*height=\"[^\"]*\"[^>]*viewBox=\"[^\"]*\"[^>]*>"
)
svg_header_sub = "<svg\\g<1>width=\"{}\" height=\"{}\" viewBox=\"{}\">"

stackup_regex = re.compile(
    r"\(stackup\s*(?:\s*\([^\(\)]*(?:\([^\)]*\)\s*)*\)\s*)*\)", re.MULTILINE
)
stackup_thickness_regex = re.compile(r"\(thickness\s+([^) ]*)[^)]*\)")
stackup_mask_regex  = re.compile(
    r"\(layer\s+\"[FB].Mask\".*?\(color\s+\"([^\)]*)\"\s*\)", re.DOTALL
)
stackup_silks_regex = re.compile(
    r"\(layer\s+\"[FB].SilkS\".*?\(color\s+\"([^\)]*)\"\s*\)", re.DOTALL
)
stackup_copper_finish_regex = re.compile(r"\(copper_finish\s+\"([^\"]*)\"\s*\)")
