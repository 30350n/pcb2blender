import re
import shutil
import tempfile
from pathlib import Path
from typing import Union
from zipfile import ZIP_DEFLATED, ZipFile

import pcbnew

from .pcb3d import (
    PCB3D,
    Board,
    Bounds,
    DrillShape,
    KiCadColor,
    Pad,
    PadFabType,
    PadShape,
    PadType,
    StackedBoard,
    Stackup,
    SurfaceFinish,
)

SVG_MARGIN = 1.0  # mm

SURFACE_FINISH_MAP = {
    "ENIG": SurfaceFinish.ENIG,
    "ENEPIG": SurfaceFinish.ENIG,
    "Hard gold": SurfaceFinish.ENIG,
    "Immersion gold": SurfaceFinish.ENIG,
    "HT_OSP": SurfaceFinish.NONE,
    "OSP": SurfaceFinish.NONE,
    "None": SurfaceFinish.NONE,
}


def export_pcb3d(filepath: Path, boarddefs: dict[str, Board]):
    init_tempdir()

    wrl_path = get_temppath(PCB3D.PCB)
    components_path = get_temppath(PCB3D.COMPONENTS)
    pcbnew.ExportVRML(wrl_path, 0.001, True, False, True, True, components_path, 0.0, 0.0)

    layers_path = get_temppath(PCB3D.LAYERS)
    board: pcbnew.BOARD = pcbnew.GetBoard()
    box: pcbnew.BOX2I = board.ComputeBoundingBox(aBoardEdgesOnly=True)
    bounds = Bounds(
        (ToMM(box.GetLeft()) - SVG_MARGIN, ToMM(box.GetTop()) - SVG_MARGIN),
        (ToMM(box.GetWidth()) + SVG_MARGIN * 2, ToMM(box.GetHeight()) + SVG_MARGIN * 2),
    )
    export_layers(board, bounds, layers_path)

    pads = {}
    footprint: pcbnew.FOOTPRINT
    for i, footprint in enumerate(board.Footprints()):
        has_model = len(footprint.Models()) > 0
        is_tht_or_smd = bool(footprint.GetAttributes() & (pcbnew.FP_THROUGH_HOLE | pcbnew.FP_SMD))
        value = footprint.GetValue()
        reference = footprint.GetReference()

        pad: pcbnew.PAD
        for j, pad in enumerate(footprint.Pads()):
            name = sanitized(f"{value}_{reference}_{i}_{j}")
            is_flipped: bool = pad.IsFlipped()
            has_paste: bool = pad.IsOnLayer(pcbnew.B_Paste if is_flipped else pcbnew.F_Paste)
            pads[name] = Pad(
                ToMM2D(pad.GetPosition()),
                is_flipped,
                has_model,
                is_tht_or_smd,
                has_paste,
                PadType(pad.GetAttribute()),
                PadShape(pad.GetShape()),
                ToMM2D(pad.GetSize()),
                pad.GetOrientation().AsRadians(),
                pad.GetRoundRectRadiusRatio(),
                DrillShape(pad.GetDrillShape()),
                ToMM2D(pad.GetDrillSize()),
                PadFabType(pad.GetProperty()),
            )

    with ZipFile(filepath, mode="w", compression=ZIP_DEFLATED) as file:
        PCB3D(bounds, get_stackup(board), boarddefs, pads).write(
            file, wrl_path, components_path, layers_path
        )


def get_boarddefs(board: pcbnew.BOARD):
    boarddefs: dict[str, Board] = {}
    ignored: list[str] = []

    tls: dict[str, tuple[float, float]] = {}
    brs: dict[str, tuple[float, float]] = {}
    stacks: dict[str, tuple[float, float]] = {}

    drawing: pcbnew.BOARD_ITEM
    for drawing in board.GetDrawings():
        if drawing.Type() == pcbnew.PCB_TEXT_T:
            text_obj: pcbnew.PCB_TEXT = drawing.Cast()
            text: str = text_obj.GetText()

            if not text.startswith("PCB3D_"):
                continue

            pos = ToMM2D(text_obj.GetPosition())
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

            boarddefs[sanitized(name)] = Board(
                Bounds((tl_pos[0], tl_pos[1]), (br_pos[0] - tl_pos[0], br_pos[1] - tl_pos[1])),
            )

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
        target_pos = boarddefs[target_name].bounds.top_left
        stacked = StackedBoard(
            (stack_pos[0] - target_pos[0], stack_pos[1] - target_pos[1], z_offset)
        )
        boarddefs[target_name].stacked_boards[other_name] = stacked

    ignored += list(tls.keys()) + list(brs.keys()) + list(stacks.keys())

    return boarddefs, ignored


def get_stackup(board: pcbnew.BOARD) -> Stackup:
    stackup = Stackup()

    tmp_path = get_temppath("pcb2blender_tmp.kicad_pcb")
    pcbnew.SaveBoard(str(tmp_path), board, aSkipSettings=True)
    content = tmp_path.read_text(encoding="utf-8")

    if not (match := STACKUP_REGEX.search(content)):
        return stackup
    stackup_content = match.group(0)

    if matches := STACKUP_THICKNESS_REGEX.finditer(stackup_content):
        stackup.thickness_mm = sum(float(match.group(1)) for match in matches)

    if match := STACKUP_MASK_REGEX.search(stackup_content):
        stackup.mask_color, stackup.mask_color_custom = parse_kicad_color(match.group(1))

    if match := STACKUP_SILKS_REGEX.search(stackup_content):
        stackup.silks_color, stackup.silks_color_custom = parse_kicad_color(match.group(1))

    if match := STACKUP_COPPER_FINISH_REGEX.search(stackup_content):
        stackup.surface_finish = SURFACE_FINISH_MAP.get(match.group(1), SurfaceFinish.HASL)

    return stackup


def parse_kicad_color(string: str) -> tuple[KiCadColor, tuple[int, int, int]]:
    if string[0] == "#":
        return KiCadColor.CUSTOM, hex2rgb(string[1:7])
    else:
        return KiCadColor[string.upper()], (0, 0, 0)


def export_layers(board: pcbnew.BOARD, bounds: Bounds, output_directory: Path):
    plot_controller = pcbnew.PLOT_CONTROLLER(board)
    plot_options: pcbnew.PCB_PLOT_PARAMS = plot_controller.GetPlotOptions()
    plot_options.SetOutputDirectory(output_directory)

    plot_options.SetPlotFrameRef(False)
    plot_options.SetAutoScale(False)
    plot_options.SetScale(1)
    plot_options.SetMirror(False)
    plot_options.SetUseGerberAttributes(True)
    plot_options.SetDrillMarksType(pcbnew.DRILL_MARKS_NO_DRILL_SHAPE)

    for layer in PCB3D.INCLUDED_LAYERS:
        plot_controller.SetLayer(getattr(pcbnew, layer))
        plot_controller.OpenPlotfile(layer, pcbnew.PLOT_FORMAT_SVG, "")
        plot_controller.PlotLayer()
        filepath = Path(plot_controller.GetPlotFileName())
        plot_controller.ClosePlot()
        filepath = filepath.replace(filepath.parent / f"{layer}.svg")

        content = filepath.read_text(encoding="utf-8")
        width = f"{bounds.size[0]:.6f}mm"
        height = f"{bounds.size[1]:.6f}mm"
        viewBox = " ".join(f"{value:.6f}" for value in (*bounds.top_left, *bounds.size))
        content = SVG_HEADER_REGEX.sub(SVG_HEADER_SUB.format(width, height, viewBox), content)
        filepath.write_text(content, encoding="utf-8")


def sanitized(name: str):
    return re.sub(r"[\W]+", "_", name)


def get_tempdir():
    return Path(tempfile.gettempdir()) / "pcb2blender_tmp"


def get_temppath(filename: str):
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


def hex2rgb(hex_string: str):
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


def ToMM(value: Union[float, int]) -> float:
    return pcbnew.ToMM(value)  # pyright: ignore[reportReturnType]


def ToMM2D(value: Union[tuple[float, float], tuple[int, int]]) -> tuple[float, float]:
    return pcbnew.ToMM(value)  # pyright: ignore[reportReturnType]


SVG_HEADER_REGEX = re.compile(
    r"<svg([^>]*)width=\"[^\"]*\"[^>]*height=\"[^\"]*\"[^>]*viewBox=\"[^\"]*\"[^>]*>"
)
SVG_HEADER_SUB = '<svg\\g<1>width="{}" height="{}" viewBox="{}">'

STACKUP_REGEX = re.compile(r"\(stackup\s*(?:\s*\([^\(\)]*(?:\([^\)]*\)\s*)*\)\s*)*\)", re.MULTILINE)
STACKUP_THICKNESS_REGEX = re.compile(r"\(thickness\s+([^) ]*)[^)]*\)")
STACKUP_MASK_REGEX = re.compile(
    r"\(layer\s+\"[FB].Mask\"\s+(?:\([^()]*\)\s+)*?\(color\s+\"([^\)]*)\"\s*\)", re.DOTALL
)
STACKUP_SILKS_REGEX = re.compile(
    r"\(layer\s+\"[FB].SilkS\"\s+(?:\([^()]*\)\s+)*?\(color\s+\"([^\)]*)\"\s*\)", re.DOTALL
)
STACKUP_COPPER_FINISH_REGEX = re.compile(r"\(copper_finish\s+\"([^\"]*)\"\s*\)")
