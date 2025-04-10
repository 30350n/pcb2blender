import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from zipfile import Path as ZipPath, ZipFile

if "bpy" in locals():
    from error_helper import warning
else:
    warning = print


class PadType(Enum):
    UNKNOWN = -1
    THT = 0
    SMD = 1
    CONN = 2
    NPTH = 3

    @classmethod
    def _missing_(cls, value: Any):
        warning(f"unknown pad type '{value}'")
        return cls.UNKNOWN


class PadShape(Enum):
    UNKNOWN = -1
    CIRCLE = 0
    RECT = 1
    OVAL = 2
    TRAPEZOID = 3
    ROUNDRECT = 4
    CHAMFERED_RECT = 5
    CUSTOM = 6

    @classmethod
    def _missing_(cls, value: Any):
        warning(f"unknown pad shape '{value}'")
        return cls.UNKNOWN


class DrillShape(Enum):
    UNKNOWN = -1
    CIRCULAR = 0
    OVAL = 1

    @classmethod
    def _missing_(cls, value: Any):
        warning(f"unknown drill shape '{value}'")
        return cls.UNKNOWN


@dataclass
class Pad:
    position: tuple[float, float]
    is_flipped: bool
    has_model: bool
    is_tht_or_smd: bool
    has_paste: bool
    pad_type: PadType
    shape: PadShape
    size: tuple[float, float]
    rotation: float
    roundness: float
    drill_shape: DrillShape
    drill_size: tuple[float, float]

    FORMAT = "!ff????BBffffBff"
    FORMAT_SIZE = struct.calcsize(FORMAT)

    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) != cls.FORMAT_SIZE:
            data = data[: cls.FORMAT_SIZE].ljust(cls.FORMAT_SIZE, b"\x00")
            warning(f"unexpected pad data size '{len(data)}' (expected {cls.FORMAT_SIZE})")

        unpacked = struct.unpack(cls.FORMAT, data)
        return Pad(
            (unpacked[0], -unpacked[1]),
            unpacked[2],
            unpacked[3],
            unpacked[4],
            unpacked[5],
            PadType(unpacked[6]),
            PadShape(unpacked[7]),
            (unpacked[8], unpacked[9]),
            unpacked[10],
            unpacked[11],
            DrillShape(unpacked[12]),
            (unpacked[13], unpacked[14]),
        )

    def to_bytes(self):
        return struct.pack(
            self.FORMAT,
            self.position,
            self.is_flipped,
            self.has_model,
            self.is_tht_or_smd,
            self.has_paste,
            self.pad_type.value,
            self.shape.value,
            self.size,
            self.rotation,
            self.roundness,
            self.drill_shape,
            self.drill_size,
        )


class KiCadColor(Enum):
    CUSTOM = 0
    GREEN = 1
    RED = 2
    BLUE = 3
    PURPLE = 4
    BLACK = 5
    WHITE = 6
    YELLOW = 7


class SurfaceFinish(Enum):
    HASL = 0
    ENIG = 1
    NONE = 2


@dataclass
class Stackup:
    thickness_mm: float = 1.6
    mask_color: KiCadColor = KiCadColor.GREEN
    mask_color_custom: tuple[float, ...] = (0.0, 0.0, 0.0)
    silks_color: KiCadColor = KiCadColor.WHITE
    silks_color_custom: tuple[float, ...] = (0.0, 0.0, 0.0)
    surface_finish: SurfaceFinish = SurfaceFinish.HASL

    FORMAT = "!fbBBBbBBBb"
    FORMAT_SIZE = struct.calcsize(FORMAT)

    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) != cls.FORMAT_SIZE:
            data = data[: cls.FORMAT_SIZE].ljust(cls.FORMAT_SIZE, b"\x00")
            warning(f"unexpected stackup data size '{len(data)}' (expected {cls.FORMAT_SIZE})")

        unpacked = struct.unpack(cls.FORMAT, data)
        return Stackup(
            unpacked[0],
            KiCadColor(unpacked[1]),
            (unpacked[2] / 255, unpacked[3] / 255, unpacked[4] / 255),
            KiCadColor(unpacked[5]),
            (unpacked[6] / 255, unpacked[7] / 255, unpacked[8] / 255),
            SurfaceFinish(unpacked[9]),
        )

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.thickness_mm,
            self.mask_color,
            *self.mask_color_custom,
            self.silks_color,
            *self.silks_color_custom,
            self.surface_finish,
        )


@dataclass
class Bounds:
    top_left: tuple[float, float]
    size: tuple[float, float]

    FORMAT = "!ffff"

    @classmethod
    def from_bytes(cls, data: bytes):
        unpacked = struct.unpack(cls.FORMAT, data)
        return Bounds(unpacked[0:2], unpacked[2:4])

    def to_bytes(self):
        return struct.pack(self.FORMAT, *self.top_left, *self.size)

    @property
    def bottom_right(self):
        return (self.top_left[0] + self.size[0], self.top_left[1] + self.size[1])

    @property
    def center(self):
        return self.top_left[0] + self.size[0] * 0.5, self.top_left[1] + self.size[1] * 0.5


class StackedBoard(tuple[float, float, float]):
    FORMAT = "!fff"

    @classmethod
    def from_bytes(cls, data: bytes):
        return StackedBoard(struct.unpack(cls.FORMAT, data))

    def to_bytes(self):
        return struct.pack(self.FORMAT, *self)


@dataclass
class Board:
    bounds: Bounds
    stacked_boards: dict[str, StackedBoard] = field(default_factory=dict)


@dataclass
class PCB3D:
    layers_bounds: Bounds
    stackup: Stackup
    boards: dict[str, Board]
    pads: dict[str, Pad]
    content: str = ""

    @classmethod
    def from_file(
        cls,
        file: ZipFile,
        extract_dir: Path,
        on_error: Callable[[str], Any] = print,
        on_warning: Callable[[str], None] = print,
    ):
        members = {path.name for path in ZipPath(file).iterdir()}
        if missing := cls.REQUIRED_MEMBERS.difference(members):
            return on_error(f"not a valid .pcb3d file: missing {str(missing)[1:-1]}")
        zip_path = ZipPath(file)

        with file.open(cls.PCB) as pcb_file:
            pcb_file_content = pcb_file.read().decode("utf-8")
            with open(extract_dir / cls.PCB, "wb") as filtered_file:
                filtered = cls.REGEX_FILTER_COMPONENTS.sub("\\g<prefix>", pcb_file_content)
                filtered_file.write(filtered.encode("utf-8"))

        components = {
            name
            for name in file.namelist()
            if name.startswith(f"{cls.COMPONENTS}/") and name.endswith(".wrl")
        }

        file.extractall(extract_dir, components)

        layers = (f"{cls.LAYERS}/{layer}.svg" for layer in cls.INCLUDED_LAYERS)
        file.extractall(extract_dir, layers)

        layers_bounds_path = zip_path / cls.LAYERS / cls.LAYERS_BOUNDS
        layers_bounds = Bounds.from_bytes(layers_bounds_path.read_bytes())

        if (layers_stackup_path := zip_path / cls.LAYERS / cls.LAYERS_STACKUP).exists():
            stackup = Stackup.from_bytes(layers_stackup_path.read_bytes())
        else:
            stackup = Stackup()
            on_warning("old file format: cls file doesn't contain stackup")

        boards = {}
        if not (boards_path := (zip_path / cls.BOARDS)).exists():
            on_warning(f'old file format: cls file doesn\'t contain "{cls.BOARDS}" dir')
        else:
            for board_dir in boards_path.iterdir():
                bounds_path = board_dir / cls.BOUNDS
                if not bounds_path.exists():
                    continue

                try:
                    bounds = Bounds.from_bytes(bounds_path.read_bytes())
                except struct.error:
                    on_warning(f'ignoring board "{board_dir}" (corrupted)')
                    continue

                stacked_boards: dict[str, StackedBoard] = {}
                for path in board_dir.iterdir():
                    if not path.name.startswith(cls.STACKED):
                        continue

                    try:
                        stacked_board = StackedBoard.from_bytes(path.read_bytes())
                    except struct.error:
                        on_warning("ignoring stacked board (corrupted)")
                        continue

                    stacked_board_name = path.name.split(cls.STACKED, 1)[-1]
                    stacked_boards[stacked_board_name] = stacked_board

                boards[board_dir.name] = Board(bounds, stacked_boards)

        pads = {}
        if not (pads_path := (zip_path / cls.PADS)).exists():
            on_warning(f'old file format: cls file doesn\'t contain "{cls.PADS}" dir')
        else:
            for path in pads_path.iterdir():
                try:
                    pads[path.name] = Pad.from_bytes(path.read_bytes())
                except struct.error:
                    on_warning("old file format: failed to parse pads")
                    break

        return PCB3D(layers_bounds, stackup, boards, pads, pcb_file_content)

    def write(self, file: ZipFile, wrl_file: Path, components_dir: Path, layers_dir: Path):
        # always ensure the COMPONENTS, LAYERS and BOARDS directories are created
        file.writestr(f"{self.COMPONENTS}/", "")
        file.writestr(f"{self.LAYERS}/", "")
        file.writestr(f"{self.BOARDS}/", "")

        file.write(wrl_file, self.PCB)
        for path in components_dir.glob("**/*.wrl"):
            file.write(path, f"{self.COMPONENTS}/{path.name}")

        for path in layers_dir.glob("**/*.svg"):
            file.write(path, f"{self.LAYERS}/{path.name}")
        file.writestr(f"{self.LAYERS}/{self.LAYERS_BOUNDS}", self.layers_bounds.to_bytes())
        file.writestr(f"{self.LAYERS}/{self.LAYERS_STACKUP}", self.stackup.to_bytes())

        for board_name, board in self.boards.items():
            subdir = f"{self.BOARDS}/{board_name}"
            file.writestr(f"{subdir}/{self.BOUNDS}", board.bounds.to_bytes())

            for stacked_name, stacked in board.stacked_boards.items():
                file.writestr(f"{subdir}/{self.STACKED}{stacked_name}", stacked.to_bytes())

        for pad_name, pad in self.pads.items():
            file.writestr(f"{self.PADS}/{pad_name}", pad.to_bytes())

    PCB = "pcb.wrl"
    COMPONENTS = "components"
    LAYERS = "layers"
    LAYERS_BOUNDS = "bounds"
    LAYERS_STACKUP = "stackup"
    BOARDS = "boards"
    BOUNDS = "bounds"
    STACKED = "stacked_"
    PADS = "pads"

    REQUIRED_MEMBERS = {PCB, LAYERS}

    _INCLUDED_LAYERS = ["Cu", "Paste", "SilkS", "Mask"]
    INCLUDED_LAYERS_FRONT = [f"F_{layer}" for layer in _INCLUDED_LAYERS]
    INCLUDED_LAYERS_BACK = [f"B_{layer}" for layer in _INCLUDED_LAYERS]
    INCLUDED_LAYERS = list(sum(zip(INCLUDED_LAYERS_FRONT, INCLUDED_LAYERS_BACK), ()))

    REGEX_FILTER_COMPONENTS = re.compile(
        r"(?P<prefix>Transform\s*{\s*"
        r"(?:rotation (?P<r>[^\n]*)\n)?\s*"
        r"(?:translation (?P<t>[^\n]*)\n)?\s*"
        r"(?:scale (?P<s>[^\n]*)\n)?\s*"
        r"children\s*\[\s*)"
        r"(?P<instances>(?:Transform\s*{\s*"
        r"(?:rotation [^\n]*\n)?\s*(?:translation [^\n]*\n)?\s*(?:scale [^\n]*\n)?\s*"
        r"children\s*\[\s*Inline\s*{\s*url\s*\"[^\"]*\"\s*}\s*]\s*}\s*)+)"
    )

    REGEX_COMPONENT = re.compile(
        r"Transform\s*{\s*"
        r"(?:rotation (?P<r>[^\n]*)\n)?\s*"
        r"(?:translation (?P<t>[^\n]*)\n)?\s*"
        r"(?:scale (?P<s>[^\n]*)\n)?\s*"
        r"children\s*\[\s*Inline\s*{\s*url\s*\"(?P<url>[^\"]*)\"\s*}\s*]\s*}\s*"
    )
