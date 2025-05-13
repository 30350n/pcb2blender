import re
import struct
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from types import GenericAlias
from typing import Any, Callable
from zipfile import Path as ZipPath, ZipFile

if "bpy" in locals():
    from error_helper import warning
else:
    warning = print


@dataclass
class TOMLSerializable:
    @classmethod
    def from_toml(cls, data: str):
        toml = load_toml(data)
        values = {}
        for f in fields(cls):
            value = toml[f.name]
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                if value in f.type._member_names_:
                    values[f.name] = f.type[value]
                elif -1 in f.type._value2member_map_:
                    f.type._missing_(value)
                    values[f.name] = f.type(-1)
            elif isinstance(f.type, (type, GenericAlias)):
                values[f.name] = f.type(value)  # pyright: ignore[reportCallIssue]
            else:
                values[f.name] = value
        return cls(**values)

    def to_toml(self):
        data = ""
        for f in fields(self):
            value = getattr(self, f.name)
            data += f"{f.name} = {self._toml_value(value)}\n"
        return data

    @classmethod
    def _toml_value(cls, value: Any) -> str:
        if isinstance(value, (tuple, list)):
            return f"[ {', '.join(cls._toml_value(subvalue) for subvalue in value)} ]"
        elif isinstance(value, Enum):
            return f'"{value.name}"'
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return str(value)


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


class PadFabType(Enum):
    NONE = 0
    BGA = 1
    FIDUCIAL = (2, 3)
    TESTPOINT = 4
    HEATSINK = 5
    CASTELLATED = 6
    MECHANICAL = 7

    @classmethod
    def _missing_(cls, value: Any):
        warning(f"unknown pad fabrication attribute '{value}'")
        return cls.NONE


@dataclass
class Pad(TOMLSerializable):
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
    fab_type: PadFabType = PadFabType.NONE

    FORMAT = "!ff????BBffffBff"
    FORMAT_SIZE = struct.calcsize(FORMAT)

    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) != cls.FORMAT_SIZE:
            data = data[: cls.FORMAT_SIZE].ljust(cls.FORMAT_SIZE, b"\x00")
            warning(f"unexpected pad data size '{len(data)}' (expected {cls.FORMAT_SIZE})")

        unpacked = struct.unpack(cls.FORMAT, data)
        return Pad(
            (unpacked[0], unpacked[1]),
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
class Stackup(TOMLSerializable):
    thickness_mm: float = 1.6
    mask_color: KiCadColor = KiCadColor.GREEN
    mask_color_custom: tuple[float, ...] = (0.0, 0.0, 0.0)
    silks_color: KiCadColor = KiCadColor.WHITE
    silks_color_custom: tuple[float, ...] = (0.0, 0.0, 0.0)
    surface_finish: SurfaceFinish = SurfaceFinish.HASL

    OLD_FORMAT = "!fbBBBbBBBb"
    OLD_FORMAT_SIZE = struct.calcsize(OLD_FORMAT)

    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) != cls.OLD_FORMAT_SIZE:
            data = data[: cls.OLD_FORMAT_SIZE].ljust(cls.OLD_FORMAT_SIZE, b"\x00")
            warning(f"unexpected stackup data size '{len(data)}' (expected {cls.OLD_FORMAT_SIZE})")

        unpacked = struct.unpack(cls.OLD_FORMAT, data)
        return Stackup(
            unpacked[0],
            KiCadColor(unpacked[1]),
            (unpacked[2] / 255, unpacked[3] / 255, unpacked[4] / 255),
            KiCadColor(unpacked[5]),
            (unpacked[6] / 255, unpacked[7] / 255, unpacked[8] / 255),
            SurfaceFinish(unpacked[9]),
        )


@dataclass
class Bounds(TOMLSerializable):
    top_left: tuple[float, float]
    size: tuple[float, float]

    OLD_FORMAT = "!ffff"

    @classmethod
    def from_bytes(cls, data: bytes):
        unpacked = struct.unpack(cls.OLD_FORMAT, data)
        return Bounds(unpacked[0:2], unpacked[2:4])

    @property
    def bottom_right(self):
        return (self.top_left[0] + self.size[0], self.top_left[1] + self.size[1])

    @property
    def center(self):
        return self.top_left[0] + self.size[0] * 0.5, self.top_left[1] + self.size[1] * 0.5


class StackedBoard(tuple[float, float, float]):
    OLD_FORMAT = "!fff"

    @classmethod
    def from_toml(cls, data: str):
        return StackedBoard(load_toml(data)["offset"])

    def to_toml(self):
        return f"offset = [ {self[0]}, {self[1]}, {self[2]} ]"

    @classmethod
    def from_bytes(cls, data: bytes):
        return StackedBoard(struct.unpack(cls.OLD_FORMAT, data))


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

        if (layers_bounds_path := zip_path / cls.LAYERS / cls.LAYERS_BOUNDS).exists():
            layers_bounds = Bounds.from_toml(layers_bounds_path.read_text())
        else:
            old_layers_bounds_path = zip_path / cls.LAYERS / cls.LAYERS_BOUNDS[:-5]
            layers_bounds = Bounds.from_bytes(old_layers_bounds_path.read_bytes())

        if (layers_stackup_path := zip_path / cls.LAYERS / cls.LAYERS_STACKUP).exists():
            stackup = Stackup.from_toml(layers_stackup_path.read_text())
        elif (old_layers_stackup_path := zip_path / cls.LAYERS / cls.LAYERS_STACKUP[:-5]).exists():
            stackup = Stackup.from_bytes(old_layers_stackup_path.read_bytes())
        else:
            stackup = Stackup()
            on_warning("old file format: cls file doesn't contain stackup")

        boards = {}
        if not (boards_path := (zip_path / cls.BOARDS)).exists():
            on_warning(f'old file format: cls file doesn\'t contain "{cls.BOARDS}" dir')
        else:
            for board_dir in boards_path.iterdir():
                try:
                    if (bounds_path := board_dir / cls.BOUNDS).exists():
                        bounds = Bounds.from_toml(bounds_path.read_text())
                    elif (old_bounds_path := board_dir / cls.BOUNDS[:-5]).exists():
                        bounds = Bounds.from_bytes(old_bounds_path.read_bytes())
                    else:
                        continue
                except (struct.error, ValueError):
                    on_warning(f'ignoring board "{board_dir}" (corrupted)')
                    continue

                stacked_boards: dict[str, StackedBoard] = {}
                for path in board_dir.iterdir():
                    if not path.name.startswith(cls.STACKED):
                        continue

                    try:
                        if path.suffix == ".toml":
                            stacked_board = StackedBoard.from_toml(path.read_text())
                        elif path.suffix == "":
                            stacked_board = StackedBoard.from_bytes(path.read_bytes())
                        else:
                            continue
                    except (struct.error, ValueError):
                        on_warning("ignoring stacked board (corrupted)")
                        continue

                    stacked_board_name = path.stem.split(cls.STACKED, 1)[-1]
                    stacked_boards[stacked_board_name] = stacked_board

                boards[board_dir.name] = Board(bounds, stacked_boards)

        pads = {}
        if not (pads_path := (zip_path / cls.PADS)).exists():
            on_warning(f'old file format: cls file doesn\'t contain "{cls.PADS}" dir')
        else:
            for path in pads_path.iterdir():
                try:
                    if path.suffix == ".toml":
                        pads[path.stem] = Pad.from_toml(path.read_text())
                    elif path.suffix == "":
                        pads[path.stem] = Pad.from_bytes(path.read_bytes())
                    else:
                        continue
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
        file.writestr(f"{self.LAYERS}/{self.LAYERS_BOUNDS}", self.layers_bounds.to_toml())
        file.writestr(f"{self.LAYERS}/{self.LAYERS_STACKUP}", self.stackup.to_toml())

        for board_name, board in self.boards.items():
            subdir = f"{self.BOARDS}/{board_name}"
            file.writestr(f"{subdir}/{self.BOUNDS}", board.bounds.to_toml())

            for stacked_name, stacked in board.stacked_boards.items():
                file.writestr(f"{subdir}/{self.STACKED}{stacked_name}.toml", stacked.to_toml())

        for pad_name, pad in self.pads.items():
            file.writestr(f"{self.PADS}/{pad_name}.toml", pad.to_toml())

    PCB = "pcb.wrl"
    COMPONENTS = "components"
    LAYERS = "layers"
    LAYERS_BOUNDS = "bounds.toml"
    LAYERS_STACKUP = "stackup.toml"
    BOARDS = "boards"
    BOUNDS = "bounds.toml"
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


def load_toml(data: str):
    import tomllib

    return tomllib.loads(data)
