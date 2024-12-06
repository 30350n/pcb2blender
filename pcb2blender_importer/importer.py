import io, random, re, shutil, struct, sys, tempfile
from dataclasses import dataclass
from enum import Enum
from math import inf, radians
from pathlib import Path
from zipfile import BadZipFile, Path as ZipPath, ZipFile

import numpy as np
from error_helper import error, warning
from PIL import Image, ImageOps
from skia import SVGDOM, Color4f, Stream, Surface

import addon_utils, bmesh, bpy
from bpy.props import *
from bpy_extras.io_utils import ImportHelper, axis_conversion, orientation_helper
from mathutils import Matrix, Vector

from .io_scene_x3d.source import ImportX3D, X3D_PT_import_transform, import_x3d
from .materials import *

ENABLE_PROFILER = False
if ENABLE_PROFILER:
    from cProfile import Profile

    def has_debugger_attached():
        return sys.gettrace() is not None

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

REQUIRED_MEMBERS = {PCB, LAYERS}

@dataclass
class Board:
    bounds: tuple[Vector, Vector]
    stacked_boards: list[tuple[str, Vector]]
    obj: bpy.types.Object = None

class PadType(Enum):
    UNKNOWN = -1
    THT = 0
    SMD = 1
    CONN = 2
    NPTH = 3

    @classmethod
    def _missing_(cls, value):
        print(f"warning: unknown pad type '{value}'")
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
    def _missing_(cls, value):
        print(f"warning: unknown pad shape '{value}'")
        return cls.UNKNOWN

class DrillShape(Enum):
    UNKNOWN = -1
    CIRCULAR = 0
    OVAL = 1

    @classmethod
    def _missing_(cls, value):
        print(f"warning: unknown drill shape '{value}'")
        return cls.UNKNOWN

@dataclass
class Pad:
    position: Vector
    is_flipped: bool
    has_model: bool
    is_tht_or_smd: bool
    has_paste: bool
    pad_type: PadType
    shape: PadShape
    size: Vector
    rotation: float
    roundness: float
    drill_shape: DrillShape
    drill_size: Vector

    @staticmethod
    def from_bytes(data):
        unpacked = struct.unpack("!ff????BBffffBff", data)
        return Pad(
            Vector((unpacked[0], -unpacked[1])),
            *unpacked[2:6],
            PadType(unpacked[6]),
            PadShape(unpacked[7]),
            Vector(unpacked[8:10]),
            *unpacked[10:12],
            DrillShape(unpacked[12]),
            Vector(unpacked[13:15]),
        )

class KiCadColor(Enum):
    CUSTOM = 0
    GREEN  = 1
    RED    = 2
    BLUE   = 3
    PURPLE = 4
    BLACK  = 5
    WHITE  = 6
    YELLOW = 7

class SurfaceFinish(Enum):
    HASL = 0
    ENIG = 1
    NONE = 2

@dataclass
class Stackup:
    thickness_mm: float = 1.6
    mask_color: KiCadColor = KiCadColor.GREEN
    mask_color_custom: tuple[float, float, float] = None
    silks_color: KiCadColor = KiCadColor.WHITE
    silks_color_custom: tuple[float, float, float] = None
    surface_finish: SurfaceFinish = SurfaceFinish.HASL

    def from_bytes(data):
        unpacked = struct.unpack("!fbBBBbBBBb", data)
        return Stackup(
            unpacked[0],
            KiCadColor(unpacked[1]),
            tuple(Vector(unpacked[2:5]) / 255),
            KiCadColor(unpacked[5]),
            tuple(Vector(unpacked[6:9]) / 255),
            SurfaceFinish(unpacked[9]),
        )

@dataclass
class PCB3D:
    content: str
    components: list[str]
    layers_bounds: tuple[float, float, float, float]
    stackup: Stackup
    boards: dict[str, Board]
    pads: dict[str, Pad]

class ErrorHelper:
    def error(self, msg, prefix="error: "):
        error(msg, prefix=prefix)
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    def warning(self, msg, prefix="warning: "):
        warning(msg, prefix=prefix)
        self.report({"WARNING"}, msg)

class PCB2BLENDER_OT_import_pcb3d(bpy.types.Operator, ImportHelper, ErrorHelper):
    """Import a PCB3D file"""
    bl_idname = "pcb2blender.import_pcb3d"
    bl_label = "Import .pcb3d"
    bl_options = {"PRESET", "UNDO"}

    import_components: BoolProperty(name="Import Components", default=True)
    add_solder_joints: EnumProperty(name="Add Solder Joints", default="SMART",
        items=(
            ("NONE", "None", "Do not add any solder joints"),
            ("SMART", "Smart", "Only add solder joints to footprints that have THT/SMD "
                "attributes set and that have 3D models and only to pads which have a "
                "solder paste layer (for SMD pads)"),
            ("ALL", "All", "Add solder joints to all pads")))

    center_boards:     BoolProperty(name="Center PCBs", default=True)
    cut_boards:        BoolProperty(name="Cut PCBs", default=True)
    stack_boards:      BoolProperty(name="Stack PCBs", default=True)

    merge_materials:   BoolProperty(name="Merge Materials", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)
    pcb_material:      EnumProperty(name="PCB Material", default="RASTERIZED",
        items=(("RASTERIZED", "Rasterized (Cycles)", ""), ("3D", "3D (deprecated)", "")))
    texture_dpi:       FloatProperty(name="Texture DPI",
        default=1016.0, min=0.0, soft_min=508.0, soft_max=2032.0)

    import_fpnl:       BoolProperty(name="Import Frontpanel (.fpnl)", default=True,
        description="Import the specified .fpnl file and align it (if its stacked to a pcb).")
    fpnl_path:         StringProperty(name="", subtype="FILE_PATH",
        description="")
    fpnl_thickness:    FloatProperty(name="Panel Thickness (mm)",
        default=2.0, min=0.0, soft_max=5.0)
    fpnl_bevel_depth:  FloatProperty(name="Bevel Depth (mm)",
        default=0.05, min=0.0, soft_max=0.25)
    fpnl_setup_camera: BoolProperty(name="Setup Orthographic Camera", default=True)

    filter_glob:       StringProperty(default="*.pcb3d", options={"HIDDEN"})

    def __init__(self):
        self.last_fpnl_path = ""
        self.component_cache = {}
        self.new_materials = set()
        super().__init__()

    def execute(self, context):
        filepath = Path(self.filepath)

        if ENABLE_PROFILER and has_debugger_attached():
            profiler = Profile()
            profiler.enable()

        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # import boards

        if (pcb := self.import_pcb3d(context, filepath)) == {"CANCELLED"}:
            return {"CANCELLED"}

        # import front panel

        if has_svg2blender() and self.import_fpnl and self.fpnl_path != "":
            if Path(self.fpnl_path).is_file():
                bpy.ops.svg2blender.import_fpnl(
                    filepath=self.fpnl_path,
                    thickness=self.fpnl_thickness,
                    bevel_depth=self.fpnl_bevel_depth,
                    setup_camera=self.fpnl_setup_camera
                )
                pcb.boards["FPNL"] = Board((Vector(), Vector()), [], context.object)
            else:
                self.warning(f"frontpanel file \"{filepath}\" does not exist")

        # stack boards

        if self.stack_boards:
            for board in pcb.boards.values():
                for (name, offset) in board.stacked_boards:
                    if name not in pcb.boards:
                        self.warning(f"ignoring stacked board \"{name}\" (unknown board)")
                        continue

                    if not pcb.boards[name].obj:
                        self.warning(
                            f"ignoring stacked board \"{name}\" (cut_boards is set to False)")
                        continue

                    stacked_obj = pcb.boards[name].obj
                    stacked_obj.parent = board.obj

                    pcb_offset = Vector((0, 0, np.sign(offset.z) * pcb.stackup.thickness_mm))
                    if name == "FPNL":
                        pcb_offset.z += (self.fpnl_thickness - pcb.stackup.thickness_mm) * 0.5
                    stacked_obj.location = (offset + pcb_offset) * MM_TO_M

        # select pcb objects and make one active

        bpy.ops.object.select_all(action="DESELECT")
        top_level_boards = [board for board in pcb.boards.values() if not board.obj.parent]
        context.view_layer.objects.active = top_level_boards[0].obj
        for board in top_level_boards:
            board.obj.select_set(True)

        # center boards

        if self.center_boards:
            center = Vector((0, 0))
            for board in top_level_boards:
                center += (board.bounds[0] + board.bounds[1]) * 0.5
            center /= len(top_level_boards)

            for board in top_level_boards:
                board.obj.location.xy = (board.bounds[0] - center) * MM_TO_M

        # materials

        if self.pcb_material == "RASTERIZED":
            context.scene.render.engine = "CYCLES"

        if self.merge_materials:
            merge_materials(self.component_cache.values())

        for material in self.new_materials.copy():
            if not material.users:
                self.new_materials.remove(material)
                bpy.data.materials.remove(material)

        if self.enhance_materials:
            enhance_materials(self.new_materials)

        if ENABLE_PROFILER and has_debugger_attached():
            profiler.disable()
            profiler.dump_stats(
                Path(__file__).parent.resolve() / Path(__file__).with_suffix(".prof").name
            )

        return {"FINISHED"}

    def import_pcb3d(self, context, filepath):
        if not filepath.is_file():
            return self.error(f"file \"{filepath}\" does not exist")

        dirname = filepath.name.replace(".", "_") + f"_{random.getrandbits(64)}"
        tempdir = Path(tempfile.gettempdir()) / "pcb2blender_tmp" / dirname
        tempdir.mkdir(parents=True, exist_ok=True)

        try:
            with ZipFile(filepath) as file:
                MEMBERS = {path.name for path in ZipPath(file).iterdir()}
                if missing := REQUIRED_MEMBERS.difference(MEMBERS):
                    return self.error(f"not a valid .pcb3d file: missing {str(missing)[1:-1]}")
                pcb = self.parse_pcb3d(file, tempdir)
        except BadZipFile:
            return self.error("not a valid .pcb3d file: not a zip file")
        except (KeyError, struct.error) as e:
            return self.error(f"pcb3d file is corrupted: {e}")

        # import objects

        materials_before = set(bpy.data.materials)

        objects_before = set(bpy.data.objects)
        bpy.ops.pcb2blender.import_x3d(
            filepath=str(tempdir / PCB), global_scale=1.0, join=False, enhance_materials=False)
        pcb_objects = set(bpy.data.objects).difference(objects_before)
        pcb_objects = sorted(pcb_objects, key=lambda obj: obj.name)

        for obj in pcb_objects:
            obj.data.transform(Matrix.Diagonal((*obj.scale, 1)))
            obj.scale = (1, 1, 1)
            self.setup_uvs(obj, pcb.layers_bounds)

        # rasterize/import layer svgs

        if self.enhance_materials and self.pcb_material == "RASTERIZED":
            layers_path = tempdir / LAYERS
            dpi = self.texture_dpi
            images = {}
            for f_layer, b_layer in zip(INCLUDED_LAYERS[0::2], INCLUDED_LAYERS[1::2]):
                front = self.svg2img(layers_path / f"{f_layer}.svg", dpi).getchannel(0)
                back  = self.svg2img(layers_path / f"{b_layer}.svg", dpi).getchannel(0)

                if (layer := f_layer[2:]) != "Mask":
                    front = ImageOps.invert(front)
                    back  = ImageOps.invert(back)
                    empty = Image.new("L", front.size)

                png_path = layers_path / f"{filepath.stem}_{layer}.png"
                merged = Image.merge("RGB", (front, back, empty))
                merged.save(png_path)

                image = bpy.data.images.load(str(png_path))
                image.colorspace_settings.name = "Non-Color"
                image.pack()
                image.filepath = ""

                images[layer] = image

        # import components

        if self.import_components:
            for component in pcb.components:
                bpy.ops.pcb2blender.import_x3d(
                    filepath=str(tempdir / component), enhance_materials=False)
                obj = context.object
                obj.data.name = component.rsplit("/", 1)[1].rsplit(".", 1)[0]
                self.component_cache[component] = obj.data
                bpy.data.objects.remove(obj)

        self.new_materials |= set(bpy.data.materials) - materials_before

        shutil.rmtree(tempdir)

        # improve board mesh

        board_obj = pcb_objects[0]
        self.improve_board_mesh(context, board_obj)

        # enhance materials

        pcb_meshes = {obj.data for obj in pcb_objects if obj.type == "MESH"}

        if self.enhance_materials and self.pcb_material == "RASTERIZED":
            for obj in pcb_objects[1:]:
                bpy.data.objects.remove(obj)
            pcb_object = pcb_objects[0]

            pcb_object.data.transform(Matrix.Diagonal((1, 1, 1.015, 1)))

            board_material = pcb_object.data.materials[0]
            self.new_materials.discard(board_material)
            board_material.name = f"PCB_{filepath.stem}"
            setup_pcb_material(board_material.node_tree, images, pcb.stackup)
            if self.import_components and self.add_solder_joints != "NONE":
                for node_name in ("paste", "seperate_paste", "solder"):
                    board_material.node_tree.nodes[node_name].mute = True
        else:
            if len(pcb_objects) == len(PCB2_LAYER_NAMES):
                layers = dict(zip(PCB2_LAYER_NAMES, pcb_objects))
                self.enhance_pcb_layers(context, layers)
                pcb_objects = list(layers.values())

            pcb_object = pcb_objects[0]
            bpy.ops.object.select_all(action="DESELECT")
            for obj in pcb_objects:
                obj.select_set(True)
            context.view_layer.objects.active = pcb_object
            bpy.ops.object.join()
            bpy.ops.object.transform_apply()

        for mesh in pcb_meshes:
            if not mesh.users:
                bpy.data.meshes.remove(mesh)

        # cut boards

        # TODO: maybe handle this differently by always providing at least one board def?
        if not (has_multiple_boards := bool(pcb.boards and self.cut_boards)):
            name = f"PCB_{filepath.stem}"
            pcb_object.name = pcb_object.data.name = name
            if self.enhance_materials and self.pcb_material == "RASTERIZED":
                pcb_object.data.materials[0].name = name

            bounds = (
                Vector(pcb_object.bound_box[3]).xy * M_TO_MM,
                Vector(pcb_object.bound_box[5]).xy * M_TO_MM,
            )
            matrix = Matrix.Translation(bounds[0].to_3d() * MM_TO_M)
            pcb_object.data.transform(matrix.inverted())
            pcb_object.matrix_world = matrix @ pcb_object.matrix_world

            pcb.boards.clear()
            pcb.boards[name] = Board(bounds, [], pcb_object)

        else:
            pcb_mesh = pcb_object.data
            bpy.data.objects.remove(pcb_object)
            for name, board in pcb.boards.items():
                board_obj = bpy.data.objects.new(f"PCB_{name}", pcb_mesh.copy())
                context.collection.objects.link(board_obj)

                cut_material_index = len(board_obj.material_slots) + 1
                boundingbox = self.get_boundingbox(context, board.bounds, cut_material_index)
                self.cut_object(context, board_obj, boundingbox, "INTERSECT")
                bpy.data.objects.remove(boundingbox)

                # cleanup and reapply board edge vcs on the newly cut edge faces
                bm = bmesh.new()
                bm.from_mesh(board_obj.data)

                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-8)
                bmesh.ops.triangulate(bm, faces=bm.faces)

                board_edge = bm.faces.layers.int[LAYER_BOARD_EDGE]
                for face in bm.faces:
                    if face.material_index == cut_material_index:
                        face[board_edge] = 1
                        face.material_index = 0

                board_edge_faces = {face for face in bm.faces if face[board_edge]}
                board_edge_verts = {vert for face in board_edge_faces for vert in face.verts}

                keep_verts = set()
                for face in board_edge_faces:
                    for edge in face.edges:
                        if not board_edge_faces.issuperset(edge.link_faces):
                            continue
                        if edge.calc_face_angle(0) < ANGLE_LIMIT:
                            continue
                        keep_verts = keep_verts.union(edge.verts)

                dissolve_verts = board_edge_verts - keep_verts

                merge_distance = pcb.stackup.thickness_mm * MM_TO_M * 0.5
                merge_distance_sq = merge_distance ** 2

                # TODO: doing this separately for top/bottom verts could improve performance

                targetmap = {}
                for dissolve_vert in dissolve_verts:
                    closest_keep_vert = None
                    closest_distance = inf
                    for keep_vert in keep_verts:
                        distance = (keep_vert.co - dissolve_vert.co).length_squared
                        if distance < closest_distance:
                            closest_keep_vert = keep_vert
                            closest_distance = distance
                    if closest_distance < merge_distance_sq:
                        targetmap[dissolve_vert] = closest_keep_vert

                dissolve_verts.difference_update(targetmap)

                # merge the verts that didn't get merged to any keep_verts
                planar_edge_doubles = bmesh.ops.find_doubles(
                    bm, verts=list(dissolve_verts), dist=merge_distance
                )["targetmap"]

                bmesh.ops.weld_verts(bm, targetmap=targetmap | planar_edge_doubles)

                remaining_top_edge_verts = set()
                remaining_bot_edge_verts = set()
                for vert in planar_edge_doubles.values():
                    if vert not in planar_edge_doubles:
                        if vert.co.z > 0:
                            remaining_top_edge_verts.add(vert)
                        else:
                            remaining_bot_edge_verts.add(vert)

                for top_vert in remaining_top_edge_verts:
                    for bot_vert in (edge.other_vert(top_vert) for edge in top_vert.link_edges):
                        if bot_vert not in remaining_bot_edge_verts:
                            continue
                        if (top_vert.co.xy - bot_vert.co.xy).length_squared > merge_distance_sq:
                            continue
                        mean_xy = (top_vert.co.xy + bot_vert.co.xy) / 2
                        top_vert.co.xy = bot_vert.co.xy = mean_xy
                        remaining_bot_edge_verts.remove(bot_vert)
                        break

                bm.to_mesh(board_obj.data)
                bm.free()

                # uvs get messed up by weld_verts
                self.setup_uvs(board_obj, pcb.layers_bounds)

                bpy.ops.object.select_all(action="DESELECT")
                board_obj.select_set(True)
                context.view_layer.objects.active = board_obj

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.beautify_fill(angle_limit=ANGLE_LIMIT)
                bpy.ops.mesh.tris_convert_to_quads()
                bpy.ops.object.mode_set(mode="OBJECT")

                offset = board.bounds[0].to_3d() * MM_TO_M
                board_obj.data.transform(Matrix.Translation(-offset))
                board_obj.location = offset

                board.obj = board_obj

        # fix smooth shading issues
        bpy.ops.object.shade_smooth_by_angle(angle=radians(89), keep_sharp_edges=False)

        related_objects = []

        # populate components

        if self.import_components and self.component_cache:
            match = regex_filter_components.search(pcb.content)
            matrix_all = match2matrix(match)

            for match_instance in regex_component.finditer(match.group("instances")):
                matrix_instance = match2matrix(match_instance)
                url = match_instance.group("url")

                component = self.component_cache[url]
                instance = bpy.data.objects.new(component.name, component)
                add_smooth_by_angle_modifier(instance)
                instance.matrix_world = matrix_all @ matrix_instance @ MATRIX_FIX_SCALE_INV
                context.collection.objects.link(instance)
                related_objects.append(instance)

        # add solder joints

        solder_joint_cache = {}
        if self.import_components and self.add_solder_joints != "NONE" and pcb.pads:
            for pad_name, pad in pcb.pads.items():
                if self.add_solder_joints == "SMART":
                    if not pad.has_model or not pad.is_tht_or_smd:
                        continue
                    if pad.pad_type == PadType.SMD and not pad.has_paste:
                        continue

                if pad.pad_type not in {PadType.THT, PadType.SMD}:
                    continue
                if pad.shape == PadShape.UNKNOWN or pad.drill_shape == DrillShape.UNKNOWN:
                    continue

                pad_type = pad.pad_type.name
                pad_size = pad.size
                hole_shape = pad.drill_shape.name
                hole_size = pad.drill_size
                roundness = 0.0
                match pad.shape:
                    case PadShape.CIRCLE:
                        pad_size = (pad.size[0], pad.size[0])
                        roundness = 1.0
                    case PadShape.OVAL:
                        roundness = 1.0
                    case PadShape.ROUNDRECT:
                        roundness = pad.roundness * 2.0
                    case PadShape.TRAPEZOID | PadShape.CHAMFERED_RECT | PadShape.CUSTOM:
                        print(f"skipping solder joint for '{pad_name}', "
                              f"unsupported shape '{pad.shape.name}'")
                        continue

                cache_id = (pad_type, tuple(pad_size), hole_shape, tuple(hole_size), roundness)
                if not (solder_joint := solder_joint_cache.get(cache_id)):
                    bpy.ops.pcb2blender.solder_joint_add(
                        pad_type=pad_type,
                        pad_shape="RECTANGULAR",
                        pad_size=pad_size,
                        roundness=roundness,
                        hole_shape=hole_shape,
                        hole_size=hole_size,
                        pcb_thickness=pcb.stackup.thickness_mm,
                        reuse_material=True,
                    )
                    solder_joint = context.object
                    solder_joint_cache[cache_id] = solder_joint

                obj = solder_joint.copy()
                obj.name = f"SOLDER_{pad_name}"
                obj.location.xy = pad.position * MM_TO_M
                obj.rotation_euler.z = pad.rotation
                obj.scale.z *= 1.0 if pad.is_flipped ^ (pad.pad_type == PadType.SMD) else -1.0
                context.collection.objects.link(obj)
                related_objects.append(obj)

        for obj in solder_joint_cache.values():
            bpy.data.objects.remove(obj)

        if not has_multiple_boards:
            pcb_board = next(iter(pcb.boards.values()))
            for obj in related_objects:
                obj.location.xy -= pcb_board.bounds[0] * MM_TO_M
                obj.parent = pcb_board.obj
        else:
            for obj in related_objects:
                for board in pcb.boards.values():
                    x, y = obj.location.xy * M_TO_MM
                    p_min, p_max = board.bounds
                    if x >= p_min.x and x < p_max.x and y <= p_min.y and y > p_max.y:
                        parent_board = board
                        break
                else:
                    closest = None
                    min_distance = inf
                    for name, board in pcb.boards.items():
                        center = (board.bounds[0] + board.bounds[1]) * 0.5
                        distance = (obj.location.xy * M_TO_MM - center).length_squared
                        if distance < min_distance:
                            min_distance = distance
                            closest = (name, board)

                    name, parent_board = closest
                    self.warning(
                        f"assigning \"{obj.name}\" (out of bounds) "
                        f"to closest board \"{name}\"")

                obj.location.xy -= parent_board.bounds[0] * MM_TO_M
                obj.parent = parent_board.obj

        return pcb

    def parse_pcb3d(self, file, extract_dir) -> PCB3D:
        zip_path = ZipPath(file)

        with file.open(PCB) as pcb_file:
            pcb_file_content = pcb_file.read().decode("utf-8")
            with open(extract_dir / PCB, "wb") as filtered_file:
                filtered = regex_filter_components.sub("\\g<prefix>", pcb_file_content)
                filtered_file.write(filtered.encode("utf-8"))

        components = list({
            name for name in file.namelist()
            if name.startswith(f"{COMPONENTS}/") and name.endswith(".wrl")
        })
        file.extractall(extract_dir, components)

        layers = (f"{LAYERS}/{layer}.svg" for layer in INCLUDED_LAYERS)
        file.extractall(extract_dir, layers)

        layers_bounds_path = zip_path / LAYERS / LAYERS_BOUNDS
        layers_bounds = struct.unpack("!ffff", layers_bounds_path.read_bytes())

        if (layers_stackup_path := zip_path / LAYERS / LAYERS_STACKUP).exists():
            stackup = Stackup.from_bytes(layers_stackup_path.read_bytes())
        else:
            stackup = Stackup()
            self.warning("old file format: PCB3D file doesn't contain stackup")

        boards = {}
        if not (boards_path := (zip_path / BOARDS)).exists():
            self.warning(f"old file format: PCB3D file doesn't contain \"{BOARDS}\" dir")
        else:
            for board_dir in boards_path.iterdir():
                bounds_path = board_dir / BOUNDS
                if not bounds_path.exists():
                    continue

                try:
                    bounds = struct.unpack("!ffff", bounds_path.read_bytes())
                except struct.error:
                    self.warning(f"ignoring board \"{board_dir}\" (corrupted)")
                    continue

                bounds = (
                    Vector((bounds[0], -bounds[1])),
                    Vector((bounds[0] + bounds[2], -(bounds[1] + bounds[3])))
                )

                stacked_boards = []
                for path in board_dir.iterdir():
                    if not path.name.startswith(STACKED):
                        continue

                    try:
                        offset = struct.unpack("!fff", path.read_bytes())
                    except struct.error:
                        self.warning("ignoring stacked board (corrupted)")
                        continue

                    stacked_boards.append((
                        path.name.split(STACKED, 1)[-1],
                        Vector((offset[0], -offset[1], offset[2])),
                    ))

                boards[board_dir.name] = Board(bounds, stacked_boards)

        pads = {}
        if not (pads_path := (zip_path / PADS)).exists():
            self.warning(f"old file format: PCB3D file doesn't contain \"{PADS}\" dir")
        else:
            for path in pads_path.iterdir():
                try:
                    pads[path.name] = Pad.from_bytes(path.read_bytes())
                except struct.error:
                    self.warning(f"old file format: failed to parse pads")
                    break

        return PCB3D(pcb_file_content, components, layers_bounds, stackup, boards, pads)

    @staticmethod
    def get_boundingbox(context, bounds, material_index):
        NAME = "pcb2blender_bounds_tmp"
        mesh = bpy.data.meshes.new(NAME)
        obj = bpy.data.objects.new(NAME, mesh)
        context.collection.objects.link(obj)

        MARGIN_MM = -0.01
        size = Vector((1.0, -1.0, 1.0)) * (bounds[1] - bounds[0]).to_3d()
        scale = (size + 2.0 * Vector((MARGIN_MM, MARGIN_MM, 5.0))) * MM_TO_M
        translation = (bounds[0] - Vector((MARGIN_MM, -MARGIN_MM))).to_3d() * MM_TO_M
        matrix_scale = Matrix.Diagonal(scale).to_4x4()
        matrix_offset = Matrix.Translation(translation)
        bounds_matrix = matrix_offset @ matrix_scale @ Matrix.Translation((0.5, -0.5, 0))

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, matrix=bounds_matrix)
        for face in bm.faces:
            face.material_index = material_index
        bm.to_mesh(obj.data)
        bm.free()

        return obj

    @staticmethod
    def setup_uvs(obj, layers_bounds):
        mesh = obj.data

        vertices = np.empty(len(mesh.vertices) * 3)
        mesh.attributes["position"].data.foreach_get("vector", vertices)
        vertices = vertices.reshape((len(mesh.vertices), 3))

        indices = np.empty(len(mesh.loops), dtype=int)
        mesh.loops.foreach_get("vertex_index", indices)

        offset = np.array((layers_bounds[0], -layers_bounds[1]))
        size = np.array((layers_bounds[2], layers_bounds[3]))
        uvs = (vertices[:, :2][indices] * M_TO_MM - offset) / size + np.array((0, 1))

        uv_layer = mesh.uv_layers[0]
        uv_layer.uv.foreach_set("vector", uvs.flatten())

    @staticmethod
    def improve_board_mesh(context, obj):
        # fill holes in board mesh to make subsurface shading work
        # create vertex color layer for board edge and through holes

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        # TODO: these layers should be created with numpy, not bmesh

        board_edge = bm.faces.layers.int.new(LAYER_BOARD_EDGE)
        through_holes = bm.faces.layers.int.new(LAYER_THROUGH_HOLES)

        board_edge_verts = set()
        for face in bm.faces:
            if face.calc_area() > 1e-10:
                if abs(face.normal.z) > 1e-3:
                    continue
            else:
                lastsign = np.sign(face.verts[0].co.z)
                for vert in face.verts[1:]:
                    if lastsign != (lastsign := np.sign(vert.co.z)):
                        break
                else:
                    continue

            face[board_edge] = 1
            board_edge_verts = board_edge_verts.union(face.verts)

        midpoint = len(bm.verts) // 2
        bm.verts.ensure_lookup_table()
        for i, top_vert in enumerate(bm.verts[:midpoint]):
            if top_vert in board_edge_verts:
                continue
            bot_vert = bm.verts[midpoint + i]
            bm.edges.new((top_vert, bot_vert))

        filled = bmesh.ops.holes_fill(bm, edges=bm.edges)

        for face in filled["faces"]:
            face[through_holes] = 1

        bm.to_mesh(obj.data)
        bm.free()

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.dissolve_limited(angle_limit=ANGLE_LIMIT)
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.mesh.beautify_fill(angle_limit=ANGLE_LIMIT)
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode="OBJECT")

    @classmethod
    def enhance_pcb_layers(cls, context, layers):
        for side, direction in reversed(list(zip(("F", "B"), (1, -1)))):
            mask = layers[f"{side}_Mask"]
            copper = layers[f"{side}_Cu"]
            silk = layers[f"{side}_Silk"]

            # split copper layer into tracks and pads

            tracks = copper
            pads = cls.copy_object(copper, context.collection)
            layers[f"{side}_Pads"] = pads

            mask_cutter = cls.copy_object(mask, context.collection)
            cls.extrude_mesh_z(mask_cutter.data, 1e-3, True)
            cls.cut_object(context, tracks, mask_cutter, "INTERSECT")
            cls.cut_object(context, pads, mask_cutter, "DIFFERENCE")
            bpy.data.objects.remove(mask_cutter)

            # remove silkscreen on pads

            pads_cutter = cls.copy_object(pads, context.collection)
            cls.extrude_mesh_z(pads_cutter.data, 1e-3, True)
            cls.cut_object(context, silk, pads_cutter, "DIFFERENCE")
            bpy.data.objects.remove(pads_cutter)

            silk.visible_shadow = False

            # align the layers

            cls.translate_mesh_z(mask.data, -2e-5 * direction)
            cls.translate_mesh_z(silk.data, -7e-5 * direction)
            cls.translate_mesh_z(tracks.data, -1e-5 * direction)
            cls.translate_mesh_z(pads.data, -1e-5 * direction)

        # scale down vias to match the other layers
        vias = layers["Vias"]
        vias.data.transform(Matrix.Diagonal((1, 1, 0.97, 1)))
        vias.data.polygons.foreach_set("use_smooth", [True] * len(vias.data.polygons))

    @staticmethod
    def copy_object(obj, collection):
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        collection.objects.link(new_obj)
        return new_obj

    @staticmethod
    def cut_object(context, obj, cutter, mode):
        mod_name = "Cut Object"
        modifier = obj.modifiers.new(mod_name, type="BOOLEAN")
        modifier.operation = mode
        modifier.object = cutter
        modifier.use_self = True

        if (modifier_index := len(obj.modifiers) - 1) > 0:
            obj.modifiers.move(modifier_index, 0)
        context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod_name)

    @staticmethod
    def extrude_mesh_z(mesh, z, symmetric=False):
        vec = Vector((0, 0, z))
        bm = bmesh.new()
        bm.from_mesh(mesh)

        result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [v for v in result["geom"] if isinstance(v, bmesh.types.BMVert)]
        if symmetric:
            bmesh.ops.translate(bm, vec=vec * 2, verts=extruded_verts)
            bmesh.ops.translate(bm, vec=-vec, verts=bm.verts[:])
        else:
            bmesh.ops.translate(bm, vec=vec, verts=extruded_verts)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

        bm.to_mesh(mesh)
        bm.free()

    @staticmethod
    def translate_mesh_z(mesh, z):
        mesh.transform(Matrix.Translation((0, 0, z)))

    @staticmethod
    def apply_transformation(obj, matrix):
        obj.data.transform(matrix)
        for child in obj.children:
            child.matrix_basis = matrix @ child.matrix_basis

    @staticmethod
    def svg2img(svg_path, dpi):
        svg = SVGDOM.MakeFromStream(Stream.MakeFromFile(str(svg_path)))

        SKIA_MAGIC = 0.282222222
        dpmm = dpi * INCH_TO_MM * SKIA_MAGIC

        width, height = svg.containerSize()
        pixels_width, pixels_height = round(width * dpmm), round(height * dpmm)
        surface = Surface(pixels_width, pixels_height)

        with surface as canvas:
            canvas.clear(Color4f.kWhite)
            canvas.scale(pixels_width / width, pixels_height / height)
            svg.render(canvas)

        with io.BytesIO(surface.makeImageSnapshot().encodeToData()) as file:
            image = Image.open(file)
            image.load()

        return image

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "import_components")
        col = layout.column()
        col.enabled = self.import_components
        col.label(text="Add Solder Joints")
        col.prop(self, "add_solder_joints", text="")
        layout.split()
        layout.prop(self, "center_boards")
        layout.prop(self, "cut_boards")
        layout.prop(self, "stack_boards")
        layout.split()

        layout.prop(self, "merge_materials")
        layout.prop(self, "enhance_materials")
        col = layout.column()
        col.enabled = self.enhance_materials
        col.label(text="PCB Material")
        col.prop(self, "pcb_material", text="")
        if self.pcb_material == "RASTERIZED":
            col.prop(self, "texture_dpi", slider=True)

        if has_svg2blender():
            layout.split()
            layout.prop(self, "import_fpnl")
            box = layout.box()
            box.enabled = self.import_fpnl
            box.prop(self, "fpnl_path")

            box.prop(self, "fpnl_thickness", slider=True)
            box.prop(self, "fpnl_bevel_depth", slider=True)
            box.prop(self, "fpnl_setup_camera")

            filebrowser_params = context.space_data.params
            filename  = Path(filebrowser_params.filename)
            directory = Path(filebrowser_params.directory.decode("utf-8"))

            if filename.suffix == ".pcb3d":
                if self.fpnl_path == "" or self.fpnl_path == self.last_fpnl_path:
                    auto_path = directory / (filename.stem + ".fpnl")
                    if auto_path.is_file():
                        self.fpnl_path = str(auto_path)
                        self.last_fpnl_path = self.fpnl_path
                    else:
                        self.fpnl_path = ""
            else:
                self.fpnl_path = ""

PCB2_LAYER_NAMES = (
    "Board",
    "F_Cu",
    "F_Paste",
    "F_Mask",
    "B_Cu",
    "B_Paste",
    "B_Mask",
    "Vias",
    "F_Silk",
    "B_Silk",
)

MM_TO_M = 1e-3
M_TO_MM = 1e3
INCH_TO_MM = 1 / 25.4

FIX_X3D_SCALE = 2.54 * MM_TO_M
MATRIX_FIX_SCALE_INV = Matrix.Scale(FIX_X3D_SCALE, 4).inverted()

ANGLE_LIMIT = radians(0.1)

regex_filter_components = re.compile(
    r"(?P<prefix>Transform\s*{\s*"
    r"(?:rotation (?P<r>[^\n]*)\n)?\s*"
    r"(?:translation (?P<t>[^\n]*)\n)?\s*"
    r"(?:scale (?P<s>[^\n]*)\n)?\s*"
    r"children\s*\[\s*)"
    r"(?P<instances>(?:Transform\s*{\s*"
    r"(?:rotation [^\n]*\n)?\s*(?:translation [^\n]*\n)?\s*(?:scale [^\n]*\n)?\s*"
    r"children\s*\[\s*Inline\s*{\s*url\s*\"[^\"]*\"\s*}\s*]\s*}\s*)+)"
)

regex_component = re.compile(
    r"Transform\s*{\s*"
    r"(?:rotation (?P<r>[^\n]*)\n)?\s*"
    r"(?:translation (?P<t>[^\n]*)\n)?\s*"
    r"(?:scale (?P<s>[^\n]*)\n)?\s*"
    r"children\s*\[\s*Inline\s*{\s*url\s*\"(?P<url>[^\"]*)\"\s*}\s*]\s*}\s*"
)

def match2matrix(match):
    rotation    = match.group("r")
    translation = match.group("t")
    scale       = match.group("s")

    matrix = Matrix()
    if translation:
        translation = map(float, translation.split())
        matrix = matrix @ Matrix.Translation(translation)
    if rotation:
        rotation = tuple(map(float, rotation.split()))
        matrix = matrix @ Matrix.Rotation(rotation[3], 4, rotation[:3])
    if scale:
        scale = map(float, scale.split())
        matrix = matrix @ Matrix.Diagonal(scale).to_4x4()

    return matrix

@orientation_helper(axis_forward="Y", axis_up="Z")
class PCB2BLENDER_OT_import_x3d(bpy.types.Operator, ImportHelper):
    __doc__ = ImportX3D.__doc__
    bl_idname = "pcb2blender.import_x3d"
    bl_label = ImportX3D.bl_label
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".x3d"
    filter_glob: StringProperty(default="*.x3d;*.wrl;*.wrz", options={"HIDDEN"})
    file_unit: EnumProperty(
        name="File Unit",
        items=(
            ("M", "Meter", ""),
            ("DM", "Decimeter", ""),
            ("CM", "Centimeter", ""),
            ("MM", "Millimeter", ""),
            ("IN", "Inch", ""),
            ("CUSTOM", "CUSTOM", ""),
        ),
        description="Unit used in the input file",
        default="CUSTOM",
        update=ImportX3D._file_unit_update,
    )
    global_scale: FloatProperty(
        name="Scale",
        soft_min=0.001, soft_max=1000.0,
        default=FIX_X3D_SCALE,
        precision=4,
        step=1.0,
        description="Scale value used when 'File Unit' is set to 'CUSTOM'",
    )

    join:              BoolProperty(name="Join Shapes", default=True)
    tris_to_quads:     BoolProperty(name="Tris to Quads", default=True)
    auto_smooth:       BoolProperty(name="Auto Smooth", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)

    def execute(self, context):
        bpy.ops.object.select_all(action="DESELECT")

        objects_before = set(bpy.data.objects)
        matrix = axis_conversion(from_forward=self.axis_forward, from_up=self.axis_up).to_4x4()
        result = import_x3d.load(
            context, self.filepath, global_scale=self.global_scale, global_matrix=matrix,
        )
        if not result == {"FINISHED"}:
            return result

        if not (objects := list(set(bpy.data.objects).difference(objects_before))):
            return {"FINISHED"}

        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = objects[0]

        if self.auto_smooth:
            bpy.ops.object.shade_smooth()

        if self.join:
            for obj in objects.copy():
                if obj.type != "MESH":
                    bpy.data.objects.remove(obj)
                    objects.remove(obj)
            context.view_layer.objects.active = objects[0]

            if len(objects) > 1:
                bpy.ops.object.join()
            bpy.ops.object.transform_apply()

            joined_obj = context.object
            joined_obj.name = Path(self.filepath).name.rsplit(".", 1)[0]
            joined_obj.data.name = joined_obj.name
            objects = [joined_obj]

            if self.auto_smooth:
                add_smooth_by_angle_modifier(joined_obj)

        else:
            bpy.ops.object.transform_apply(location=False, rotation=False)
            if self.auto_smooth:
                for obj in objects:
                    if obj.type == "MESH":
                        add_smooth_by_angle_modifier(obj)

        if self.tris_to_quads:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.tris_convert_to_quads()
            bpy.ops.object.mode_set(mode="OBJECT")

        if self.enhance_materials:
            materials = sum((obj.data.materials[:] for obj in objects), [])
            merge_materials([obj.data for obj in objects])
            for material in materials[:]:
                if not material.users:
                    materials.remove(material)
                    bpy.data.materials.remove(material)
            enhance_materials(materials)

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "join")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_smooth")
        layout.prop(self, "enhance_materials")

class PCB2BLENDER_PT_import_transform_x3d(X3D_PT_import_transform):
    @classmethod
    def poll(cls, context):
        return context.space_data.active_operator.bl_idname == "PCB2BLENDER_OT_import_x3d"

# https://projects.blender.org/blender/blender/issues/117399
def get_internal_asset_path():
    for path_type in ("LOCAL", "SYSTEM", "USER"):
        path = Path(bpy.utils.resource_path(path_type)) / "datafiles" / "assets"
        if path.exists():
            return path
    assert False

SMOOTH_BY_ANGLE_ASSET_PATH = str(
    get_internal_asset_path() / "geometry_nodes" / "smooth_by_angle.blend"
)
SMOOTH_BY_ANGLE_NODE_GROUP_NAME = "Smooth by Angle"
def add_smooth_by_angle_modifier(obj):
    global SMOOTH_BY_ANGLE_NODE_GROUP_NAME

    smooth_by_angle_node_group = bpy.data.node_groups.get(SMOOTH_BY_ANGLE_NODE_GROUP_NAME)
    if not smooth_by_angle_node_group or smooth_by_angle_node_group.type != "GEOMETRY":
        with bpy.data.libraries.load(SMOOTH_BY_ANGLE_ASSET_PATH) as (data_from, data_to):
            data_to.node_groups = [SMOOTH_BY_ANGLE_NODE_GROUP_NAME]
        smooth_by_angle_node_group = data_to.node_groups[0]
        SMOOTH_BY_ANGLE_NODE_GROUP_NAME = smooth_by_angle_node_group.name

    modifier = obj.modifiers.new("Smooth by Angle", "NODES")
    modifier.node_group = smooth_by_angle_node_group

def has_svg2blender():
    return addon_utils.check("svg2blender_importer") == (True, True)

def menu_func_import_pcb3d(self, context):
    self.layout.operator(PCB2BLENDER_OT_import_pcb3d.bl_idname, text="PCB (.pcb3d)")

def menu_func_import_x3d(self, context):
    self.layout.operator(PCB2BLENDER_OT_import_x3d.bl_idname,
        text="X3D/VRML (.x3d/.wrl) (for pcb3d)")

classes = (
    PCB2BLENDER_OT_import_pcb3d,
    PCB2BLENDER_OT_import_x3d,
    PCB2BLENDER_PT_import_transform_x3d,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_x3d)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_pcb3d)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_pcb3d)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_x3d)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
