import io
import random
import re
import shutil
import struct
import sys
import tempfile
from math import inf, radians
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast
from zipfile import BadZipFile, Path as ZipPath, ZipFile

from error_helper import error, warning
from PIL import Image, ImageOps
from skia import SVGDOM, Color4f, Stream, Surface

import addon_utils
import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Mesh
from bpy_extras.io_utils import ImportHelper, axis_conversion, orientation_helper
from mathutils import Matrix, Vector

from .io_scene_x3d.source import ImportX3D, X3D_PT_import_transform, import_x3d
from .materials import (
    LAYER_BOARD_EDGE,
    LAYER_THROUGH_HOLES,
    enhance_materials,
    merge_materials,
    setup_pcb_material,
)
from .pcb3d import PCB3D, Board, Bounds, DrillShape, PadFabType, PadShape, PadType

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
else:
    OperatorReturnItems = str


T = TypeVar("T", bound=bpy.types.ID | None)


class Object(Generic[T], bpy.types.Object):
    data: T  # pyright: ignore[reportIncompatibleVariableOverride]


ENABLE_PROFILER = False


def has_debugger_attached():
    return sys.gettrace() is not None


class PCB2BLENDER_OT_import_pcb3d(ImportHelper, bpy.types.Operator):
    """Import a PCB3D file"""

    bl_idname = "pcb2blender.import_pcb3d"
    bl_label = "Import .pcb3d"
    bl_options = {"PRESET", "UNDO"}

    import_components: BoolProperty(name="Import Components", default=True)
    add_solder_joints: EnumProperty(
        name="Add Solder Joints",
        default="SMART",
        items=(
            ("NONE", "None", "Do not add any solder joints"),
            (
                "SMART",
                "Smart",
                "Only add solder joints to footprints that have THT/SMD attributes set and that "
                "have 3D models and only to pads which have a solder paste layer (for SMD pads)",
            ),
            ("ALL", "All", "Add solder joints to all pads"),
        ),
    )

    center_boards: BoolProperty(name="Center PCBs", default=True)
    cut_boards: BoolProperty(name="Cut PCBs", default=True)
    stack_boards: BoolProperty(name="Stack PCBs", default=True)

    merge_materials: BoolProperty(name="Merge Materials", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)
    pcb_material: EnumProperty(
        name="PCB Material",
        default="RASTERIZED",
        items=(
            ("RASTERIZED", "Rasterized (Cycles)", ""),
            ("3D", "3D (deprecated)", ""),
        ),
    )
    texture_dpi: FloatProperty(
        name="Texture DPI", default=1016.0, min=0.0, soft_min=508.0, soft_max=2032.0
    )

    import_fpnl: BoolProperty(
        name="Import Frontpanel (.fpnl)",
        default=True,
        description="Import the specified .fpnl file and align it (if its stacked to a pcb).",
    )
    fpnl_path: StringProperty(name="", subtype="FILE_PATH", description="")
    fpnl_thickness: FloatProperty(name="Panel Thickness (mm)", default=2.0, min=0.0, soft_max=5.0)
    fpnl_bevel_depth: FloatProperty(name="Bevel Depth (mm)", default=0.05, min=0.0, soft_max=0.25)
    fpnl_setup_camera: BoolProperty(name="Setup Orthographic Camera", default=True)

    filter_glob: StringProperty(default="*.pcb3d", options={"HIDDEN"})

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_fpnl_path = ""
        self.board_objects: dict[str, Object[Mesh]] = {}
        self.component_cache: dict[str, Mesh] = {}
        self.new_materials = set()

    def execute(self, context: bpy.types.Context) -> set[OperatorReturnItems]:
        assert context.view_layer and context.scene

        profiler = None
        if ENABLE_PROFILER and has_debugger_attached():
            from cProfile import Profile

            profiler = Profile()
            profiler.enable()

        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # import boards

        filepath = Path(self.filepath)
        if not isinstance(result := self.import_pcb3d(context, filepath), PCB3D):
            return result
        pcb = result

        # import front panel

        if has_svg2blender() and self.import_fpnl and self.fpnl_path != "":
            if Path(self.fpnl_path).is_file():
                bpy.ops.svg2blender.import_fpnl(  # pyright: ignore[reportAttributeAccessIssue]
                    filepath=self.fpnl_path,
                    thickness=self.fpnl_thickness,
                    bevel_depth=self.fpnl_bevel_depth,
                    setup_camera=self.fpnl_setup_camera,
                )
                pcb.boards["FPNL"] = Board(Bounds((0.0, 0.0), (0.0, 0.0)), {})
                self.board_objects["FPNL"] = cast(Object[Mesh], context.object)
            else:
                self.warning(f'frontpanel file "{filepath}" does not exist')

        # stack boards

        if self.stack_boards:
            for board_name, board in pcb.boards.items():
                for stacked_name, offset in board.stacked_boards.items():
                    if stacked_name not in pcb.boards:
                        self.warning(f'ignoring stacked board "{stacked_name}" (unknown board)')
                        continue

                    if not (stacked_obj := self.board_objects[stacked_name]):
                        self.warning(
                            f'ignoring stacked board "{stacked_name}" (cut_boards is set to False)'
                        )
                        continue

                    stacked_obj.parent = self.board_objects[board_name]

                    offset = Vector(offset) * Vector((1, -1, 1))
                    pcb_offset = Vector((0, 0, np.sign(offset.z) * pcb.stackup.thickness_mm))
                    if stacked_name == "FPNL":
                        pcb_offset.z += (self.fpnl_thickness - pcb.stackup.thickness_mm) * 0.5
                    stacked_obj.location = (offset + pcb_offset) * MM_TO_M

        # select pcb objects and make one active

        bpy.ops.object.select_all(action="DESELECT")
        top_level_boards = {name: obj for name, obj in self.board_objects.items() if not obj.parent}
        for obj in top_level_boards.values():
            obj.select_set(True)
        context.view_layer.objects.active = next(iter(top_level_boards.values()))

        # center boards

        if self.center_boards:
            center = Vector((0, 0))
            for board_name in top_level_boards:
                center += Vector(pcb.boards[board_name].bounds.center)
            center /= len(top_level_boards)

            for board_name, obj in top_level_boards.items():
                location = Vector(pcb.boards[board_name].bounds.top_left) - center
                obj.location.xy = location * Vector((1, -1)) * MM_TO_M

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

        if profiler:
            profiler.disable()
            profiler.dump_stats(
                Path(__file__).parent.resolve() / Path(__file__).with_suffix(".prof").name
            )

        return {"FINISHED"}

    def import_pcb3d(
        self, context: bpy.types.Context, filepath: Path
    ) -> PCB3D | set[OperatorReturnItems]:
        assert context.collection and context.view_layer

        if not filepath.is_file():
            return self.error(f'file "{filepath}" does not exist')

        dirname = filepath.name.replace(".", "_") + f"_{random.getrandbits(64)}"
        tempdir = Path(tempfile.gettempdir()) / "pcb2blender_tmp" / dirname
        tempdir.mkdir(parents=True, exist_ok=True)

        try:
            with ZipFile(filepath) as file:
                MEMBERS = {path.name for path in ZipPath(file).iterdir()}
                if missing := PCB3D.REQUIRED_MEMBERS.difference(MEMBERS):
                    return self.error(f"not a valid .pcb3d file: missing {str(missing)[1:-1]}")
                result = PCB3D.from_file(
                    file, tempdir, on_error=self.error, on_warning=self.warning
                )
                if not isinstance(result, PCB3D):
                    return result
        except BadZipFile:
            return self.error("not a valid .pcb3d file: not a zip file")
        except (KeyError, struct.error) as e:
            return self.error(f"pcb3d file is corrupted: {e}")
        pcb = result

        # import objects

        materials_before = set(bpy.data.materials)

        objects_before = set(bpy.data.objects)
        bpy.ops.pcb2blender.import_x3d(  # pyright: ignore[reportAttributeAccessIssue]
            filepath=str(tempdir / PCB3D.PCB), global_scale=1.0, join=False, enhance_materials=False
        )
        pcb_objects = set(bpy.data.objects).difference(objects_before)
        pcb_objects = cast(list[Object[Mesh]], sorted(pcb_objects, key=lambda obj: obj.name))

        for obj in pcb_objects:
            obj.data.transform(Matrix.Diagonal((*obj.scale, 1)))
            obj.scale = (1, 1, 1)
            self.setup_uvs(obj, pcb.layers_bounds)

        # rasterize/import layer svgs

        images = {}
        if self.enhance_materials and self.pcb_material == "RASTERIZED":
            layers_path = tempdir / PCB3D.LAYERS
            dpi = self.texture_dpi
            for f_layer, b_layer in zip(PCB3D.INCLUDED_LAYERS[0::2], PCB3D.INCLUDED_LAYERS[1::2]):
                front = self.svg2img(layers_path / f"{f_layer}.svg", dpi).getchannel(0)
                back = self.svg2img(layers_path / f"{b_layer}.svg", dpi).getchannel(0)

                if (layer := f_layer[2:]) != "Mask":
                    front = ImageOps.invert(front)
                    back = ImageOps.invert(back)

                empty = Image.new("L", front.size)
                merged = Image.merge("RGB", (front, back, empty))
                png_path = layers_path / f"{filepath.stem}_{layer}.png"
                merged.save(png_path)

                image = bpy.data.images.load(str(png_path))
                assert image.colorspace_settings
                image.colorspace_settings.name = "Non-Color"  # pyright: ignore[reportAttributeAccessIssue]
                image.pack()
                image.filepath = ""

                images[layer] = image

        # import components

        if self.import_components:
            for component in tempdir.glob(f"{PCB3D.COMPONENTS}/*.wrl"):
                bpy.ops.pcb2blender.import_x3d(filepath=str(component), enhance_materials=False)  # pyright: ignore[reportAttributeAccessIssue]
                obj = cast(Object[Mesh], context.object)
                obj.data.name = component.stem
                self.component_cache[f"{PCB3D.COMPONENTS}/{component.name}"] = obj.data
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

            assert (board_material := pcb_object.data.materials[0]) and board_material.node_tree
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
                assert (material := pcb_object.data.materials[0])
                material.name = name

            top_left = Vector(pcb_object.bound_box[3]).xy
            size = Vector(pcb_object.bound_box[5]).xy - top_left
            matrix = Matrix.Translation(top_left.to_3d())
            pcb_object.data.transform(matrix.inverted())
            pcb_object.matrix_world = matrix @ pcb_object.matrix_world

            top_left *= M_TO_MM
            size *= M_TO_MM
            pcb.boards.clear()
            pcb.boards[name] = Board(Bounds((top_left[0], -top_left[1]), (size[0], -size[1])), {})
            self.board_objects[name] = pcb_object

        else:
            pcb_mesh = pcb_object.data
            bpy.data.objects.remove(pcb_object)
            for name, board in pcb.boards.items():
                board_obj = cast(Object[Mesh], bpy.data.objects.new(f"PCB_{name}", pcb_mesh.copy()))
                context.collection.objects.link(board_obj)

                cut_material_index = len(board_obj.material_slots) + 1
                boundingbox = self.get_boundingbox(context, board.bounds, cut_material_index)
                self.cut_object(context, board_obj, boundingbox, "INTERSECT")
                bpy.data.objects.remove(boundingbox)

                # cleanup and reapply board edge vcs on the newly cut edge faces
                bm = bmesh.new()
                bm.from_mesh(board_obj.data)

                bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-8)
                bmesh.ops.triangulate(bm, faces=bm.faces[:])

                board_edge = bm.faces.layers.int[LAYER_BOARD_EDGE]  # pyright: ignore[reportIndexIssue]
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
                merge_distance_sq = merge_distance**2

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

                offset = Vector(board.bounds.top_left).to_3d() * Vector((1, -1, 1)) * MM_TO_M
                board_obj.data.transform(Matrix.Translation(-offset))
                board_obj.location = offset

                self.board_objects[name] = board_obj

            bpy.ops.object.select_all(action="DESELECT")
            for obj in self.board_objects.values():
                obj.select_set(True)
            context.view_layer.objects.active = next(iter(self.board_objects.values()))

        # fix smooth shading issues
        bpy.ops.object.shade_smooth_by_angle(angle=radians(89), keep_sharp_edges=False)

        related_objects: list[Object[Mesh]] = []

        # populate components

        if self.import_components and self.component_cache:
            assert (match := PCB3D.REGEX_FILTER_COMPONENTS.search(pcb.content))
            matrix_all = match2matrix(match)

            for match_instance in PCB3D.REGEX_COMPONENT.finditer(match.group("instances")):
                matrix_instance = match2matrix(match_instance)
                url = match_instance.group("url")

                component = self.component_cache[url]
                instance = cast(Object[Mesh], bpy.data.objects.new(component.name, component))
                add_smooth_by_angle_modifier(instance)
                instance.matrix_world = matrix_all @ matrix_instance @ MATRIX_FIX_SCALE_INV
                context.collection.objects.link(instance)
                related_objects.append(instance)

        # add solder joints

        solder_joint_cache: dict[Any, Object[Mesh]] = {}
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
                if pad.fab_type not in {PadFabType.NONE, PadFabType.BGA, PadFabType.CASTELLATED}:
                    continue

                pad_type = pad.pad_type.name
                pad_size = pad.size
                hole_shape = pad.drill_shape.name
                hole_size = pad.drill_size
                match pad.shape:
                    case PadShape.RECT:
                        roundness = 0.0
                    case PadShape.CIRCLE:
                        pad_size = (pad.size[0], pad.size[0])
                        roundness = 1.0
                    case PadShape.OVAL:
                        roundness = 1.0
                    case PadShape.ROUNDRECT:
                        roundness = pad.roundness * 2.0
                    case PadShape.TRAPEZOID | PadShape.CHAMFERED_RECT | PadShape.CUSTOM:
                        print(
                            f"skipping solder joint for '{pad_name}', "
                            f"unsupported shape '{pad.shape.name}'"
                        )
                        continue

                cache_id = (pad_type, pad_size, hole_shape, hole_size, roundness)
                if not (solder_joint := solder_joint_cache.get(cache_id)):
                    bpy.ops.pcb2blender.solder_joint_add(  # pyright: ignore[reportAttributeAccessIssue]
                        pad_type=pad_type,
                        pad_shape="RECTANGULAR",
                        pad_size=pad_size,
                        roundness=roundness,
                        hole_shape=hole_shape,
                        hole_size=hole_size,
                        pcb_thickness=pcb.stackup.thickness_mm,
                        reuse_material=True,
                    )
                    solder_joint = cast(Object[Mesh], context.object)
                    solder_joint_cache[cache_id] = solder_joint

                obj = solder_joint.copy()
                obj.name = f"SOLDER_{pad_name}"
                obj.location.xy = Vector(pad.position) * Vector((1, -1)) * MM_TO_M
                obj.rotation_euler.z = pad.rotation
                obj.scale.z *= 1.0 if pad.is_flipped ^ (pad.pad_type == PadType.SMD) else -1.0
                context.collection.objects.link(obj)
                related_objects.append(obj)

        for obj in solder_joint_cache.values():
            bpy.data.objects.remove(obj)

        if not has_multiple_boards:
            pcb_board = next(iter(pcb.boards.values()))
            for obj in related_objects:
                obj.location.xy -= Vector(pcb_board.bounds.top_left) * Vector((1, -1)) * MM_TO_M
                obj.parent = next(iter(self.board_objects.values()))
        else:
            for obj in related_objects:
                x, y = xy = obj.location.xy * Vector((1, -1)) * M_TO_MM
                for board_name, board in pcb.boards.items():
                    p_min = board.bounds.top_left
                    p_max = board.bounds.bottom_right
                    if x >= p_min[0] and x < p_max[0] and y >= p_min[1] and y < p_max[1]:
                        parent_board = board_name
                        break
                else:
                    parent_board = ""
                    min_distance = inf
                    for board_name, board in pcb.boards.items():
                        distance = (xy - Vector(board.bounds.center)).length_squared
                        if distance < min_distance:
                            min_distance = distance
                            parent_board = board_name

                    self.warning(
                        f'assigning "{obj.name}" (out of bounds) to closest board "{parent_board}"'
                    )

                obj.location.xy -= (
                    Vector(pcb.boards[parent_board].bounds.top_left) * Vector((1, -1)) * MM_TO_M
                )
                obj.parent = self.board_objects[parent_board]

        return pcb

    @staticmethod
    def get_boundingbox(context: bpy.types.Context, bounds: Bounds, material_index: int):
        assert context.collection

        NAME = "pcb2blender_bounds_tmp"
        mesh = bpy.data.meshes.new(NAME)
        obj = cast(Object[Mesh], bpy.data.objects.new(NAME, mesh))
        context.collection.objects.link(obj)

        MARGIN_MM = -0.01
        size = Vector(bounds.size).to_3d()
        scale = size + 2.0 * Vector((MARGIN_MM, MARGIN_MM, 5.0))
        translation = (Vector(bounds.top_left) - Vector.Fill(2, MARGIN_MM)) * Vector((1, -1))
        matrix_scale = Matrix.Diagonal(scale * MM_TO_M).to_4x4()
        matrix_offset = Matrix.Translation(translation.to_3d() * MM_TO_M)
        bounds_matrix = matrix_offset @ matrix_scale @ Matrix.Translation((0.5, -0.5, 0))

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, matrix=bounds_matrix)
        for face in bm.faces:
            face.material_index = material_index
        bm.to_mesh(obj.data)
        bm.free()

        return obj

    @staticmethod
    def setup_uvs(obj: Object[Mesh], layers_bounds: Bounds):
        mesh = obj.data

        vertices = np.empty(len(mesh.vertices) * 3)

        assert isinstance(attribute := mesh.attributes["position"], bpy.types.FloatVectorAttribute)
        attribute.data.foreach_get("vector", vertices)
        vertices = vertices.reshape((len(mesh.vertices), 3))

        indices = np.empty(len(mesh.loops), dtype=int)
        mesh.loops.foreach_get("vertex_index", indices)

        offset = layers_bounds.top_left * np.array((1, -1))
        uvs = (vertices[:, :2][indices] * M_TO_MM - offset) / layers_bounds.size + np.array((0, 1))

        uv_layer = mesh.uv_layers[0]
        uv_layer.uv.foreach_set("vector", uvs.flatten())

    @staticmethod
    def improve_board_mesh(context: bpy.types.Context, obj: Object[Mesh]):
        # fill holes in board mesh to make subsurface shading work
        # create vertex color layer for board edge and through holes

        assert context.view_layer

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

        filled = bmesh.ops.holes_fill(bm, edges=bm.edges[:])

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
    def enhance_pcb_layers(cls, context: bpy.types.Context, layers: dict[str, Object[Mesh]]):
        assert context.collection

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
    def copy_object(obj: Object[Mesh], collection: bpy.types.Collection):
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        collection.objects.link(new_obj)
        return new_obj

    @staticmethod
    def cut_object(
        context: bpy.types.Context,
        obj: Object[Mesh],
        cutter: Object[Mesh],
        mode: Literal["INTERSECT", "UNION", "DIFFERENCE"],
    ):
        assert context.view_layer

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
    def extrude_mesh_z(mesh: Mesh, z: float, symmetric: bool = False):
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
    def translate_mesh_z(mesh: Mesh, z: float):
        mesh.transform(Matrix.Translation((0, 0, z)))

    @staticmethod
    def apply_transformation(obj: Object[Mesh], matrix: Matrix):
        obj.data.transform(matrix)
        for child in obj.children:
            child.matrix_basis = matrix @ child.matrix_basis

    @staticmethod
    def svg2img(svg_path: Path, dpi: float):
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

    def draw(self, context: bpy.types.Context):
        assert isinstance(context.space_data, bpy.types.SpaceFileBrowser)
        assert (layout := self.layout)

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

            assert (filebrowser_params := context.space_data.params)
            filename = Path(filebrowser_params.filename)
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

    def error(self, msg: str, prefix: str = "error: ") -> set[OperatorReturnItems]:
        error(msg, prefix=prefix)
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    def warning(self, msg: str, prefix: str = "warning: "):
        warning(msg, prefix=prefix)
        self.report({"WARNING"}, msg)


PCB2_LAYER_NAMES = [
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
]

MM_TO_M = 1e-3
M_TO_MM = 1e3
INCH_TO_MM = 1 / 25.4

FIX_X3D_SCALE = 2.54 * MM_TO_M
MATRIX_FIX_SCALE_INV = Matrix.Scale(FIX_X3D_SCALE, 4).inverted()

ANGLE_LIMIT = radians(0.1)


def match2matrix(match: re.Match[str]):
    rotation = match.group("r")
    translation = match.group("t")
    scale = match.group("s")

    matrix = Matrix()
    if translation:
        translation = list(map(float, translation.split()))
        matrix = matrix @ Matrix.Translation(translation)
    if rotation:
        rotation = list(map(float, rotation.split()))
        matrix = matrix @ Matrix.Rotation(rotation[3], 4, rotation[:3])
    if scale:
        scale = list(map(float, scale.split()))
        matrix = matrix @ Matrix.Diagonal(scale).to_4x4()

    return matrix


@orientation_helper(axis_forward="Y", axis_up="Z")  # pyright: ignore[reportUntypedClassDecorator, reportOptionalCall]
class PCB2BLENDER_OT_import_x3d(bpy.types.Operator, ImportHelper):
    __doc__ = ImportX3D.__doc__
    bl_idname = "pcb2blender.import_x3d"
    bl_label = ImportX3D.bl_label
    bl_options = {"PRESET", "UNDO"}

    axis_forward: Literal["X", "Y", "Z", "-X", "-Y", "-Z"]
    axis_up: Literal["X", "Y", "Z", "-X", "-Y", "-Z"]

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
        update=ImportX3D._file_unit_update,  # pyright: ignore[reportPrivateUsage]
    )
    global_scale: FloatProperty(
        name="Scale",
        soft_min=0.001,
        soft_max=1000.0,
        default=FIX_X3D_SCALE,
        precision=4,
        step=1,
        description="Scale value used when 'File Unit' is set to 'CUSTOM'",
    )

    join: BoolProperty(name="Join Shapes", default=True)
    tris_to_quads: BoolProperty(name="Tris to Quads", default=True)
    auto_smooth: BoolProperty(name="Auto Smooth", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)

    def execute(self, context: bpy.types.Context) -> set[OperatorReturnItems]:
        assert context.view_layer

        bpy.ops.object.select_all(action="DESELECT")

        objects_before = set(bpy.data.objects)
        matrix = axis_conversion(from_forward=self.axis_forward, from_up=self.axis_up).to_4x4()
        import_x3d.load(
            context, self.filepath, global_scale=self.global_scale, global_matrix=matrix
        )

        if not (objects := list(set(bpy.data.objects).difference(objects_before))):
            return {"FINISHED"}

        for obj in objects.copy():
            if obj.type == "MESH":
                obj.select_set(True)
            else:
                bpy.data.objects.remove(obj)
                objects.remove(obj)
        objects = cast(list[Object[Mesh]], objects)
        context.view_layer.objects.active = objects[0]

        if self.auto_smooth:
            bpy.ops.object.shade_smooth()

        if self.join:
            if len(objects) > 1:
                bpy.ops.object.join()
            bpy.ops.object.transform_apply()

            assert (joined_obj := cast(Object[Mesh] | None, context.object))
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
            meshes = [obj.data for obj in objects]
            materials = sum((list(mesh.materials) for mesh in meshes), [])
            merge_materials(meshes)
            for material in materials[:]:
                if not material.users:
                    materials.remove(material)
                    bpy.data.materials.remove(material)
            enhance_materials(materials)

        return {"FINISHED"}

    def draw(self, context: bpy.types.Context):
        assert (layout := self.layout)

        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "join")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_smooth")
        layout.prop(self, "enhance_materials")


class PCB2BLENDER_PT_import_transform_x3d(X3D_PT_import_transform):
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        space_data = context.space_data
        assert isinstance(space_data, bpy.types.SpaceFileBrowser) and space_data.active_operator
        return space_data.active_operator.bl_idname == "PCB2BLENDER_OT_import_x3d"


# https://projects.blender.org/blender/blender/issues/117399
def get_internal_asset_path():
    for path_type in ("LOCAL", "SYSTEM", "USER"):
        path = Path(bpy.utils.resource_path(path_type)) / "datafiles" / "assets"
        if path.exists():
            return path
    assert False


SMOOTH_BY_ANGLE_ASSET_PATH = str(
    get_internal_asset_path()
    / (
        "nodes/geometry_nodes_essentials.blend"
        if bpy.app.version >= (5, 0, 0)
        else "geometry_nodes/smooth_by_angle.blend"
    )
)
smooth_by_angle_node_group_name = "Smooth by Angle"


def add_smooth_by_angle_modifier(obj: Object[Mesh]):
    global smooth_by_angle_node_group_name

    smooth_by_angle_node_group = bpy.data.node_groups.get(smooth_by_angle_node_group_name)
    if not smooth_by_angle_node_group or smooth_by_angle_node_group.type != "GEOMETRY":
        with bpy.data.libraries.load(SMOOTH_BY_ANGLE_ASSET_PATH) as (_data_from, data_to):
            data_to.node_groups = cast(list[Any], [smooth_by_angle_node_group_name])
        smooth_by_angle_node_group = cast(bpy.types.NodeTree, data_to.node_groups[0])
        smooth_by_angle_node_group_name = smooth_by_angle_node_group.name

    modifier = obj.modifiers.new("Smooth by Angle", "NODES")
    modifier.node_group = smooth_by_angle_node_group
    modifier.show_group_selector = False


def has_svg2blender():
    return addon_utils.check("svg2blender_importer") == (True, True)


def menu_func_import_pcb3d(self: bpy.types.Menu, context: bpy.types.Context):
    assert self.layout
    self.layout.operator(PCB2BLENDER_OT_import_pcb3d.bl_idname, text="PCB (.pcb3d)")


def menu_func_import_x3d(self: bpy.types.Menu, context: bpy.types.Context):
    assert self.layout
    self.layout.operator(
        PCB2BLENDER_OT_import_x3d.bl_idname, text="X3D/VRML (.x3d/.wrl) (for pcb3d)"
    )


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
