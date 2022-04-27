import bpy, bmesh, addon_utils
from bpy_extras.io_utils import ImportHelper, orientation_helper, axis_conversion
from bpy.props import *
from mathutils import Vector, Matrix

import tempfile, random, shutil, re, struct, math
from pathlib import Path
from zipfile import ZipFile, BadZipFile, Path as ZipPath
from dataclasses import dataclass
import numpy as np

from cairosvg import svg2png
from PIL import Image, ImageOps

from .materials import setup_pcb_material, merge_materials, enhance_materials

from io_scene_x3d import ImportX3D, X3D_PT_import_transform, import_x3d
from io_scene_x3d import menu_func_import as menu_func_import_x3d_original

PCB = "pcb.wrl"
COMPONENTS = "components"
LAYERS = "layers"
LAYERS_BOUNDS = "bounds"
BOARDS = "boards"
BOUNDS = "bounds"
STACKED = "stacked_"

INCLUDED_LAYERS = (
    "F_Cu", "B_Cu", "F_Paste", "B_Paste", "F_SilkS", "B_SilkS", "F_Mask", "B_Mask"
)

REQUIRED_MEMBERS = {PCB, LAYERS}

PCB_THICKNESS = 1.6 # mm

class PCB2BLENDER_OT_import_pcb3d(bpy.types.Operator, ImportHelper):
    """Import a PCB3D file"""
    bl_idname = "pcb2blender.import_pcb3d"
    bl_label = "Import .pcb3d"
    bl_options = {"PRESET", "UNDO"}

    import_components: BoolProperty(name="Import Components", default=True)
    center_pcb:        BoolProperty(name="Center PCB", default=True)
    
    merge_materials:   BoolProperty(name="Merge Materials", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)
    pcb_material:      EnumProperty(name="PCB Material", default="RASTERIZED",
        items=(("RASTERIZED", "Rasterized", ""), ("3D", "3D", "")))

    cut_boards:        BoolProperty(name="Cut PCBs", default=True)
    stack_boards:      BoolProperty(name="Stack PCBs", default=True)

    texture_dpi:       FloatProperty(name="Texture DPI", default=2032.0, soft_min=508.0, soft_max=4064.0)

    import_fpnl:       BoolProperty(name="Import Frontpanel (.fpnl)", default=True,
        description="Import the specified .fpnl file and align it (if its stacked to a pcb).")
    fpnl_path:         StringProperty(name="", subtype="FILE_PATH",
        description="")
    
    fpnl_thickness:    FloatProperty(name="Panel Thickness (mm)", default=2.0, soft_min=0.0, soft_max=5.0)
    fpnl_bevel_depth:  FloatProperty(name="Bevel Depth (mm)", default=0.05, soft_min=0.0, soft_max=0.25)
    fpnl_setup_camera: BoolProperty(name="Setup Orthographic Camera", default=True)

    filter_glob:       StringProperty(default="*.pcb3d", options={"HIDDEN"})

    def __init__(self):
        self.last_fpnl_path = ""
        super().__init__()

    def execute(self, context):
        props = self.properties

        filepath = Path(props.filepath)

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
            filepath=str(tempdir / PCB), scale=1.0, join=False, enhance_materials=False)
        pcb_objects = set(bpy.data.objects).difference(objects_before)
        pcb_objects = sorted(pcb_objects, key=lambda obj: obj.name)

        for obj in pcb_objects:
            obj.data.transform(Matrix.Diagonal((*obj.scale, 1)))
            obj.scale = (1, 1, 1)

            self.setup_uvs(obj, pcb.layers_bounds)

        # rasterize/import layer svgs

        if self.pcb_material == "RASTERIZED":
            for layer in INCLUDED_LAYERS:
                svg_path = str(tempdir / LAYERS / f"{layer}.svg")
                png_path = str(tempdir / LAYERS / f"{layer}.png")
                svg2png(url=svg_path, write_to=png_path, dpi=self.texture_dpi, negate_colors=True)

            images = {}
            for f_layer, b_layer in zip(INCLUDED_LAYERS[0::2], INCLUDED_LAYERS[1::2]):
                layer = f_layer[2:]
                front = Image.open(tempdir / LAYERS / f"{f_layer}.png").getchannel("R")
                back  = Image.open(tempdir / LAYERS / f"{b_layer}.png").getchannel("R")
                empty = Image.new("L", front.size)

                if layer == "Mask":
                    front = ImageOps.invert(front)
                    back  = ImageOps.invert(back)

                png_path = tempdir / LAYERS / f"{layer}.png"
                merged = Image.merge("RGB", (front, back, empty))
                merged.save(png_path)

                image = bpy.data.images.load(str(png_path))
                image.colorspace_settings.name = "Non-Color"
                image.pack()

                images[layer] = image

        # import components

        component_map = {}
        if self.import_components:
            for component in pcb.components:
                bpy.ops.pcb2blender.import_x3d(
                    filepath=str(tempdir / component), scale=1.0, enhance_materials=False)
                obj = context.object
                obj.data.name = component.rsplit("/", 1)[1].rsplit(".", 1)[0]
                obj.data.transform(MATRIX_FIX_SCALE)
                component_map[component] = obj.data
                bpy.data.objects.remove(obj)

        pcb_materials = set(bpy.data.materials) - materials_before

        shutil.rmtree(tempdir)

        # enhance pcb

        can_enhance = len(pcb_objects) == len(PCB2_LAYER_NAMES)
        if can_enhance:
            layers = dict(zip(PCB2_LAYER_NAMES, pcb_objects))
            for name, obj in layers.items():
                obj.data.materials[0].name = name

            board = layers["Board"]
            self.improve_board_mesh(board.data)
        else:
            self.warning(f"cannot enhance pcb"\
                f"(imported {len(pcb_objects)} layers, expected {len(PCB2_LAYER_NAMES)})")

        pcb_meshes = {obj.data for obj in pcb_objects if obj.type == "MESH"}

        if self.pcb_material == "3D":
            if can_enhance:
                self.enhance_pcb_layers(context, layers)
                pcb_objects = list(layers.values())

            pcb_object = pcb_objects[0]
            bpy.ops.object.select_all(action="DESELECT")
            for obj in pcb_objects:
                obj.select_set(True)
            context.view_layer.objects.active = pcb_object
            bpy.ops.object.join()
            bpy.ops.object.transform_apply()

        else:
            for obj in pcb_objects[1:]:
                bpy.data.objects.remove(obj)
            pcb_object = pcb_objects[0]

            pcb_object.data.transform(Matrix.Diagonal((1, 1, 1.015, 1)))

            board_material = pcb_object.data.materials[0]
            setup_pcb_material(board_material.node_tree, images)

            bpy.ops.object.select_all(action="DESELECT")
            pcb_object.select_set(True)
            context.view_layer.objects.active = pcb_object

        for mesh in pcb_meshes:
            if not mesh.users:
                bpy.data.meshes.remove(mesh)

        # cut boards

        if pcb.boards and self.cut_boards:
            pcb_mesh = pcb_object.data
            bpy.data.objects.remove(pcb_object)
            for name, board in pcb.boards.items():
                board_obj = bpy.data.objects.new(f"PCB_{name}", pcb_mesh.copy())
                context.collection.objects.link(board_obj)
                boundingbox = self.get_boundingbox(context, board.bounds)

                self.cut_object(context, board_obj, boundingbox, "INTERSECT")

                # get rid of the bounding box if it got merged into the board for some reason
                # also reapply board edge vcs on the newly cut edge faces
                bm = bmesh.new()
                bm.from_mesh(board_obj.data)

                for bb_vert in boundingbox.data.vertices:
                    for vert in reversed(bm.verts[:]):
                        if (bb_vert.co - vert.co).length_squared < 1e-8:
                            bm.verts.remove(vert)
                            break

                board_edge = bm.loops.layers.color[0]
                for bb_face in boundingbox.data.polygons:
                    point = boundingbox.data.vertices[bb_face.vertices[0]].co
                    board_faces = (face for face in bm.faces if face.material_index == 0)
                    faces = self.get_overlapping_faces(board_faces, point, bb_face.normal)
                    for face in faces:
                        for loop in face.loops:
                            loop[board_edge] = (1, 1, 1, 1)

                bm.to_mesh(board_obj.data)
                bm.free()

                bpy.data.objects.remove(boundingbox)

                offset = 0.001 * board.bounds[0].to_3d()
                board_obj.data.transform(Matrix.Translation(-offset))
                board_obj.location = offset

                board.obj = board_obj
        else:
            pcb_object.name = pcb_object.data.name = "PCB"

        # populate components

        if self.import_components and component_map:
            match = regex_filter_components.search(pcb.content)
            matrix_all = match2matrix(match)
            
            for match_instance in regex_component.finditer(match.group("instances")):
                matrix_instance = match2matrix(match_instance)
                url = match_instance.group("url")

                component = component_map[url]
                instance = bpy.data.objects.new(component.name, component)
                instance.matrix_world = matrix_all @ matrix_instance @ MATRIX_FIX_SCALE_INV
                context.collection.objects.link(instance)

                if pcb.boards:
                    partial_matches = []
                    for board in pcb.boards.values():
                        x, y = instance.location.xy * 1000
                        p_min, p_max = board.bounds

                        in_bounds_x = x >= p_min.x and x < p_max.x
                        in_bounds_y = y <= p_min.y and y > p_max.y
                        if in_bounds_x and in_bounds_y:
                            instance.parent = board.obj
                            instance.location -= p_min.to_3d() * 0.001
                            break
                        elif in_bounds_x or in_bounds_y:
                            partial_matches.append((board.obj, p_min.to_3d() * 0.001))
                    else:
                        if len(partial_matches) == 1:
                            instance.parent = partial_matches[0][0]
                            instance.location -= partial_matches[0][1]
                            continue

                        closest = None
                        min_distance = math.inf
                        for name, board in pcb.boards.items():
                            center = (board.bounds[0] + board.bounds[1]) * 0.5
                            distance = (instance.location.xy * 1000 - center).length_squared
                            if distance < min_distance:
                                min_distance = distance
                                closest = (name, board)

                        name, board = closest
                        instance.parent = board.obj
                        instance.location -= board.bounds[0].to_3d() * 0.001
                        self.warning(
                            f"assigning component \"{component.name}\" (out of bounds) " \
                            f"to closest board \"{name}\""
                        )
                else:
                    instance.parent = pcb_object

        # import front panel

        if has_svg2blender() and self.import_fpnl and self.fpnl_path != "":
            if Path(self.fpnl_path).is_file():
                bpy.ops.svg2blender.import_fpnl(
                    filepath=self.fpnl_path,
                    thickness=self.fpnl_thickness,
                    bevel_depth=self.fpnl_bevel_depth,
                    setup_camera=self.fpnl_setup_camera
                )

                pcb.boards["FPNL"] = [(Vector((0, 0)), None), [], context.object]
            else:
                self.warning(f"frontpanel file \"{filepath}\" does not exist")
        
        # stack boards
        
        if self.stack_boards:
            for board in pcb.boards.values():
                for (name, offset) in board.stacked_boards:
                    if not name in pcb.boards:
                        self.warning(f"ignoring stacked board \"{name}\" (unknown board)")
                        continue

                    stacked_obj = pcb.boards[name].obj
                    stacked_obj.parent = board.obj

                    pcb_offset = Vector((0, 0, np.sign(offset.z) * PCB_THICKNESS))
                    if name == "FPNL":
                        pcb_offset.z += (self.fpnl_thickness - PCB_THICKNESS) * 0.5
                    stacked_obj.location = (offset + pcb_offset) * 0.001

        # select pcb objects and make one active

        if pcb.boards:
            bpy.ops.object.select_all(action="DESELECT")
            top_level_boards = [board for board in pcb.boards.values() if not board.obj.parent]
            context.view_layer.objects.active = top_level_boards[0].obj
            for board in top_level_boards:
                board.obj.select_set(True)

        # center pcbs

        if self.center_pcb:
            if pcb.boards:
                center = Vector((0, 0))
                for board in top_level_boards:
                    center += (board.bounds[0] + board.bounds[1]) * 0.5 * 0.001
                center /= len(top_level_boards)

                matrix = Matrix.Translation(-center.to_3d())
                for board in top_level_boards:
                    self.apply_transformation(board.obj, matrix)
            else:
                center = Vector(pcb_object.bound_box[0]) + pcb_object.dimensions * 0.5
                matrix = Matrix.Translation(-center)
                self.apply_transformation(pcb_object, matrix)

        # materials

        if self.merge_materials:
            merge_materials(component_map.values())

        for material in pcb_materials.copy():
            if not material.users:
                pcb_materials.remove(material)
                bpy.data.materials.remove(material)

        if self.enhance_materials:
            enhance_materials(pcb_materials)

        return {"FINISHED"}

    def parse_pcb3d(self, file, extract_dir):
        zip_path = ZipPath(file)

        with file.open(PCB) as pcb_file:
            pcb_file_content = pcb_file.read().decode("UTF-8")
            with open(extract_dir / PCB, "wb") as filtered_file:
                filtered = regex_filter_components.sub("\g<prefix>", pcb_file_content)
                filtered_file.write(filtered.encode("UTF-8"))

        components = list({
            name for name in file.namelist()
            if name.startswith(f"{COMPONENTS}/") and name.endswith(".wrl")
        })
        file.extractall(extract_dir, components)

        layers = (f"{LAYERS}/{layer}.svg" for layer in INCLUDED_LAYERS)
        file.extractall(extract_dir, layers)

        layers_bounds_path = zip_path / LAYERS / LAYERS_BOUNDS
        layers_bounds = struct.unpack("!ffff", layers_bounds_path.read_bytes())

        boards = {}
        for board_dir in (zip_path / BOARDS).iterdir():
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
                if path.name.startswith(STACKED):
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

        return PCB3D(pcb_file_content, components, layers_bounds, boards)

    @staticmethod
    def get_boundingbox(context, bounds):
        name = "pcb2blender_bounds_tmp"
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)

        margin = 0.01

        size = Vector((1.0, -1.0, 1.0)) * (bounds[1] - bounds[0]).to_3d()
        scale = 0.001 * (size + 2.0 * Vector((margin, margin, 5.0)))
        translation = 0.001 * (bounds[0] - Vector.Fill(2, margin)).to_3d()
        matrix_scale = Matrix.Diagonal(scale).to_4x4()
        matrix_offset = Matrix.Translation(translation)
        bounds_matrix = matrix_offset @ matrix_scale @ Matrix.Translation((0.5, -0.5, 0))

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, matrix=bounds_matrix)
        bm.to_mesh(obj.data)
        bm.free()

        return obj

    @staticmethod
    def setup_uvs(obj, layers_bounds):
        mesh = obj.data

        vertices = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", vertices)
        vertices = vertices.reshape((len(mesh.vertices), 3))

        indices = np.empty(len(mesh.loops), dtype=int)
        mesh.loops.foreach_get("vertex_index", indices)

        offset = np.array((layers_bounds[0], -layers_bounds[1]))
        size = np.array((layers_bounds[2], layers_bounds[3]))
        uvs = (1e3 * vertices[:,:2][indices] - offset) / size + np.array((0, 1))

        uv_layer = mesh.uv_layers[0]
        uv_layer.data.foreach_set("uv", uvs.flatten())

    @staticmethod
    def improve_board_mesh(mesh):
        # fill holes in board mesh to make subsurface shading work
        # create vertex color layer for board edge

        bm = bmesh.new()
        bm.from_mesh(mesh)

        board_edge = bm.loops.layers.color.new("Board Edge")

        for face in bm.faces:
            color = ((1, 1, 1, 1) if abs(face.normal.z) < 1e-3 and face.calc_area() > 1e-10
                else (0, 0, 0, 1))
            for loop in face.loops:
                loop[board_edge] = color

        n_upper_verts = len(bm.verts) // 2
        bm.verts.ensure_lookup_table()
        for i, vert in enumerate(bm.verts[:n_upper_verts]):
            other_vert = bm.verts[n_upper_verts + i]
            try:
                bm.edges.new((vert, other_vert))
            except ValueError:
                pass

        bmesh.ops.holes_fill(bm, edges=bm.edges[:])

        bm.to_mesh(mesh)
        bm.free()

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
    def get_overlapping_faces(bm_faces, point, normal):
        overlapping_faces = []
        for face in bm_faces:
            if (1.0 - normal.dot(face.normal)) > 1e-4:
                continue

            direction = point - face.verts[0].co
            distance = abs(
                direction.normalized().dot(face.normal) * direction.length)
            if distance > 1e-4:
                continue

            overlapping_faces.append(face)

        return overlapping_faces

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "import_components")
        layout.prop(self, "center_pcb")
        layout.split()
        layout.prop(self, "merge_materials")
        layout.prop(self, "enhance_materials")
        layout.split()
        layout.prop(self, "cut_boards")
        layout.prop(self, "stack_boards")
        layout.split()

        layout.label(text="PCB Material")
        layout.prop(self, "pcb_material", text="")
        if self.pcb_material == "RASTERIZED":
            layout.prop(self, "texture_dpi", slider=True)

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
            directory = Path(filebrowser_params.directory.decode())

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

    def error(self, msg):
        print(f"error: {msg}")
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    def warning(self, msg):
        print(f"warning: {msg}")
        self.report({"WARNING"}, msg)

@dataclass
class Board:
    bounds: tuple[Vector, Vector]
    stacked_boards: list[tuple[str, Vector]]
    obj: bpy.types.Object = None

@dataclass
class PCB3D:
    content: str
    components: list[str]
    layers_bounds: tuple[float, float, float, float]
    boards: dict[str, Board]

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

MATRIX_FIX_SCALE = Matrix.Scale(2.54e-3, 4)
MATRIX_FIX_SCALE_INV = MATRIX_FIX_SCALE.inverted()

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
    filter_glob: StringProperty(default="*.x3d;*.wrl", options={"HIDDEN"})

    join:              BoolProperty(name="Join Shapes", default=True)
    tris_to_quads:     BoolProperty(name="Tris to Quads", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)
    scale:             FloatProperty(name="Scale", default=0.001)

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        objects_before = set(bpy.data.objects)
        matrix = axis_conversion(from_forward=self.axis_forward, from_up=self.axis_up).to_4x4()
        result = import_x3d.load(context, self.filepath, global_matrix=matrix)
        if not result == {"FINISHED"}:
            return result
        objects = list(set(bpy.data.objects).difference(objects_before))

        for obj in objects:
            obj.scale *= self.scale
            obj.select_set(True)
        context.view_layer.objects.active = objects[0]

        bpy.ops.object.shade_smooth()
        for obj in objects:
            if obj.type == "MESH":
                obj.data.use_auto_smooth = True

        if self.join:
            meshes = {obj.data for obj in objects if obj.type == "MESH"}
            bpy.ops.object.join()
            for mesh in meshes:
                if not mesh.users:
                    bpy.data.meshes.remove(mesh)

            joined_obj = context.object
            joined_obj.name = Path(self.filepath).name.rsplit(".", 1)[0]
            joined_obj.data.name = joined_obj.name
            objects = [joined_obj]

        if self.tris_to_quads:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.tris_convert_to_quads()
            bpy.ops.object.mode_set(mode="OBJECT")

        if self.enhance_materials:
            merge_materials([obj.data for obj in objects])
            enhance_materials(sum((obj.data.materials[:] for obj in objects), []))

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "join")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "enhance_materials")
        layout.split()
        layout.prop(self, "scale")

bases = X3D_PT_import_transform.__bases__
namespace = dict(X3D_PT_import_transform.__dict__)
del namespace["bl_rna"]
X3D_PT_import_transform_copy = type("X3D_PT_import_transform_copy", bases, namespace)
class PCB2BLENDER_PT_import_transform_x3d(X3D_PT_import_transform_copy):
    @classmethod
    def poll(cls, context):
        return context.space_data.active_operator.bl_idname == "PCB2BLENDER_OT_import_x3d"

def has_svg2blender():
    return addon_utils.check("svg2blender-importer") == (True, True)

def menu_func_import_pcb3d(self, context):
    self.layout.operator(PCB2BLENDER_OT_import_pcb3d.bl_idname, text="PCB (.pcb3d)")

def menu_func_import_x3d(self, context):
    self.layout.operator(PCB2BLENDER_OT_import_x3d.bl_idname,
        text="X3D Extensible 3D (.x3d/.wrl)")

classes = (
    PCB2BLENDER_OT_import_pcb3d,
    PCB2BLENDER_OT_import_x3d,
    PCB2BLENDER_PT_import_transform_x3d,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_x3d_original)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_x3d)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_pcb3d)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_pcb3d)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_x3d)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_x3d_original)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
