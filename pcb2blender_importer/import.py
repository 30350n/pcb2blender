import bpy, bmesh, addon_utils
from bpy_extras.io_utils import ImportHelper
from bpy.props import *
from mathutils import Vector, Matrix

import tempfile, random, shutil, re, struct, math
from pathlib import Path
from zipfile import ZipFile, BadZipFile, Path as ZipPath
import numpy as np

from .materials import *

PCB = "pcb.wrl"
COMPONENTS = "components"
BOARDS = "boards"
BOUNDS = "bounds"
STACKED = "stacked_"

REQUIRED_MEMBERS = {PCB}

PCB_THICKNESS = 1.6 # mm

class PCB2BLENDER_OT_import_pcb3d(bpy.types.Operator, ImportHelper):
    """Load a PCB file."""
    bl_idname = "pcb2blender.import_pcb3d"
    bl_label = "Import .pcb3d"

    import_components: BoolProperty(name="Import Components", default=True)
    center_pcb:        BoolProperty(name="Center PCB", default=True)
    enhance_pcb:       BoolProperty(name="Enhance PCB", default=True)

    merge_materials:   BoolProperty(name="Merge Materials", default=True)
    enhance_materials: BoolProperty(name="Enhance Materials", default=True)

    cut_boards:        BoolProperty(name="Cut PCBs", default=True)
    stack_boards:      BoolProperty(name="Stack PCBs", default=True)

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

        # parse file

        try:
            with ZipFile(filepath) as file:
                zip_path = ZipPath(file)
                MEMBERS = {path.name for path in zip_path.iterdir()}
                if missing := REQUIRED_MEMBERS.difference(MEMBERS):
                    return self.error(f"not a valid .pcb3d file: missing {str(missing)[1:-1]}")

                with file.open(PCB) as pcb_file:
                    pcb_file_content = pcb_file.read().decode("UTF-8")
                    with open(tempdir / PCB, "wb") as filtered_file:
                        filtered = regex_filter_components.sub("\g<prefix>", pcb_file_content)
                        filtered_file.write(filtered.encode("UTF-8"))

                components = set(filter(
                    lambda name: name.startswith(f"{COMPONENTS}/") and name.endswith(".wrl"),
                    file.namelist()
                ))
                file.extractall(tempdir, components)

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
                    stack_paths = [
                        path for path in board_dir.iterdir() if path.name.startswith(STACKED)
                    ]
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

                    boards[board_dir.name] = [bounds, stacked_boards, None]

        except BadZipFile:
            return self.error("not a valid .pcb3d file: not a zip file")

        # import objects

        materials_before = set(bpy.data.materials)

        pcb_objects = self.import_wrl(context, tempdir / PCB, join=False)

        # enhance pcb

        can_enhance = len(pcb_objects) == len(PCB2_LAYER_NAMES)
        if can_enhance:
            layers = dict(zip(PCB2_LAYER_NAMES, pcb_objects))
            for name, obj in layers.items():
                obj.data.materials[0].name = name
        else:
            print(f"warning: cannot enhance pcb"\
                f"(imported {len(pcb_objects)} layers, expected {len(PCB2_LAYER_NAMES)})")

        if can_enhance and self.enhance_pcb:
            for side, direction in reversed(list(zip(("F", "B"), (1, -1)))):
                mask = layers[f"{side}.Mask"]
                copper = layers[f"{side}.Cu"]
                silk = layers[f"{side}.Silk"]
                paste = layers[f"{side}.Paste"]

                pcb_objects.remove(paste)
                bpy.data.objects.remove(paste)

                # split copper layer into tracks and pads

                tracks = copper
                pads = self.copy_object(copper, context.collection)
                pcb_objects.append(pads)

                mask_cutter = self.copy_object(mask, context.collection)
                self.extrude_mesh_z(mask_cutter.data, 0.25, True)
                self.cut_object(context, tracks, mask_cutter, "INTERSECT")
                self.cut_object(context, pads, mask_cutter, "DIFFERENCE")
                bpy.data.objects.remove(mask_cutter)

                # add 3d tracks to mask layer

                self.extrude_mesh_z(mask.data, 0.5 * direction)

                copper_cutter = self.copy_object(tracks, context.collection)

                bpy.ops.object.select_all(action="DESELECT")
                copper_cutter.select_set(True)
                context.view_layer.objects.active = copper_cutter

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.remove_doubles()
                bpy.ops.mesh.intersect_boolean(operation="UNION", use_self=True)
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(5))
                bpy.ops.mesh.quads_convert_to_tris()
                bpy.ops.mesh.beautify_fill()
                bpy.ops.mesh.inset(thickness=0.015, depth=0.008)
                bpy.ops.mesh.remove_doubles(threshold=0.015)
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")

                self.cut_object(context, mask, copper_cutter, "DIFFERENCE")

                bpy.data.objects.remove(copper_cutter)

                bm = bmesh.new()
                bm.from_mesh(mask.data)
                bmesh.ops.delete(bm, geom=[v for v in bm.verts[:] if abs(v.co.z) > 0.8])
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
                bm.to_mesh(mask.data)
                bm.free()

                # remove silkscreen on pads

                pads_cutter = self.copy_object(pads, context.collection)
                self.extrude_mesh_z(pads_cutter.data, 0.25, True)
                self.cut_object(context, silk, pads_cutter, "DIFFERENCE")
                bpy.data.objects.remove(pads_cutter)

                silk.visible_shadow = False

                # align the layers

                self.translate_mesh_z(mask.data, -0.008 * direction)
                self.translate_mesh_z(silk.data, -0.028 * direction)
                self.translate_mesh_z(tracks.data, -0.004 * direction)
                self.translate_mesh_z(pads.data, -0.004 * direction)

            # scale down vias to match the other layers
            vias = layers["Vias"]
            vias.data.transform(Matrix.Diagonal((1, 1, 0.97, 1)))
            vias.data.polygons.foreach_set("use_smooth", [True] * len(vias.data.polygons))

        pcb_object = pcb_objects[0]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in pcb_objects:
            obj.select_set(True)
        context.view_layer.objects.active = pcb_object
        bpy.ops.object.join()
        bpy.ops.object.transform_apply()

        component_map = {}
        if self.import_components:
            for component in components:
                obj = self.import_wrl(context, tempdir / component)
                component_map[component] = obj.data
                bpy.data.objects.remove(obj)

        shutil.rmtree(tempdir)

        pcb_materials = set(bpy.data.materials) - materials_before

        # cut boards

        if boards and self.cut_boards:
            pcb_mesh = pcb_object.data
            bpy.data.objects.remove(pcb_object)
            for name, (bounds, *dummy) in boards.items():
                board_obj = bpy.data.objects.new(f"PCB_{name}", pcb_mesh.copy())
                context.collection.objects.link(board_obj)
                boundingbox = self.get_boundingbox(context, bounds)

                mod_name = "Cut PCB"
                mod = board_obj.modifiers.new(mod_name, type="BOOLEAN")
                mod.operation = "INTERSECT"
                mod.object = boundingbox
                context.view_layer.objects.active = board_obj
                bpy.ops.object.modifier_apply(modifier=mod_name)
                bpy.data.objects.remove(boundingbox)

                offset = 0.001 * bounds[0].to_3d()
                board_obj.data.transform(Matrix.Translation(-offset))
                board_obj.location = offset

                boards[name][2] = board_obj
        else:
            pcb_object.name = pcb_object.data.name = "PCB"

        # populate components

        if self.import_components:
            match = regex_filter_components.search(pcb_file_content)
            matrix_all = match2matrix(match)
            
            for match_instance in regex_component.finditer(match.group("instances")):
                matrix_instance = match2matrix(match_instance)
                url = match_instance.group("url")

                component = component_map[url]
                instance = bpy.data.objects.new(component.name, component)
                instance.matrix_world = matrix_all @ matrix_instance
                context.collection.objects.link(instance)

                if boards:
                    partial_matches = []
                    for (bounds, _, board_obj) in boards.values():
                        x, y = instance.location.xy * 1000
                        p_min, p_max = bounds

                        in_bounds_x = x >= p_min.x and x < p_max.x
                        in_bounds_y = y <= p_min.y and y > p_max.y
                        if in_bounds_x and in_bounds_y:
                            instance.parent = board_obj
                            instance.location -= p_min.to_3d() * 0.001
                            break
                        elif in_bounds_x or in_bounds_y:
                            partial_matches.append((board_obj, p_min.to_3d() * 0.001))
                    else:
                        if len(partial_matches) == 1:
                            instance.parent = partial_matches[0][0]
                            instance.location -= partial_matches[0][1]
                            continue

                        closest = None
                        min_distance = math.inf
                        for (name, board) in boards.items():
                            center = (board[0][0] + board[0][1]) * 0.5
                            distance = (instance.location.xy * 1000 - center).length_squared
                            if distance < min_distance:
                                min_distance = distance
                                closest = (name, board)

                        name, board = closest
                        instance.parent = board[2]
                        instance.location -= board[0][0].to_3d() * 0.001
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

                boards["FPNL"] = [(Vector((0, 0)), None), [], context.object]
            else:
                self.warning(f"frontpanel file \"{filepath}\" does not exist")
        
        # stack boards
        
        if self.stack_boards:
            for (_, stacked_boards, board_obj) in boards.values():
                for (name, offset) in stacked_boards:
                    if not name in boards:
                        self.warning(f"ignoring stacked board \"{name}\" (unknown board)")
                        continue

                    stacked_obj = boards[name][2]
                    stacked_obj.parent = board_obj

                    pcb_offset = Vector((0, 0, np.sign(offset.z) * PCB_THICKNESS))
                    if name == "FPNL":
                        pcb_offset.z += (self.fpnl_thickness - PCB_THICKNESS) * 0.5
                    stacked_obj.location = (offset + pcb_offset) * 0.001

        # select pcb objects and make one active

        if boards:
            bpy.ops.object.select_all(action="DESELECT")
            top_level_boards = [board for board in boards.values() if not board[2].parent]
            context.view_layer.objects.active = top_level_boards[0][2]
            for (_, _, obj) in top_level_boards:
                obj.select_set(True)

        # center pcbs

        if self.center_pcb:
            if boards:
                center = Vector((0, 0))
                for ((pos1, pos2), _, _) in top_level_boards:
                    center += (pos1 + pos2) * 0.5 * 0.001
                center /= len(top_level_boards)

                for (_, _, obj) in top_level_boards:
                    obj.location -= center.to_3d()
            else:
                center = pcb_object.dimensions * 0.5 * Vector((1, -1, 1))
                pcb_object.location = -(Vector(pcb_object.bound_box[3]) + center)

        # materials

        if self.merge_materials:
            merge_materials(component_map.values())

        if self.enhance_materials:
            enhance_materials(pcb_materials)

        return {"FINISHED"}

    @staticmethod
    def import_wrl(context, filepath, join=True):
        objects_before = set(bpy.data.objects)
        bpy.ops.import_scene.x3d(filepath=str(filepath), axis_forward="Y", axis_up="Z")
        wrl_objects = set(bpy.data.objects).difference(objects_before)

        if not join:
            return sorted(wrl_objects, key=lambda obj: obj.name)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in wrl_objects:
            if obj.type == "MESH":
                obj.select_set(True)

        joined_obj = wrl_objects.pop()
        joined_obj.name = joined_obj.data.name = filepath.name.split(".")[0]
        context.view_layer.objects.active = joined_obj

        bpy.ops.object.join()

        return joined_obj

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

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "import_components")
        layout.prop(self, "center_pcb")
        layout.prop(self, "enhance_pcb")

        layout.split()

        layout.prop(self, "merge_materials")
        layout.prop(self, "enhance_materials")

        layout.split()

        layout.prop(self, "cut_boards")
        layout.prop(self, "stack_boards")

        layout.split()

        if has_svg2blender():
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

def has_svg2blender():
    return addon_utils.check("svg2blender-importer") == (True, True)

def menu_func_import_pcb3d(self, context):
    self.layout.operator(PCB2BLENDER_OT_import_pcb3d.bl_idname, text="PCB (.pcb3d)")

operators = (PCB2BLENDER_OT_import_pcb3d,)

def register():
    for operator in operators:
        bpy.utils.register_class(operator)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_pcb3d)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_pcb3d)

    for operator in reversed(operators):
        bpy.utils.unregister_class(operator)
