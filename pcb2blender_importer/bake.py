import bpy, bmesh
from bpy.props import *
from mathutils import Vector, Matrix

import numpy as np

from .importer import PCB_THICKNESS, PCB_THICKNESS_MM
from .materials import LAYER_BOARD_EDGE, LAYER_THROUGH_HOLES

BAKE_LAYERS = { # is_data
    "Base Color": False,
    "Metallic":   True,
    "Roughness":  True,
    "Normal":     True,
    "Metallic":   True,
}
EXTA_PCB_BAKE_LAYERS = { # is_data
    "Clearcoat":           True,
    "Clearcoat Roughness": True,
}

BAKED_PREFIX = "BAKED_"

LAYER_PROXY_FACES = "bake_proxy_faces"

class PCB2BLENDER_OT_bake_pcb(bpy.types.Operator):
    """Bake PCB Material for exporting to .gltf or using in Eevee"""
    bl_idname = "pcb2blender.bake_pcb"
    bl_label = "Bake PCB Material"
    bl_options = {"UNDO"}

    dpi: FloatProperty(name="DPI", default=1016.0, min=0.0, soft_min=508.0, soft_max=2032.0)

    UV_MARGIN_MM = 1.0
    ISLAND_MARGIN_MM = 0.25

    @classmethod
    def poll(cls, context):
        material = context.material
        return bool(material) and material.p2b_type != "NONE" and context.mode == "OBJECT"

    def execute(self, context):
        material = context.material
        rebaking = False

        if material.p2b_type == "BAKED":
            if not material.p2b_other_material:
                return self.error(
                    f"baked material '{material.name}' is missing original material")
            
            rebaking = True
            material_baked = material
            material = material.p2b_other_material

        pcb_meshes = set()
        for mesh in bpy.data.meshes:
            if not mesh.p2b_is_pcb:
                continue
            if material.name in mesh.materials:
                pcb_meshes.add(mesh)
            elif rebaking and material_baked.name in mesh.materials:
                pcb_meshes.add(mesh)

        self.setup_bake_proxy(context, material, pcb_meshes)

        return {"FINISHED"}

    def setup_bake_proxy(self, context, material, meshes):
        layer_size_mm = np.array(material.p2b_bounds[2:4])
        uv_axis = layer_size_mm.argmax()
        uv_margin = self.UV_MARGIN_MM / layer_size_mm
        island_margin = self.ISLAND_MARGIN_MM / layer_size_mm
        seam_size = PCB_THICKNESS_MM / layer_size_mm[1 - uv_axis]

        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new()
        layer_board_edge    = bm.faces.layers.int.new(LAYER_BOARD_EDGE)
        layer_through_holes = bm.faces.layers.int.new(LAYER_THROUGH_HOLES)
        layer_proxy_faces   = bm.faces.layers.int.new(LAYER_PROXY_FACES)

        QUAD_INDICES = np.array(((0, 0), (1, 0), (1, 1), (0, 1)))
        XY_INDICES = np.tile((0, 1), 4).reshape(-1, 2)
        UV_INDICES = np.tile((2, 3), 4).reshape(-1, 2)

        bot_uv_offset = np.array((0, 1.0) if uv_axis == 0 else (1.0, 0))

        mesh_face_start_indices = []
        for mesh in meshes:
            mesh_face_start_indices.append(face_start_index := len(bm.faces))
            verts = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", verts)
            xs = verts.reshape((-1, 3))[:, 0]
            x_offset = xs.max() - xs.min() + 0.05
            bmesh.ops.translate(bm, verts=bm.verts, vec=Vector((x_offset, 0, 0)))
            bm.from_mesh(mesh)

            xyzuvs = np.array([
                (*loop.vert.co, *loop[uv_layer].uv)
                for face in bm.faces[face_start_index:] if not face[layer_board_edge]
                for loop in face.loops])

            top_xyzuvs = xyzuvs[np.nonzero(xyzuvs[:, 2] > 0)[0]]
            top_min = np.argmin(top_xyzuvs[:, :2], axis=0)
            top_xyuv_min = top_xyzuvs[[*top_min, *top_min], [0, 1, 3, 4]]
            top_max = np.argmax(top_xyzuvs[:, :2], axis=0)
            top_xyuv_max = top_xyzuvs[[*top_max, *top_max], [0, 1, 3, 4]]

            bot_xyzuvs = xyzuvs[np.nonzero(xyzuvs[:, 2] <= 0)[0]]
            bot_min = np.argmin(bot_xyzuvs[:, :2], axis=0)
            bot_xyuv_min = bot_xyzuvs[[*bot_min, *bot_min], [0, 1, 3, 4]]
            bot_max = np.argmax(bot_xyzuvs[:, :2], axis=0)
            bot_xyuv_max = bot_xyzuvs[[*bot_max, *bot_max], [0, 1, 3, 4]]

            top_face = np.array((top_xyuv_min, top_xyuv_max))
            bot_face = np.array((bot_xyuv_min, bot_xyuv_max))

            for xyuv_face, z in zip((top_face, bot_face), PCB_THICKNESS * np.array((1, -1))):
                verts = [bm.verts.new((*xy, z)) for xy in xyuv_face[QUAD_INDICES, XY_INDICES]]
                face = bm.faces.new(verts)
                face[layer_proxy_faces] = 1
                for loop, uv in zip(face.loops, xyuv_face[QUAD_INDICES, UV_INDICES]):
                    if xyuv_face is bot_face:
                        uv += bot_uv_offset
                    loop[uv_layer].uv = uv

        bm.faces.index_update()

        remove_faces = []
        for face in bm.faces:
            if face[layer_board_edge] or face[layer_through_holes] or face[layer_proxy_faces]:
                face.material_index = 0
            else:
                remove_faces.append(face)

        bmesh.ops.delete(bm, geom=remove_faces, context="FACES")

        bm.faces.index_update()

        uv_islands = []
        seam_indices = []

        traversed_faces = set()
        for face in bm.faces:
            if face[layer_proxy_faces] or face in traversed_faces:
                continue

            for i, edge in enumerate(face.edges):
                if not edge.is_boundary:
                    edge.seam = True
                    seam_indices.append((face.index, i))
                    break
            else:
                continue

            uv_island = []
            queued_faces = {face,}
            while queued_faces:
                next_face = queued_faces.pop()
                uv_island.append(next_face.index)
                traversed_faces.add(next_face)
                neighboring_faces = (edge.link_faces for edge in next_face.edges)
                queued_faces = queued_faces.union(*neighboring_faces) - traversed_faces

            uv_islands.append(np.array(uv_island))

        bm.select_mode = {"FACE"}
        for face in bm.faces:
            face.select = not face[layer_proxy_faces]
        bm.select_flush_mode()

        NAME = "PCB2BLENDER_BAKE_PROXY"
        mesh = bpy.data.meshes.new(NAME)
        bm.to_mesh(mesh)
        mesh.materials.append(material)
        obj = bpy.data.objects.new(NAME, mesh)
        context.scene.collection.objects.link(obj)

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj

        bake_uv_layer = mesh.uv_layers.new()
        bake_uv_layer.active = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.unwrap(margin=0)
        bpy.ops.object.mode_set(mode="OBJECT")

        bake_uv_layer = mesh.uv_layers.active

        uvs = np.empty(2 * len(bake_uv_layer.data))
        bake_uv_layer.data.foreach_get("uv", uvs)
        uvs = uvs.reshape(-1, 2)

        loop_starts = np.empty(len(mesh.polygons), dtype=int)
        mesh.polygons.foreach_get("loop_start", loop_starts)
        loop_totals = np.empty(len(mesh.polygons), dtype=int)
        mesh.polygons.foreach_get("loop_total", loop_totals)

        island_faces, seam_loop0s = np.array(seam_indices).T
        seam_loop1s = (seam_loop0s + 1) % loop_totals[island_faces]
        seam_loop_starts = loop_starts[island_faces]
        seam_loops = np.array((seam_loop0s, seam_loop1s)).T + seam_loop_starts.reshape(-1, 1)
        seam_uvs = uvs[seam_loops, :]
        seam_dirs = seam_uvs[:, 1, :] - seam_uvs[:, 0, :]
        seam_dirs = seam_dirs / np.sqrt(np.sum(seam_dirs ** 2, axis=1)).reshape(-1, 1)
        seam_axis = np.array((1.0, 0) if uv_axis == 0 else (0, 1.0))
        seam_dots = np.dot(seam_dirs, seam_axis)
        seam_angles = np.arccos(np.abs(seam_dots))

        rotate_seams = seam_angles < np.radians(45)

        island_loops = []
        for island in uv_islands:
            loops = np.concatenate([np.arange(loop_totals[face]) + loop_starts[face]
                for face in island])
            island_loops.append(loops)

        matrix_rotate_90 = np.array(Matrix.Rotation(np.radians(90), 2))
        for island_index in np.nonzero(rotate_seams)[0]:
            loops = island_loops[island_index]
            uvs[loops] = (matrix_rotate_90 @ uvs[loops].T).T

        islands_uv_offset = np.array((0, 2.0)) if uv_axis == 0 else np.array((2.0, 0))
        island_sizes = []
        for loops in island_loops:
            island_uvs = uvs[loops]
            island_min = island_uvs.min(axis=0)
            island_uvs -= island_min

            island_seam_size = island_uvs[:, 1 - uv_axis].max()
            fix_scale = seam_size / island_seam_size
            island_uvs *= fix_scale
            uvs[loops] = island_uvs + islands_uv_offset

            island_sizes.append(island_uvs[:, uv_axis].max())

        max_island_size = 1.0 - uv_margin[uv_axis] * 2
        for i, (island, size) in enumerate(zip(uv_islands, island_sizes)):
            if size <= max_island_size:
                continue

            per_face_loops = np.array([np.arange(loop_totals[face]) + loop_starts[face]
                for face in island], dtype=object)
            per_face_uvs = (uvs[face_loops.astype(int)] for face_loops in per_face_loops)
            keep_faces = np.array([np.all(face_uvs[:, uv_axis] <= max_island_size)
                for face_uvs in per_face_uvs])
            keep_face_indices = np.nonzero(keep_faces)[0]
            new_faces = np.nonzero(np.logical_not(keep_faces))[0]

            uv_islands[i] = island[keep_face_indices]
            uv_islands.append(island[new_faces])

            loops =     np.concatenate(per_face_loops[keep_face_indices]).astype(int)
            new_loops = np.concatenate(per_face_loops[new_faces]).astype(int)
            island_loops[i] = loops
            island_loops.append(new_loops)

            island_uvs =     uvs[loops]
            new_island_uvs = uvs[new_loops]

            new_island_uvs[:, uv_axis] -= np.min(new_island_uvs[:, uv_axis])
            uvs[new_loops] = new_island_uvs

            island_sizes[i] =   np.max(island_uvs[:, uv_axis])
            island_sizes.append(np.max(new_island_uvs[:, uv_axis]))

        sort_islands = np.argsort(island_sizes)[::-1]
        line_sizes = np.array((uv_margin[uv_axis],))
        line_offset = seam_size + island_margin[1 - uv_axis]
        for island_index in sort_islands:
            size = island_sizes[island_index]
            available_lines = np.nonzero((line_sizes + size + uv_margin[uv_axis]) <= 1.0)[0]
            if len(available_lines) > 0:
                line_index = available_lines[0]
            else:
                line_index = len(line_sizes)
                line_sizes = np.append(line_sizes, (uv_margin[uv_axis],))

            offset = np.array((0.0, 0.0))
            offset[uv_axis] = line_sizes[line_index]
            offset[1 - uv_axis] = uv_margin[1 - uv_axis] + line_index * line_offset
            uvs[island_loops[island_index]] += offset

            line_sizes[line_index] += size + island_margin[uv_axis]

        lines_size = \
            uv_margin[1 - uv_axis] * 2 + line_offset * (len(line_sizes) - 1) + seam_size
        uvs[:, 1 - uv_axis] /= 2.0 + lines_size

        bake_uv_layer.data.foreach_set("uv", uvs.flatten())

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.select_split()
        bpy.ops.object.mode_set(mode="OBJECT")

        return obj

    def error(self, msg):
        print(f"error: {msg}")
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    def warning(self, msg):
        print(f"warning: {msg}")
        self.report({"WARNING"}, msg)

class PCB2BLENDER_OT_bake_material_switch(bpy.types.Operator):
    """Switch between normal and baked PCB material"""
    bl_idname = "pcb2blender.bake_material_switch"
    bl_label = "Toggle baked PCB material"
    bl_options = {"UNDO"}

    toggle: BoolProperty(name="Toggle", default=True)
    mode:   EnumProperty(name="Mode", items=(("NORMAL", "Normal", ""), ("BAKED", "Baked", "")))

    @classmethod
    def poll(cls, context):
        return bool(context.object)

    def execute(self, context):
        pass

class PCB2BLENDER_PT_material(bpy.types.Panel):
    bl_label = "PCB Material"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material and context.material.p2b_type != "NONE"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        material = context.material

        is_baked_material = material.p2b_type == "BAKED"
        has_baked_material = material.p2b_other_material is not None

        layout.prop(scene, "p2b_dpi")

        props = layout.operator(PCB2BLENDER_OT_bake_pcb.bl_idname, icon="RENDER_STILL",
            text=f"{'Rebake' if has_baked_material else 'Bake'} PCB Material")
        props.dpi = scene.p2b_dpi

        row = layout.row()
        row.operator(PCB2BLENDER_OT_bake_material_switch.bl_idname, icon="FILE_REFRESH",
            text=f"Switch to {'Normal' if is_baked_material else 'Baked'} Material")
        row.enabled = has_baked_material

classes = (
    PCB2BLENDER_OT_bake_pcb,
    PCB2BLENDER_OT_bake_material_switch,
    PCB2BLENDER_PT_material,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.p2b_dpi = FloatProperty(name="DPI", options={"HIDDEN"},
        default=1016.0, min=0.0, soft_min=508.0, soft_max=2032.0)
    bpy.types.Mesh.p2b_is_pcb = BoolProperty(options={"HIDDEN"})
    bpy.types.Material.p2b_bounds = FloatVectorProperty(options={"HIDDEN"}, size=4)
    bpy.types.Material.p2b_type = EnumProperty(options={"HIDDEN"},
        items=(("NONE", "None", ""), ("NORMAL", "Normal", ""), ("BAKED", "Baked", "")))
    bpy.types.Material.p2b_other_material = PointerProperty(options={"HIDDEN"},
        type=bpy.types.Material)

def unregister():
    del bpy.types.Scene.p2b_dpi
    del bpy.types.Mesh.p2b_is_pcb
    del bpy.types.Material.p2b_bounds
    del bpy.types.Material.p2b_type
    del bpy.types.Material.p2b_other_material

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
