import bpy
from bpy.props import *

import numpy as np

from .importer import MM_TO_M
from .custom_node_utils import CustomNodetreeNodeBase

class PCB2BLENDER_OT_solder_joint_add(bpy.types.Operator):
    """Add a solder joint"""
    bl_idname = "pcb2blender.solder_joint_add"
    bl_label = "Solder Joint"
    bl_options = {"REGISTER", "UNDO"}

    pad_type: EnumProperty(name="Pad Type",
        items=(("THT", "THT", ""), ("SMD", "SMD", "")),
        description="Type of pad")
    pad_shape: EnumProperty(name="Pad Shape",
        items=(("SQUARE", "Square", ""), ("RECTANGULAR", "Rectangular", "")),
        description="Shape of the pad")
    pad_size: FloatVectorProperty(name="Pad Size", subtype="XYZ", size=2, default=(1.7, 1.7),
        description="Size of the pad in mm")
    roundness: FloatProperty(name="Roundness", min=0.0, max=1.0,
        description="Roundness of the corners of the pad")

    hole_shape: EnumProperty(name="Hole Shape",
        items=(("CIRCULAR", "Circular", ""), ("OVAL", "Oval", "")),
        description="Shape of the through hole")
    hole_size: FloatVectorProperty(name="Hole Size", subtype="XYZ", size=2, default=(1.0, 1.0),
        description="Size of the through hole in mm")

    pcb_thickness: FloatProperty(name="PCB Thickness",
        unit="LENGTH", min=0.0, soft_max=3.0, default=1.6,
        description="Thickness of the PCB in mm")

    location: FloatVectorProperty(name="Location", subtype="XYZ",
        description="Location for the newly added object")
    rotation: FloatVectorProperty(name="Rotation", subtype="EULER",
        description="Rotation for the newly added object")

    reuse_material: BoolProperty(name="Reuse Material", default=False)

    def execute(self, context):
        name = "Solder Joint"
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)

        if self.pad_shape == "SQUARE":
            self.pad_size[1] = self.pad_size[0]
        pad_size = np.array(self.pad_size) * 1.04
        if self.hole_shape == "CIRCULAR":
            self.hole_size[1] = self.hole_size[0]
        hole_size = np.array(self.hole_size)

        if self.pad_type == "THT":
            verts, faces = solder_joint_tht(
                pad_size, hole_size, self.roundness, self.pcb_thickness)
        elif self.pad_type == "SMD":
            verts, faces = solder_joint_smd(pad_size, self.roundness, self.pcb_thickness)

        verts *= MM_TO_M
        indices = faces.flatten()

        mesh.vertices.add(len(verts))
        mesh.vertices.foreach_set("co", verts.flatten())

        mesh.loops.add(len(indices))
        mesh.loops.foreach_set("vertex_index", indices)

        mesh.polygons.add(len(faces))
        mesh.polygons.foreach_set("loop_start", np.arange(0, len(indices), 4))
        mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4))
        mesh.polygons.foreach_set("use_smooth", np.full(len(faces), True))

        mesh.update()
        mesh.validate()

        if not (self.reuse_material and (material := bpy.data.materials.get(name))):
            material = bpy.data.materials.new(name)
            material.use_nodes = True
            material.node_tree.nodes.clear()
            nodes_def = {
                "shader": ("ShaderNodeBsdfSolder", {}, {}),
                "output": ("ShaderNodeOutputMaterial", {"location": (240, 0)},
                    {"Surface": ("shader", 0)}),
            }
            CustomNodetreeNodeBase.setup_nodes(material.node_tree, nodes_def, False)
        mesh.materials.append(material)

        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.normals_make_consistent()
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.pad_type == "SMD":
            smooth = obj.modifiers.new("Simple", "SUBSURF")
            smooth.subdivision_type = "SIMPLE"
            smooth.render_levels = 1
            smooth = obj.modifiers.new("Smooth", "SUBSURF")
            smooth.render_levels = 1
        else:
            smooth = obj.modifiers.new("Smooth", "SUBSURF")
            smooth.levels = 2

        TEXTURE_NAME = "PCB2BLENDER_SOLDER_NOISE"
        if not (texture := bpy.data.textures.get(TEXTURE_NAME)):
            texture = bpy.data.textures.new(TEXTURE_NAME, "CLOUDS")
            texture.noise_scale = 2e-4
            texture.noise_depth = 1
        displace = obj.modifiers.new("Noise", "DISPLACE")
        displace.texture = texture
        displace.texture_coords = "GLOBAL"
        displace.strength = 5e-5 if self.pad_type == "SMD" else 1e-4

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        layout.prop(self, "pad_type")
        layout.separator()

        layout.prop(self, "pad_shape")
        layout.prop(self, "pad_size", index=0 if self.pad_shape == "SQUARE" else -1)
        layout.prop(self, "roundness", slider=True)
        layout.separator()

        if self.pad_type == "THT":
            layout.prop(self, "hole_shape")
            layout.prop(self, "hole_size", index=0 if self.hole_shape == "CIRCULAR" else -1)
            layout.separator()

        layout.prop(self, "pcb_thickness")
        layout.separator()

        layout.prop(self, "location")
        layout.prop(self, "rotation")

def solder_joint_tht(pad_size, hole_size, roundness=0.0, pcb_thickness=1.6):
    vs = np.empty((0, 3), dtype=float)
    fs = np.empty((0, 4), dtype=int)

    avg_size = (pad_size + hole_size) * 0.5

    vs, fs = add_octagon_layer(vs, fs, hole_size * 1.4, -(pcb_thickness + 0.1), 1.0, True)
    vs, fs = add_octagon_layer(vs, fs, hole_size * 0.9, -0.30, 1.0)
    vs, fs = add_octagon_layer(vs, fs, pad_size,        -0.10, max(roundness, 0.2))
    vs, fs = add_octagon_layer(vs, fs, pad_size,         0.05, max(roundness, 0.2))
    vs, fs = add_octagon_layer(vs, fs, avg_size  * 0.9,  0.25, 0.8)
    vs, fs = add_octagon_layer(vs, fs, hole_size * 0.9,  0.55, 0.7)
    vs, fs = add_octagon_layer(vs, fs, hole_size * 0.8,  0.85, 0.2)
    vs, fs = add_octagon_layer(vs, fs, hole_size * 0.5,  1.15, 0.6)
    vs, fs = add_octagon_layer(vs, fs, hole_size * 0.2,  1.20, 1.0, True)
    vs += np.array((0, 0, pcb_thickness * 0.5))

    return vs, fs

def solder_joint_smd(pad_size, roundness=0.0, pcb_thickness=1.6):
    vs = np.empty((0, 3), dtype=float)
    fs = np.empty((0, 4), dtype=int)

    vs, fs = add_octagon_layer(vs, fs, pad_size, -0.06, max(roundness, 0.2), True)
    vs, fs = add_octagon_layer(vs, fs, pad_size,  0.04, max(roundness, 0.2), True)
    vs += np.array((0, 0, pcb_thickness * 0.5))

    return vs, fs

MAX_ROUNDNESS_FAC = 1 - 1 / np.tan(np.deg2rad(135.0 / 2))

def add_octagon_layer(verts, faces, size, z, roundness=0.5, fill=False):
    half_size = size * 0.5
    min_half_size = half_size.min()
    r_offset = min_half_size * roundness * MAX_ROUNDNESS_FAC

    new_verts = np.array((
        ( (half_size[0] - r_offset),  (half_size[1]           ), z),
        ( (half_size[0]           ),  (half_size[1] - r_offset), z),
        ( (half_size[0]           ), -(half_size[1] - r_offset), z),
        ( (half_size[0] - r_offset), -(half_size[1]           ), z),
        (-(half_size[0] - r_offset), -(half_size[1]           ), z),
        (-(half_size[0]           ), -(half_size[1] - r_offset), z),
        (-(half_size[0]           ),  (half_size[1] - r_offset), z),
        (-(half_size[0] - r_offset),  (half_size[1]           ), z),
    ))

    if len(verts) > 0:
        new_faces = np.array((
            (-8, -7, 1, 0),
            (-7, -6, 2, 1),
            (-6, -5, 3, 2),
            (-5, -4, 4, 3),
            (-4, -3, 5, 4),
            (-3, -2, 6, 5),
            (-2, -1, 7, 6),
            (-1, -8, 0, 7),
        ))
    else:
        new_faces = np.empty((0, 4), dtype=int)

    if fill:
        center_vertex = (0, 0, z)
        new_verts = np.append((center_vertex,), new_verts, axis=0)
        fill_faces = np.array((
            (1, 2, 3, 0),
            (3, 4, 5, 0),
            (5, 6, 7, 0),
            (7, 8, 1, 0),
        ))
        new_faces = new_faces + (new_faces >= 0)
        new_faces = np.append(new_faces, fill_faces, axis=0)

    faces = np.append(faces, new_faces + len(verts), axis=0)
    verts = np.append(verts, new_verts, axis=0)

    return verts, faces

classes = (
    PCB2BLENDER_OT_solder_joint_add,
)

def menu_func_solder_joint_add(self, context):
    self.layout.separator()
    self.layout.operator(PCB2BLENDER_OT_solder_joint_add.bl_idname, icon="PROP_CON")

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.VIEW3D_MT_mesh_add.append(menu_func_solder_joint_add)

def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func_solder_joint_add)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
