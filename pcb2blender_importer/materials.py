from .mat4cad import *
from .mat4cad.blender import *

import bpy
from mathutils import Vector

def merge_materials(meshes):
    for mesh in meshes:
        for i, material in enumerate(mesh.materials):
            if (material.name[-4] == "."
                    and material.name[-3:].isdecimal()
                    and material.name[:-4] in bpy.data.materials):
                mesh.materials[i] = bpy.data.materials[material.name[:-4]]

def enhance_materials(materials):
    for material in materials:
        if not material.use_nodes:
            continue

        if mat4cad_mat := Material.from_name(material.name):
            pass
        elif material.name == "PIN-01":
            mat4cad_mat = Material.from_name("special-pins_silver-default")
        elif material.name == "PIN-02":
            mat4cad_mat = Material.from_name("special-pins_gold-default")
        elif material.name == "IC-BODY-EPOXY-04":
            mat4cad_mat = Material.from_name("plastic-traffic_black-matte")
        elif material.name == "IC-LABEL-01":
            mat4cad_mat = Material.from_name("plastic-grey_white-semi_matte")
        else:
            continue

        mat4cad_mat.setup_node_tree(material.node_tree)
