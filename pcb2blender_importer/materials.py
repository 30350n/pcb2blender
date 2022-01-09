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
        if mat4cad_mat := MATERIALS.get(material.name):
            mat4cad_mat.setup_node_tree(material.node_tree)

def is_same_color(a, b):
    return (Vector(a[:3]) - Vector(b[:3])).magnitude < 0.01

PCB2_LAYER_NAMES = (
    "0 Board",
    "1 Front Copper",
    "2 Front Paste",
    "3 Front Solder Mask",
    "4 Back Copper",
    "5 Back Paste",
    "6 Back Solder Mask",
    "7 Through Holes",
    "8 Front Silkscreen",
    "9 Back Silkscreen",
)
