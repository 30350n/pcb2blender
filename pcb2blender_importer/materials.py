import bpy
from mathutils import Vector

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
        
        node_shader = material.node_tree.nodes["Principled BSDF"]

        # silver pins
        if material.name.startswith("PIN-01"):
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.25
        # silver metal
        elif material.name.startswith("MET-SILVER"):
            node_shader.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.2
        # gold pins
        elif material.name.startswith("PIN-02"):
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.3
        # shiny black plastic
        elif material.name.startswith("IC-BODY-EPOXY-04"):
            node_shader.inputs["Base Color"].default_value = (0, 0, 0, 1.0)
            node_shader.inputs["Roughness"].default_value = 0.3
            node_shader.inputs["Specular"].default_value = 0.5
        # matte black plastic
        elif material.name.startswith("RES-SMD-01"):
            node_shader.inputs["Base Color"].default_value = (0, 0, 0, 1.0)
            node_shader.inputs["Roughness"].default_value = 0.9
            node_shader.inputs["Specular"].default_value = 0.25
        # black metal
        elif material.name.startswith("MET-01"):
            node_shader.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1.0)
            node_shader.inputs["Metallic"].default_value = 1.0
        # bronze
        elif material.name.startswith("MET-BRONZE"):
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.35
        # glass white
        elif material.name.startswith("GLASS-19"):
            node_shader.inputs["IOR"].default_value = 1.46
            node_shader.inputs["Transmission"].default_value = 0.5
            node_shader.inputs["Transmission Roughness"].default_value = 0.2
            node_shader.inputs["Alpha"].default_value = 0.5
        # copper
        elif material.name in {PCB2_LAYER_NAMES[1], PCB2_LAYER_NAMES[4]}:
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.35
        # solder paste
        elif material.name in {PCB2_LAYER_NAMES[2], PCB2_LAYER_NAMES[5]}:
            node_shader.inputs["Metallic"].default_value = 1.0
            node_shader.inputs["Roughness"].default_value = 0.6
        # solder mask
        elif material.name in {PCB2_LAYER_NAMES[3], PCB2_LAYER_NAMES[6]}:
            node_shader.inputs["Transmission"].default_value = 0.75

        # brown pcb
        elif is_same_color((0.06, 0.02, 0), node_shader.inputs["Base Color"].default_value):
            node_shader.inputs["Subsurface"].default_value = 0.75
            node_shader.inputs["Subsurface Color"].default_value = (0.011, 0.006, 0.004, 1.0)
            node_shader.inputs["Roughness"].default_value = 0.4

def is_same_color(a, b):
    return (Vector(a[:3]) - Vector(b[:3])).magnitude < 0.01
