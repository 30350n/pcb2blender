from .mat4cad.core import Material, hex2rgb, srgb2lin
from .mat4cad.blender import ShaderNodeBsdfMat4cad
from .custom_node_utils import *

import bpy
from bpy.props import EnumProperty

from nodeitems_utils import NodeItem
from nodeitems_builtins import ShaderNodeCategory

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

def setup_pcb_material(node_tree: bpy.types.NodeTree, images: dict[str, bpy.types.Image]):
    node_tree.nodes.clear()

    nodes = {
        "cu":    ("ShaderNodeTexImage", {"location": (-500, -240), "hide": True,
            "interpolation": "Cubic", "image": images["Cu"]}, {}),
        "mask":  ("ShaderNodeTexImage", {"location": (-500, -280), "hide": True,
            "interpolation": "Cubic", "image": images["Mask"]}, {}),
        "silks": ("ShaderNodeTexImage", {"location": (-500, -320), "hide": True,
            "interpolation": "Cubic", "image": images["SilkS"]}, {}),
        "paste": ("ShaderNodeTexImage", {"location": (-500, -360), "hide": True,
            "interpolation": "Cubic", "image": images["Paste"]}, {}),

        "seperate_cu":    ("ShaderNodeSeparateRGB", {"location": (-200, -240), "hide": True}, 
            {"Image": ("cu",    "Color")}),
        "seperate_mask":  ("ShaderNodeSeparateRGB", {"location": (-200, -280), "hide": True}, 
            {"Image": ("mask",  "Color")}),
        "seperate_silks": ("ShaderNodeSeparateRGB", {"location": (-200, -320), "hide": True}, 
            {"Image": ("silks", "Color")}),
        "seperate_paste": ("ShaderNodeSeparateRGB", {"location": (-200, -360), "hide": True}, 
            {"Image": ("paste", "Color")}),

        "base_material": ("ShaderNodeBsdfMat4cad", {"location": (-200, 0),
            "mat_base": "PCB", "mat_color": "PCB_YELLOW"}, {}),
        "exposed_copper": ("ShaderNodeBsdfMat4cad", {"location": (-400, 0),
            "mat_base": "METAL", "mat_color": "COPPER", "mat_variant": "GLOSSY"}, {}),
        "solder_mask": ("ShaderNodeSolderMaskShader", {"location": (-660, 0)},
            {"F_Cu": ("seperate_cu", "R"), "B_Cu": ("seperate_cu", "G")}),
        "silkscreen": ("ShaderNodeBsdfMat4cad", {"location": (-860, 0),
            "mat_base": "PLASTIC", "mat_color": "PURE_WHITE", "mat_variant": "GLOSSY"}, {}),
        "solder": ("ShaderNodeBsdfMat4cad", {"location": (-1120, 0),
            "mat_base": "METAL", "mat_color": "SILVER", "mat_variant": "GLOSSY"}, {}),

        "shader": ("ShaderNodePcbShader", {}, {
            "F_Cu":    ("seperate_cu",    "R"), "B_Cu":    ("seperate_cu",    "G"),
            "F_Mask":  ("seperate_mask",  "R"), "B_Mask":  ("seperate_mask",  "G"),
            "F_SilkS": ("seperate_silks", "R"), "B_SilkS": ("seperate_silks", "G"),
            "F_Paste": ("seperate_paste", "R"), "B_Paste": ("seperate_paste", "G"),
            "Base Material":  ("base_material",  "BSDF"),
            "Exposed Copper": ("exposed_copper", "BSDF"),
            "Solder Mask":    ("solder_mask",    "BSDF"),
            "Silkscreen":     ("silkscreen",     "BSDF"),
            "Solder":         ("solder",         "BSDF"),
        }),
        "output": ("ShaderNodeOutputMaterial", {"location": (200, 0)},
            {"Surface": ("shader", "BSDF"), "Displacement": ("shader","Displacement")}),
    }

    setup_node_tree(node_tree, nodes)

class ShaderNodeSolderMaskShader(CustomNodetreeNodeBase, bpy.types.ShaderNodeCustomGroup):
    bl_label = "Solder Mask BSDF"
    bl_width_default = 200

    def update_props(self, context):
        roughness = 0.5
        match self.soldermask:
            case "GREEN":
                light_color = hex2rgb("28a125")
                dark_color  = hex2rgb("155211")
            case "RED":
                light_color = hex2rgb("e50007")
                dark_color  = hex2rgb("731114")
            case "YELLOW":
                light_color = hex2rgb("dac92b")
                dark_color  = hex2rgb("687c19")
            case "BLUE":
                light_color = hex2rgb("116cc2")
                dark_color  = hex2rgb("053059")
            case "WHITE":
                light_color = hex2rgb("d3cfc9")
                dark_color  = hex2rgb("e1dddc")
                roughness = 0.3
            case "BLACK":
                light_color = hex2rgb("191918")
                dark_color  = hex2rgb("000000")
                roughness = 0.9

        if not self.soldermask == "CUSTOM":
            self.inputs["Light Color"].default_value = (*srgb2lin(light_color), 1.0)
            self.inputs["Dark Color"].default_value  = (*srgb2lin(dark_color),  1.0)
            self.inputs["Roughness"].default_value   = roughness

        hidden = self.soldermask != "CUSTOM"
        for input_name in ("Light Color", "Dark Color", "Roughness"):
            self.inputs[input_name].hide = hidden

    soldermask: EnumProperty(name="Solder Mask", update=update_props, items=(
        ("GREEN",  "Green",  ""),
        ("RED",    "Red",    ""),
        ("YELLOW", "Yellow", ""),
        ("BLUE",   "Blue",   ""),
        ("WHITE",  "White",  ""),
        ("BLACK",  "Black",  ""),
        ("CUSTOM", "Custom", ""),
    ))

    def init(self, context):
        inputs = {
            "Light Color": ("NodeSocketColor",  {}),
            "Dark Color":  ("NodeSocketColor",  {}),
            "Roughness":   ("NodeSocketFloat",  {}),
            "Normal":      ("NodeSocketVector", {"hide_value": True}),
            "F_Cu": ("NodeSocketFloat", {"hide_value": True}),
            "B_Cu": ("NodeSocketFloat", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "separate_position": ("ShaderNodeSeparateXYZ", {},
                {"Vector" : ("tex_coord", "Object")}),
            "is_bottom_layer": ("ShaderNodeMath", {"operation": "LESS_THAN"},
                {0: ("separate_position", "Z"), 1: 0.0}),

            "cu": ("ShaderNodeMixRGB", {}, {"Fac": ("is_bottom_layer", 0),
                "Color1": ("inputs", "F_Cu"), "Color2": ("inputs", "B_Cu")}),
            "cu_invert": ("ShaderNodeInvert", {}, {"Color": ("cu", 0)}),

            "mix_color": ("ShaderNodeMixRGB", {}, {"Fac": ("cu", 0),
                1: ("inputs", "Dark Color"), 2: ("inputs", "Light Color")}),
            "subsurface": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("cu_invert", 0), 1: 0.001}),
            "ssr": ("ShaderNodeBrightContrast", {},
                {"Color": ("mix_color", 0), "Bright": 0.25}),

            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("mix_color", 0), "Subsurface Color": ("mix_color", 0),
                "Subsurface": ("subsurface", 0), "Subsurface Radius": ("ssr", 0),
                "Roughness": ("inputs", "Roughness"), "Normal": ("inputs", "Normal")}),
        }

        outputs = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

    def draw_buttons(self, context, layout):
        layout.prop(self, "soldermask", text="")

class ShaderNodePcbShader(CustomNodetreeNodeBase, bpy.types.ShaderNodeCustomGroup):
    bl_label = "PCB Shader"
    bl_width_default = 140

    def init(self, context):
        inputs = {
            "Base Material":  ("NodeSocketShader", {}),
            "Exposed Copper": ("NodeSocketShader", {}),
            "Solder Mask":    ("NodeSocketShader", {}),
            "Silkscreen":     ("NodeSocketShader", {}),
            "Solder":         ("NodeSocketShader", {}),
            "F_Cu":    ("NodeSocketFloat", {"hide_value": True}),
            "B_Cu":    ("NodeSocketFloat", {"hide_value": True}),
            "F_Mask":  ("NodeSocketFloat", {"hide_value": True}),
            "B_Mask":  ("NodeSocketFloat", {"hide_value": True}),
            "F_SilkS": ("NodeSocketFloat", {"hide_value": True}),
            "B_SilkS": ("NodeSocketFloat", {"hide_value": True}),
            "F_Paste": ("NodeSocketFloat", {"hide_value": True}),
            "B_Paste": ("NodeSocketFloat", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "separate_position": ("ShaderNodeSeparateXYZ", {},
                {"Vector" : ("tex_coord", "Object")}),
            "is_bottom_layer": ("ShaderNodeMath", {"operation": "LESS_THAN"},
                {0: ("separate_position", "Z"), 1: 0.0}),

            "cu": ("ShaderNodeMixRGB", {}, {"Fac": ("is_bottom_layer", 0),
                "Color1": ("inputs", "F_Cu"), "Color2": ("inputs", "B_Cu")}),
            "mask": ("ShaderNodeMixRGB", {}, {"Fac": ("is_bottom_layer", 0),
                "Color1": ("inputs", "F_Mask"), "Color2": ("inputs", "B_Mask")}),
            "silks": ("ShaderNodeMixRGB", {}, {"Fac": ("is_bottom_layer", 0),
                "Color1": ("inputs", "F_SilkS"), "Color2": ("inputs", "B_SilkS")}),
            "paste": ("ShaderNodeMixRGB", {}, {"Fac": ("is_bottom_layer", 0),
                "Color1": ("inputs", "F_Paste"), "Color2": ("inputs", "B_Paste")}),

            "mix_cu": ("ShaderNodeMixShader", {}, {"Fac": ("cu", 0),
                1: ("inputs", "Base Material"), 2: ("inputs", "Exposed Copper")}),
            "mix_mask": ("ShaderNodeMixShader", {}, {"Fac": ("mask", 0),
                1: ("mix_cu", 0), 2: ("inputs", "Solder Mask")}),
            "mix_silks": ("ShaderNodeMixShader", {}, {"Fac": ("silks", 0),
                1: ("mix_mask", 0), 2: ("inputs", "Silkscreen")}),
            "mix_solder": ("ShaderNodeMixShader", {}, {"Fac": ("paste", 0),
                1: ("mix_silks", 0), 2: ("inputs", "Solder")}),

            "multiply_mask": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("mask", 0), 1: 0.02}),
            "multiply_cu": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("cu", 0), 1: 0.03, 2: ("multiply_mask", 0)}),
            "multiply_silks": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("silks", 0), 1: 0.0495}),
            "max_silks": ("ShaderNodeMath", {"operation": "MAXIMUM"},
                {0: ("multiply_cu", 0), 1: ("multiply_silks", 0)}),
            "multiply_scale": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("max_silks", 0), 1: 0.001}),
            "displacement": ("ShaderNodeDisplacement", {},
                {"Height": ("multiply_scale", 0)}),
        }

        outputs = {
            "BSDF": ("NodeSocketShader", {}, ("mix_solder", 0)),
            "Displacement": ("NodeSocketVector", {}, ("displacement", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

shader_node_category = ShaderNodeCategory("SH_NEW_PCB2BLENDER", "Pcb2Blender", items=(
    NodeItem("ShaderNodeBsdfMat4cad"),
    NodeItem("ShaderNodeSolderMaskShader"),
    NodeItem("ShaderNodePcbShader"),
))

classes = (
    ShaderNodeBsdfMat4cad,
    ShaderNodeSolderMaskShader,
    ShaderNodePcbShader,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    register_node_category("SHADER", shader_node_category)

def unregister():
    unregister_node_category("SHADER", shader_node_category)

    for cls in classes:
        bpy.utils.unregister_class(cls)
