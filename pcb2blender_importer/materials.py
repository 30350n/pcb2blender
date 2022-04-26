from .mat4cad.core import Material, hex2rgb, srgb2lin
from .mat4cad.colors import PCB_YELLOW
from .mat4cad.blender import ShaderNodeBsdfMat4cad
from .custom_node_utils import *

import bpy
from bpy.props import EnumProperty
from bpy.types import ShaderNodeCustomGroup
from mathutils import Vector

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
        "cu":    ("ShaderNodeTexImage", {"location": (-500, -320), "hide": True,
            "interpolation": "Cubic", "image": images["Cu"]}, {}),
        "mask":  ("ShaderNodeTexImage", {"location": (-500, -360), "hide": True,
            "interpolation": "Cubic", "image": images["Mask"]}, {}),
        "silks": ("ShaderNodeTexImage", {"location": (-500, -400), "hide": True,
            "interpolation": "Cubic", "image": images["SilkS"]}, {}),
        "paste": ("ShaderNodeTexImage", {"location": (-500, -440), "hide": True,
            "interpolation": "Cubic", "image": images["Paste"]}, {}),

        "seperate_cu":    ("ShaderNodeSeparateRGB", {"location": (-200, -320), "hide": True}, 
            {"Image": ("cu",    "Color")}),
        "seperate_mask":  ("ShaderNodeSeparateRGB", {"location": (-200, -360), "hide": True}, 
            {"Image": ("mask",  "Color")}),
        "seperate_silks": ("ShaderNodeSeparateRGB", {"location": (-200, -400), "hide": True}, 
            {"Image": ("silks", "Color")}),
        "seperate_paste": ("ShaderNodeSeparateRGB", {"location": (-200, -440), "hide": True}, 
            {"Image": ("paste", "Color")}),

        "base_material": ("ShaderNodeBsdfMat4cad", {"location": (-260, 0),
            "mat_base": "PCB", "mat_color": "PCB_YELLOW"}, {}),
        "exposed_copper": ("ShaderNodeBsdfPcbSurfaceFinish", {"location": (-500, 0)}, {}),
        "solder_mask": ("ShaderNodeBsdfPcbSolderMask", {"location": (-740, 0)},
            {"F_Cu": ("seperate_cu", "R"), "B_Cu": ("seperate_cu", "G")}),
        "board_edge": ("ShaderNodeBsdfPcbBoardEdge", {"location": (-740, -280)},
            {"Mask Color": ("solder_mask", "Color")}),
        "silkscreen": ("ShaderNodeBsdfPcbSilkscreen", {"location": (-980, 0)}, {}),
        "solder": ("ShaderNodeBsdfSolder", {"location": (-1220, 0)}, {}),

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
            "Board Edge":     ("board_edge",     "BSDF"),
        }),
        "output": ("ShaderNodeOutputMaterial", {"location": (240, 0)},
            {"Surface": ("shader", "Shader"), "Displacement": ("shader","Displacement")}),
    }

    setup_node_tree(node_tree, nodes)

class ShaderNodeBsdfPcbSurfaceFinish(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "Surface Finish BSDF"
    bl_width_default = 180

    def update_props(self, context):
        match self.surface_finish:
            case "HASL":
                color = hex2rgb("eaeae5")
                roughness = 0.15
                texture_strength = 1.0
            case "ENIG":
                color = hex2rgb("efdfbb")
                roughness = 0.1
                texture_strength = 0.3
            case "NONE":
                color = hex2rgb("e1bbac")
                roughness = 0.1
                texture_strength = 0.25

        if not self.surface_finish == "CUSTOM":
            self.inputs["Color"].default_value = (*srgb2lin(color), 1.0)
            self.inputs["Roughness"].default_value = roughness
            self.inputs["Texture Strength"].default_value = texture_strength

        hidden = self.surface_finish != "CUSTOM"
        for input_name in ("Color", "Roughness", "Texture Strength"):
            self.inputs[input_name].hide = hidden

    surface_finish: EnumProperty(name="Surface Finish", update=update_props, items=(
        ("HASL",   "HASL",   ""),
        ("ENIG",   "ENIG",   ""),
        ("NONE",   "None",   ""),
        ("CUSTOM", "Custom", ""),
    ))

    def init(self, context):
        inputs = {
            "Color": ("NodeSocketColor", {}),
            "Roughness": ("NodeSocketFloat",  {}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": ("ShaderNodeTexNoise", {}, {"Vector": ("tex_coord", "Object"),
                "Scale": 500.0, "Detail": 4.0, "Roughness": 0.4, "Distortion": 0.1}),
            "noise_bipolar": ("ShaderNodeMapRange", {},
                {"Value": ("noise", "Fac"), "To Min": -0.5, "To Max": 0.5}),
            "noise_scaled": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("noise_bipolar", 0), 1: ("inputs", "Texture Strength")}),
            "bump": ("ShaderNodeBump", {}, {"Strength": 0.1, "Distance": 1e-3,
                "Height": ("noise_scaled", 0), "Normal": ("inputs", "Normal")}),
            "roughness": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("noise_scaled", 0), 1: 0.2, 2: ("inputs", "Roughness")}),
            
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", "Normal")}),
            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("inputs", "Color"), "Metallic": 1.0,
                "Roughness": ("roughness", 0), "Normal": ("bevel", "Normal")}),
        }

        outputs = {
            "BSDF":  ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

class ShaderNodeBsdfPcbSolderMask(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "Solder Mask BSDF"
    bl_width_default = 180

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
                light_color = hex2rgb("10100f")
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
            "Texture Strength": ("NodeSocketFloat", {"default_value": 0.5}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
            "F_Cu":   ("NodeSocketFloat",  {"hide_value": True}),
            "B_Cu":   ("NodeSocketFloat",  {"hide_value": True}),
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
                {"Color": ("mix_color", 0), "Bright": 0.1}),

            "strength_noise": ("ShaderNodeTexNoise", {},
                {"Vector": ("tex_coord", "Object"), "Scale": 100.0, "Detail": 0.0}),
            "strength_scaled": ("ShaderNodeMapRange", {},
                {"Value": ("strength_noise", "Fac"), "To Min": -0.012, "To Max": 0.044}),
            "strength": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("strength_scaled", 0), 1: ("inputs", "Texture Strength")}),
            "scale": ("ShaderNodeVectorMath", {"operation": "MULTIPLY"},
                {0: ("tex_coord", "Object"), 1: Vector((3000.0, 3000.0, 3000.0))}),
            "checkers": ("ShaderNodeTexVoronoi", {"distance": "MINKOWSKI"},
                {"Vector": ("scale", 0), "Scale": 1.0, "Exponent": 1.8, "Randomness": 0.1}),
            "scale2": ("ShaderNodeVectorMath", {"operation": "MULTIPLY"},
                {0: ("scale", 0), 1: Vector((1.0, 0.0, 0.25))}),
            "lines": ("ShaderNodeTexVoronoi", {"distance": "MINKOWSKI"},
                {"Vector": ("scale2", 0), "Scale": 1.0, "Exponent": 1.8, "Randomness": 0.1}),
            "combined": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("checkers", "Distance"), 1: 0.25, 2: ("lines", "Distance")}),
            "height_scale": ("ShaderNodeMapRange", {}, {"Value": ("cu", 0), "To Min": 0.25}),
            "height": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("combined", 0), 1: ("height_scale", 0)}),
            "bump": ("ShaderNodeBump", {}, {"Normal": ("inputs", "Normal"),
                "Strength": ("strength", 0), "Distance": 1e-3, "Height": ("height", 0)}),

            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", "Normal")}),
            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("mix_color", 0), "Subsurface Color": ("mix_color", 0),
                "Subsurface": ("subsurface", 0), "Subsurface Radius": ("ssr", 0),
                "Roughness": ("inputs", "Roughness"), "Normal": ("bevel", "Normal")}),
        }

        outputs = {
            "BSDF":  ("NodeSocketShader", {}, ("shader", 0)),
            "Color": ("NodeSocketColor",  {}, ("mix_color", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

class ShaderNodeBsdfPcbSilkscreen(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "Silkscreen BSDF"
    bl_width_default = 180

    def update_props(self, context):
        match self.silkscreen:
            case "WHITE":
                color = hex2rgb("f3f1f0")
                roughness = 0.1
            case "BLACK":
                color = hex2rgb("100f0f")
                roughness = 0.2

        if not self.silkscreen == "CUSTOM":
            self.inputs["Color"].default_value = (*srgb2lin(color), 1.0)
            self.inputs["Roughness"].default_value = roughness

        hidden = self.silkscreen != "CUSTOM"
        for input_name in ("Color", "Roughness"):
            self.inputs[input_name].hide = hidden

    silkscreen: EnumProperty(name="Surface Finish", update=update_props, items=(
        ("WHITE", "White",   ""),
        ("BLACK", "Black",   ""),
        ("CUSTOM", "Custom", ""),
    ))

    def init(self, context):
        inputs = {
            "Color": ("NodeSocketColor", {}),
            "Roughness": ("NodeSocketFloat",  {}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": ("ShaderNodeTexNoise", {}, {"Vector": ("tex_coord", "Object"),
                "Scale": 4000.0, "Detail": 0.0, "Distortion": 0.1}),
            "bump_strength": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("inputs", "Texture Strength"), 1: 0.025}),
            "bump": ("ShaderNodeBump", {}, {
                "Strength": ("bump_strength", 0), "Distance": 1e-3,
                "Height": ("noise", 0), "Normal": ("inputs", "Normal")}),
            
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 5e-5, "Normal": ("bump", "Normal")}),
            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("inputs", "Color"), "Roughness": ("inputs", "Roughness"),
                "Clearcoat": 0.75, "Normal": ("bevel", "Normal")}),
        }

        outputs = {
            "BSDF":  ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

class ShaderNodeBsdfPcbBoardEdge(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "Board Edge BSDF"
    bl_width_default = 180

    def init(self, context):
        inputs = {
            "Base Color": ("NodeSocketColor",
                {"default_value": (*srgb2lin(PCB_YELLOW), 1.0)}),
            "Mask Color": ("NodeSocketColor",
                {"default_value": (*srgb2lin(hex2rgb("28a125")), 1.0)}),
            "Mix": ("NodeSocketFloat", {"default_value": 0.95}),
            "Roughness": ("NodeSocketFloat",  {"default_value": 0.6}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 0.5}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": ("ShaderNodeTexVoronoi", {"feature": "SMOOTH_F1"},
                {"Vector": ("tex_coord", "Object"), "Scale": 15000.0}),
            "noise_mapped": ("ShaderNodeMapRange", {}, {"Value": ("noise", "Distance"),
                "From Max": 0.5, "To Min": 0.75, "To Max": -0.25}),
            "noise_scaled": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("noise_mapped", 0), 1: ("inputs", "Texture Strength")}),
            "bump": ("ShaderNodeBump", {}, {"Strength": 0.5, "Distance": 1e-3,
                "Height": ("noise_scaled", 0), "Normal": ("inputs", "Normal")}),
            "roughness": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("noise_scaled", 0), 1: 0.2, 2: ("inputs", "Roughness")}),

            "base_color": ("ShaderNodeMixRGB", {}, {"Fac": ("inputs", "Mix"),
                "Color1": ("inputs", "Base Color"), "Color2": ("inputs", "Mask Color")}),
            "bump_color": ("ShaderNodeHueSaturation", {},
                {"Saturation": 1.1, "Value": 1.5, "Color": ("inputs", "Base Color")}),
            "color": ("ShaderNodeMixRGB", {}, {"Fac": ("noise_scaled", 0),
                "Color1": ("base_color", 0), "Color2": ("bump_color", 0)}),
            "ssr": ("ShaderNodeBrightContrast", {},
                {"Color": ("color", 0), "Bright": 0.1}),
            
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", "Normal")}),
            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("color", 0), "Subsurface Color": ("color", 0),
                "Subsurface": 0.001, "Subsurface Radius": ("ssr", 0),
                "Roughness": ("roughness", 0), "Normal": ("bevel", "Normal")}),
        }

        outputs = {
            "BSDF":  ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

class ShaderNodeBsdfSolder(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "Solder BSDF"
    bl_width_default = 180

    def init(self, context):
        inputs = {
            "Color": ("NodeSocketColor",
                {"default_value": (*srgb2lin(hex2rgb("ddddd8")), 1.0)}),
            "Roughness": ("NodeSocketFloat",  {"default_value": 0.25}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise_coarse": ("ShaderNodeTexNoise", {}, {"Vector": ("tex_coord", "Generated"),
                "Scale": 10.0, "Detail": 5.0, "Distortion": 0.1}),
            "noise_fine": ("ShaderNodeTexNoise", {}, {"Vector": ("tex_coord", "Generated"),
                "Scale": 100.0, "Detail": 5.0, "Roughness": 0.75, "Distortion": 0.5}),
            "noise_combined": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("noise_fine", "Fac"), 1: -0.2, 2: ("noise_coarse", "Fac")}),
            "noise_scaled": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("noise_combined", 0), 1: ("inputs", "Texture Strength")}),
            "bump": ("ShaderNodeBump", {}, {"Strength": 0.05, "Distance": 1e-3,
                "Height": ("noise_scaled", 0), "Normal": ("inputs", "Normal")}),
            "roughness": ("ShaderNodeMapRange", {},
                {"Value": ("noise_scaled", 0), "To Min": -0.15, "To Max": 0.15}),
            
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", "Normal")}),
            "shader": ("ShaderNodeBsdfPrincipled", {}, {
                "Base Color": ("inputs", "Color"), "Metallic": 1.0,
                "Roughness": ("roughness", 0), "Normal": ("bevel", "Normal")}),
        }

        outputs = {
            "BSDF":  ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

class ShaderNodePcbShader(CustomNodetreeNodeBase, ShaderNodeCustomGroup):
    bl_label = "PCB Shader"
    bl_width_default = 180

    def init(self, context):
        inputs = {
            "Silkscreen Quality": ("NodeSocketFloat", {"default_value": 0.8}),
            "Base Material":  ("NodeSocketShader", {}),
            "Exposed Copper": ("NodeSocketShader", {}),
            "Solder Mask":    ("NodeSocketShader", {}),
            "Silkscreen":     ("NodeSocketShader", {}),
            "Solder":         ("NodeSocketShader", {}),
            "Board Edge":     ("NodeSocketShader", {}),
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
            
            "cu_exposed": ("ShaderNodeMath", {"operation": "SUBTRACT", "use_clamp": True},
                {0: ("cu", 0), 1: ("mask", 0)}),
            "silks_cut": ("ShaderNodeMath", {"operation": "SUBTRACT", "use_clamp": True},
                {0: ("silks", 0), 1: ("cu_exposed", 0)}),

            "silks_noise": ("ShaderNodeTexNoise", {}, {"Vector": ("tex_coord", "Object"),
                "Scale": 5000, "Detail": 0.5, "Roughness": 0.1, "Distortion": 0.1}),
            "silks_quality": ("ShaderNodeMapRange", {"clamp": False}, {
                "Value": ("inputs", "Silkscreen Quality"), "To Min": -0.65, "To Max": -0.95}),
            "silks_combined": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("silks_cut", 0), 1: ("silks_quality", 0), 2: ("silks_noise", "Fac")}),
            "silks_damaged": ("ShaderNodeMath", {"operation": "MULTIPLY", "use_clamp": True},
                {0: ("silks_combined", 0), 1: -15.0}),

            "edge_vc": ("ShaderNodeAttribute", {"attribute_name": "Board Edge"}, {}),
            "cu_invert": ("ShaderNodeInvert", {}, {"Color": ("cu", 0)}),
            "plated_edge": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("edge_vc", "Fac"), 1: ("cu_invert", 0)}),
            "z_scaled": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("separate_position", "Z"), 1: 1000.0}),
            "z_abs": ("ShaderNodeMath", {"operation": "ABSOLUTE"}, {0: ("z_scaled", 0)}),
            "mask_blend": ("ShaderNodeValToRGB", {}, {0: ("z_abs", 0)}),
            "board_edge": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("plated_edge", 0), 1: ("mask_blend", 0)}),

            "mix_cu": ("ShaderNodeMixShader", {}, {"Fac": ("cu", 0),
                1: ("inputs", "Base Material"), 2: ("inputs", "Exposed Copper")}),
            "mix_mask": ("ShaderNodeMixShader", {}, {"Fac": ("mask", 0),
                1: ("mix_cu", 0), 2: ("inputs", "Solder Mask")}),
            "mix_silks": ("ShaderNodeMixShader", {}, {"Fac": ("silks_damaged", 0),
                1: ("mix_mask", 0), 2: ("inputs", "Silkscreen")}),
            "mix_solder": ("ShaderNodeMixShader", {}, {"Fac": ("paste", 0),
                1: ("mix_silks", 0), 2: ("inputs", "Solder")}),
            "mix_board_edge": ("ShaderNodeMixShader", {}, {"Fac": ("board_edge", 0),
                1: ("mix_solder", 0), 2: ("inputs", "Board Edge")}),

            "multiply_mask": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("mask", 0), 1: 0.02}),
            "multiply_cu": ("ShaderNodeMath", {"operation": "MULTIPLY_ADD"},
                {0: ("cu", 0), 1: 0.03, 2: ("multiply_mask", 0)}),
            "multiply_silks": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("silks_damaged", 0), 1: 0.0495}),
            "max_silks": ("ShaderNodeMath", {"operation": "MAXIMUM"},
                {0: ("multiply_cu", 0), 1: ("multiply_silks", 0)}),
            "multiply_scale": ("ShaderNodeMath", {"operation": "MULTIPLY"},
                {0: ("max_silks", 0), 1: 5e-3}),
            "displacement": ("ShaderNodeDisplacement", {},
                {"Height": ("multiply_scale", 0)}),
        }

        outputs = {
            "Shader": ("NodeSocketShader", {}, ("mix_board_edge", 0)),
            "Displacement": ("NodeSocketVector", {}, ("displacement", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

        color_ramp = self.node_tree.nodes["mask_blend"].color_ramp
        color_ramp.interpolation = "B_SPLINE"
        color_ramp.elements[0].color = (1, 1, 1, 1)
        color_ramp.elements[1].position = 0.75
        color_ramp.elements[1].color = (0, 0, 0, 0)
        color_ramp.elements.new(0.5).color = (1, 1, 1, 1)
        color_ramp.elements.new(0.7).color = (0, 0, 0, 0)

shader_node_category = ShaderNodeCategory("SH_NEW_PCB2BLENDER", "Pcb2Blender", items=(
    NodeItem("ShaderNodeBsdfMat4cad"),
    NodeItem("ShaderNodeBsdfPcbSurfaceFinish"),
    NodeItem("ShaderNodeBsdfPcbSolderMask"),
    NodeItem("ShaderNodeBsdfPcbSilkscreen"),
    NodeItem("ShaderNodeBsdfPcbBoardEdge"),
    NodeItem("ShaderNodeBsdfSolder"),
    NodeItem("ShaderNodePcbShader"),
))

classes = (
    ShaderNodeBsdfMat4cad,
    ShaderNodeBsdfPcbSurfaceFinish,
    ShaderNodeBsdfPcbSolderMask,
    ShaderNodeBsdfPcbSilkscreen,
    ShaderNodeBsdfPcbBoardEdge,
    ShaderNodeBsdfSolder,
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
