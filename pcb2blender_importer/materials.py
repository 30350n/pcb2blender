from typing import Any, Iterable, Literal, cast, overload

import bpy
from bl_ui import node_add_menu
from bpy.props import EnumProperty
from bpy.types import NodeSocketColor, NodeSocketFloat, NodeSocketShader, NodeSocketVector
from mathutils import Color, Vector

from .custom_node_utils import NodesDef, OutputsDef, SharedCustomNodetreeNodeBase, setup_node_tree
from .mat4cad.blender import (
    BlenderMaterial as Mat4CadMaterial,
    register as register_mat4cad,
    unregister as unregister_mat4cad,
)
from .mat4cad.colors import PCB_COLORS
from .mat4cad.core import hex2rgb, lin2srgb, rgb2hex, srgb2lin

LAYER_BOARD_EDGE = "pcb_board_edge"
LAYER_THROUGH_HOLES = "pcb_through_holes"

KICAD_2_MAT4CAD = {
    "PIN-01": "special-pins_silver-default",
    "PIN-02": "special-pins_gold-default",
    "MET-BRONZE": "metal-bronze-semi_matte",
    "MET-COPPER": "metal-copper-semi_matte",
    "IC-BODY-EPOXY-04": "plastic-traffic_black-matte",
    "IC-LABEL-01": "plastic-grey_white-semi_matte",
    "RES-THT-01": "plastic-beige-semi_matte",
    "RES-SMD-01": "plastic-iron_grey-semi_matte",
    "CAP-CERAMIC-05": "plastic-signal_grey-semi_matte",
    "CAP-CERAMIC-06": "plastic-grey_beige-semi_matte",
    "BOARD-BLACK-03": "plastic-mouse_grey-semi_matte",
    "PLASTIC-BLUE-01": "plastic-brilliant_blue-semi_matte",
    "PLASTIC-ORANGE-01": "plastic-saffron_yellow-semi_matte",
    "PLASTIC-WHITE-01": "plastic-pure_white-semi_matte",
    "PLASTIC-YELLOW-01": "plastic-zinc_yellow-semi_matte",
    "PLASTIC-GREEN-01": "plastic-pale_green-semi_matte",
    "RED-BODY": "plastic-salmon_orange-semi_matte",
    "GLASS-13": "plastic_transparent-turquoise_blue-diffused",
    "LED-RED": "plastic_transparent-orient_red-diffused",
}


def merge_materials(meshes: Iterable[bpy.types.Mesh]):
    merged_materials = {}
    for mesh in meshes:
        for i, material in enumerate(mesh.materials):
            name = remove_blender_name_suffix(material.name)
            color = rgb2hex(material.diffuse_color[:])
            if merged_material := merged_materials.get((name, color)):
                mesh.materials[i] = merged_material
            else:
                material.name = name
                merged_materials[(name, color)] = material


def enhance_materials(materials: Iterable[bpy.types.Material]):
    for material in materials:
        if bpy.app.version < (5, 0, 0) and not material.use_nodes:
            continue
        assert (node_tree := material.node_tree)

        material_name = remove_blender_name_suffix(material.name)
        if mat4cad_mat := Mat4CadMaterial.from_name(material_name):
            pass
        elif mat4cad_mat := Mat4CadMaterial.from_name(KICAD_2_MAT4CAD.get(material_name, "")):
            pass
        else:
            shader_nodes = (
                cast(bpy.types.ShaderNodeBsdfPrincipled, node)
                for node in node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            )
            if not (node_shader := next(shader_nodes, None)):
                continue

            color = Color(node_shader.inputs["Base Color"].default_value[:3])
            probably_metal = color.s < 0.1 and color.v > 0.25
            base_material = "metal" if probably_metal else "plastic"
            color_hex = rgb2hex(lin2srgb(color[:]))

            mat4cad_mat = Mat4CadMaterial.from_name(
                f"{base_material}-custom_{color_hex}-semi_matte"
            )
            assert mat4cad_mat

        mat4cad_mat.setup_node_tree(node_tree)


def remove_blender_name_suffix(name: str):
    if name[-4:-3] != ".":
        return name

    prefix, suffix = name.rsplit(".", 1)
    if not suffix.isnumeric():
        return name

    return prefix


def setup_pcb_material(
    node_tree: bpy.types.ShaderNodeTree, images: dict[str, bpy.types.Image], stackup: Any
):
    node_tree.nodes.clear()

    surface_finish = stackup.surface_finish.name

    soldermask = stackup.mask_color.name
    soldermask_inputs = {}
    if soldermask == "CUSTOM":
        color = Vector(srgb2lin(stackup.mask_color_custom))
        soldermask_inputs = {"Light Color": color.to_4d(), "Dark Color": (color * 0.2).to_4d()}

    silkscreen = stackup.silks_color.name
    silkscreen_color = stackup.silks_color_custom
    if color := SILKS_COLOR_MAP.get(silkscreen):
        silkscreen = "CUSTOM"
        silkscreen_color = color

    silkscreen_inputs = {}
    if silkscreen == "CUSTOM":
        silkscreen_inputs = {"Color": Vector(silkscreen_color).to_4d(), "Roughness": 0.25}

    nodes: NodesDef = {
        "cu": (
            "ShaderNodeTexImage",
            {
                "location": (-500, -320),
                "hide": True,
                "interpolation": "Cubic",
                "image": images["Cu"],
            },
            {},
        ),
        "mask": (
            "ShaderNodeTexImage",
            {
                "location": (-500, -360),
                "hide": True,
                "interpolation": "Cubic",
                "image": images["Mask"],
            },
            {},
        ),
        "silks": (
            "ShaderNodeTexImage",
            {
                "location": (-500, -400),
                "hide": True,
                "interpolation": "Cubic",
                "image": images["SilkS"],
            },
            {},
        ),
        "paste": (
            "ShaderNodeTexImage",
            {
                "location": (-500, -440),
                "hide": True,
                "interpolation": "Cubic",
                "image": images["Paste"],
            },
            {},
        ),
        "seperate_cu": (
            "ShaderNodeSeparateColor",
            {"location": (-200, -320), "hide": True},
            {"Color": ("cu", "Color")},
        ),
        "seperate_mask": (
            "ShaderNodeSeparateColor",
            {"location": (-200, -360), "hide": True},
            {"Color": ("mask", "Color")},
        ),
        "seperate_silks": (
            "ShaderNodeSeparateColor",
            {"location": (-200, -400), "hide": True},
            {"Color": ("silks", "Color")},
        ),
        "seperate_paste": (
            "ShaderNodeSeparateColor",
            {"location": (-200, -440), "hide": True},
            {"Color": ("paste", "Color")},
        ),
        "base_material": (
            "ShaderNodeBsdfMat4cad",
            {"location": (-260, 0), "mat_base": "PCB", "mat_color": "PCB_YELLOW"},
            {},
        ),
        "exposed_copper": (
            "ShaderNodeBsdfPcbSurfaceFinish",
            {"location": (-500, 0), "surface_finish": surface_finish},
            {},
        ),
        "solder_mask": (
            "ShaderNodeBsdfPcbSolderMask",
            {"location": (-740, 0), "soldermask": soldermask},
            {"F_Cu": ("seperate_cu", "Red"), "B_Cu": ("seperate_cu", "Green"), **soldermask_inputs},
        ),
        "board_edge": (
            "ShaderNodeBsdfPcbBoardEdge",
            {"location": (-740, -280)},
            {"Mask Color": ("solder_mask", "Color")},
        ),
        "silkscreen": (
            "ShaderNodeBsdfPcbSilkscreen",
            {"location": (-980, 0), "silkscreen": silkscreen},
            {**silkscreen_inputs},
        ),
        "solder": ("ShaderNodeBsdfSolder", {"location": (-1220, 0)}, {}),
        "shader": (
            "ShaderNodePcbShader",
            {},
            {
                "F_Cu": ("seperate_cu", "Red"),
                "B_Cu": ("seperate_cu", "Green"),
                "F_Mask": ("seperate_mask", "Red"),
                "B_Mask": ("seperate_mask", "Green"),
                "F_SilkS": ("seperate_silks", "Red"),
                "B_SilkS": ("seperate_silks", "Green"),
                "F_Paste": ("seperate_paste", "Red"),
                "B_Paste": ("seperate_paste", "Green"),
                "Base Material": ("base_material", "BSDF"),
                "Exposed Copper": ("exposed_copper", "BSDF"),
                "Solder Mask": ("solder_mask", "BSDF"),
                "Silkscreen": ("silkscreen", "BSDF"),
                "Solder": ("solder", "BSDF"),
                "Board Edge": ("board_edge", "BSDF"),
            },
        ),
        "output": (
            "ShaderNodeOutputMaterial",
            {"location": (240, 0)},
            {"Surface": ("shader", "Shader"), "Displacement": ("shader", "Displacement")},
        ),
    }

    setup_node_tree(node_tree, nodes, label_nodes=False)


class ShaderNodeBsdfPcbSurfaceFinish(SharedCustomNodetreeNodeBase):
    bl_label = "Surface Finish BSDF"
    bl_width_default = 180

    def update_props(self, context: bpy.types.Context):
        surface_finish = cast(Literal["HASL", "ENIG", "NONE", "CUSTOM"], self.surface_finish)
        if surface_finish != "CUSTOM":
            match surface_finish:
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

            self.inputs["Color"].default_value = (*srgb2lin(color), 1.0)
            self.inputs["Roughness"].default_value = roughness
            self.inputs["Texture Strength"].default_value = texture_strength

        hidden = self.surface_finish != "CUSTOM"
        for input_name in ("Color", "Roughness", "Texture Strength"):
            self.inputs[input_name].hide = hidden

    surface_finish: EnumProperty(
        name="Surface Finish",
        update=update_props,
        items=(
            ("HASL", "HASL", ""),
            ("ENIG", "ENIG", ""),
            ("NONE", "None", ""),
            ("CUSTOM", "Custom", ""),
        ),
    )

    def init(self, context: bpy.types.Context):
        inputs = {
            "Color": ("NodeSocketColor", {}),
            "Roughness": ("NodeSocketFloat", {}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": (
                "ShaderNodeTexNoise",
                {},
                {
                    "Vector": ("tex_coord", "Object"),
                    "Scale": 500.0,
                    "Detail": 4.0,
                    "Roughness": 0.4,
                    "Distortion": 0.1,
                },
            ),
            "noise_bipolar": (
                "ShaderNodeMapRange",
                {},
                {"Value": ("noise", "Fac"), "To Min": -0.5, "To Max": 0.5},
            ),
            "noise_scaled": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("noise_bipolar", 0), 1: ("inputs", "Texture Strength")},
            ),
            "bump": (
                "ShaderNodeBump",
                {},
                {
                    "Strength": 0.1,
                    "Distance": 1e-3,
                    "Height": ("noise_scaled", 0),
                    "Normal": ("inputs", "Normal"),
                },
            ),
            "roughness": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("noise_scaled", 0), 1: 0.2, 2: ("inputs", "Roughness")},
            ),
            "scratches_strength": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("inputs", "Texture Strength"), 1: 0.5},
            ),
            "scratches": (
                "ShaderNodeMat4cadScratches",
                {},
                {
                    "Color": ("inputs", "Color"),
                    "Roughness": ("roughness", 0),
                    "Normal": ("bump", 0),
                    "Strength": ("scratches_strength", 0),
                    "Vector": ("tex_coord", "Object"),
                    "Scale": 3.0,
                },
            ),
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("scratches", "Normal")}),
            "shader": (
                "ShaderNodeBsdfPrincipled",
                {},
                {
                    "Base Color": ("scratches", "Color"),
                    "Metallic": 1.0,
                    "Roughness": ("scratches", "Roughness"),
                    "Normal": ("bevel", 0),
                },
            ),
        }

        outputs: OutputsDef = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Roughness"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Texture Strength"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Normal"]) -> NodeSocketVector: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        def __getitem__(self, key: Literal[0] | Literal["BSDF"]) -> NodeSocketShader: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


MASK_COLOR_MAP = {
    "GREEN": (hex2rgb("43a142"), hex2rgb("22521f")),
    "RED": (hex2rgb("de544c"), hex2rgb("733a38")),
    "YELLOW": (hex2rgb("dacb57"), hex2rgb("6a7c30")),
    "BLUE": (hex2rgb("3b69aa"), hex2rgb("1c3659")),
    "PURPLE": (hex2rgb("7448aa"), hex2rgb("3b2359")),
    "WHITE": (hex2rgb("d3cfc9"), hex2rgb("e1dddc")),
    "BLACK": (hex2rgb("10100f"), hex2rgb("000000")),
    "MATTE_BLACK": (hex2rgb("191919"), hex2rgb("191919")),
}
DEFAULT_MASK_ROUGHNESS = 0.45
MASK_ROUGHNESS_MAP = {
    "WHITE": 0.4,
    "MATTE_BLACK": 0.8,
}


class ShaderNodeBsdfPcbSolderMask(SharedCustomNodetreeNodeBase):
    bl_label = "Solder Mask BSDF"
    bl_width_default = 180

    def update_props(self, context: bpy.types.Context):
        is_custom = self.soldermask == "CUSTOM"
        for input_name in ("Light Color", "Dark Color", "Roughness"):
            self.inputs[input_name].hide = not is_custom
        if is_custom:
            return

        light_color, dark_color = MASK_COLOR_MAP[self.soldermask]
        roughness = MASK_ROUGHNESS_MAP.get(self.soldermask, DEFAULT_MASK_ROUGHNESS)

        self.inputs["Light Color"].default_value = (*srgb2lin(light_color), 1.0)
        self.inputs["Dark Color"].default_value = (*srgb2lin(dark_color), 1.0)
        self.inputs["Roughness"].default_value = roughness

    soldermask: EnumProperty(
        name="Solder Mask",
        update=update_props,
        items=(
            *(
                (name, " ".join(word.capitalize() for word in name.split("_")), "")
                for name in MASK_COLOR_MAP
            ),
            ("CUSTOM", "Custom", ""),
        ),
    )

    def init(self, context: bpy.types.Context):
        inputs = {
            "Light Color": ("NodeSocketColor", {}),
            "Dark Color": ("NodeSocketColor", {}),
            "Roughness": ("NodeSocketFloat", {"default_value": 0.25}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
            "F_Cu": ("NodeSocketFloat", {"hide_value": True}),
            "B_Cu": ("NodeSocketFloat", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "separate_position": ("ShaderNodeSeparateXYZ", {}, {"Vector": ("tex_coord", "Object")}),
            "is_bottom_layer": (
                "ShaderNodeMath",
                {"operation": "LESS_THAN"},
                {0: ("separate_position", "Z"), 1: 0.0},
            ),
            "cu": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("is_bottom_layer", 0),
                    "Color1": ("inputs", "F_Cu"),
                    "Color2": ("inputs", "B_Cu"),
                },
            ),
            "cu_invert": ("ShaderNodeInvert", {}, {"Color": ("cu", 0)}),
            "mix_color": (
                "ShaderNodeMixRGB",
                {},
                {"Fac": ("cu", 0), 1: ("inputs", "Dark Color"), 2: ("inputs", "Light Color")},
            ),
            "subsurface": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("cu_invert", 0), 1: 0.001},
            ),
            "ssr": ("ShaderNodeBrightContrast", {}, {"Color": ("mix_color", 0), "Bright": 0.1}),
            "strength_noise": (
                "ShaderNodeTexNoise",
                {},
                {"Vector": ("tex_coord", "Object"), "Scale": 100.0, "Detail": 0.0},
            ),
            "strength_scaled": (
                "ShaderNodeMapRange",
                {},
                {"Value": ("strength_noise", "Fac"), "To Min": -0.012, "To Max": 0.044},
            ),
            "strength": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("strength_scaled", 0), 1: ("inputs", "Texture Strength")},
            ),
            "scale": (
                "ShaderNodeVectorMath",
                {"operation": "MULTIPLY"},
                {0: ("tex_coord", "Object"), 1: Vector((3000.0, 3000.0, 3000.0))},
            ),
            "checkers": (
                "ShaderNodeTexVoronoi",
                {"distance": "MINKOWSKI"},
                {"Vector": ("scale", 0), "Scale": 1.0, "Exponent": 1.8, "Randomness": 0.1},
            ),
            "scale2": (
                "ShaderNodeVectorMath",
                {"operation": "MULTIPLY"},
                {0: ("scale", 0), 1: Vector((1.0, 0.0, 0.25))},
            ),
            "lines": (
                "ShaderNodeTexVoronoi",
                {"distance": "MINKOWSKI"},
                {"Vector": ("scale2", 0), "Scale": 1.0, "Exponent": 1.8, "Randomness": 0.1},
            ),
            "combined": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("checkers", "Distance"), 1: 0.25, 2: ("lines", "Distance")},
            ),
            "height_scale": ("ShaderNodeMapRange", {}, {"Value": ("cu", 0), "To Min": 0.25}),
            "height": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("combined", 0), 1: ("height_scale", 0)},
            ),
            "bump": (
                "ShaderNodeBump",
                {},
                {
                    "Normal": ("inputs", "Normal"),
                    "Strength": ("strength", 0),
                    "Distance": 1e-3,
                    "Height": ("height", 0),
                },
            ),
            "roughness_cu": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("inputs", "Roughness"), 1: 0.5},
            ),
            "roughness": (
                "ShaderNodeMapRange",
                {},
                {
                    "Value": ("cu", 0),
                    "From Min": 1.0,
                    "From Max": 0.0,
                    "To Min": ("roughness_cu", 0),
                    "To Max": ("inputs", "Roughness"),
                },
            ),
            "noise_strength": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("inputs", "Texture Strength"), 1: 0.65},
            ),
            "noise_scale": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("cu", 0), 1: 150.0, 2: 0.5},
            ),
            "noise": (
                "ShaderNodeMat4cadNoise",
                {},
                {
                    "Roughness": ("roughness", 0),
                    "Normal": ("bump", 0),
                    "Strength": ("noise_strength", 0),
                    "Vector": ("tex_coord", "Object"),
                    "Scale": ("noise_scale", 0),
                },
            ),
            "scratches_strength": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("inputs", "Texture Strength"), 1: 0.3},
            ),
            "scratches": (
                "ShaderNodeMat4cadScratches",
                {},
                {
                    "Color": ("mix_color", 0),
                    "Roughness": ("noise", "Roughness"),
                    "Normal": ("noise", "Normal"),
                    "Strength": ("scratches_strength", 0),
                    "Vector": ("tex_coord", "Object"),
                    "Scale": 2.5,
                },
            ),
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("scratches", "Normal")}),
            "shader": (
                "ShaderNodeBsdfPrincipled",
                {},
                {
                    "Base Color": ("scratches", "Color"),
                    "Metallic": 0.6,
                    "Subsurface Weight": ("subsurface", 0),
                    "Subsurface Radius": ("ssr", 0),
                    "Roughness": ("scratches", "Roughness"),
                    "Normal": ("bevel", 0),
                },
            ),
        }

        outputs: OutputsDef = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
            "Color": ("NodeSocketColor", {}, ("mix_color", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Light Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Dark Color"]) -> NodeSocketColor: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Roughness"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Texture Strength"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[4] | Literal["Normal"]) -> NodeSocketVector: ...
        @overload
        def __getitem__(self, key: Literal[5] | Literal["F_Cu"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[6] | Literal["B_Cu"]) -> NodeSocketFloat: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["BSDF"]) -> NodeSocketShader: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


SILKS_COLOR_MAP = {
    "GREEN": hex2rgb("28a125"),
    "RED": hex2rgb("e50007"),
    "BLUE": hex2rgb("116cc2"),
    "PURPLE": hex2rgb("4f33c2"),
    "YELLOW": hex2rgb("dac92b"),
}


class ShaderNodeBsdfPcbSilkscreen(SharedCustomNodetreeNodeBase):
    bl_label = "Silkscreen BSDF"
    bl_width_default = 180

    def update_props(self, context: bpy.types.Context):
        silkscreen = cast(Literal["WHITE", "BLACK", "CUSTOM"], self.silkscreen)
        if silkscreen != "CUSTOM":
            match silkscreen:
                case "WHITE":
                    color = hex2rgb("f3f1f0")
                    roughness = 0.1
                case "BLACK":
                    color = hex2rgb("100f0f")
                    roughness = 0.2

            self.inputs["Color"].default_value = (*srgb2lin(color), 1.0)
            self.inputs["Roughness"].default_value = roughness

        for input_name in ("Color", "Roughness"):
            self.inputs[input_name].hide = self.silkscreen != "CUSTOM"

    silkscreen: EnumProperty(
        name="Silkscreen",
        update=update_props,
        items=(
            ("WHITE", "White", ""),
            ("BLACK", "Black", ""),
            ("CUSTOM", "Custom", ""),
        ),
    )

    def init(self, context: bpy.types.Context):
        inputs = {
            "Color": ("NodeSocketColor", {}),
            "Roughness": ("NodeSocketFloat", {}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": (
                "ShaderNodeTexNoise",
                {},
                {
                    "Vector": ("tex_coord", "Object"),
                    "Scale": 4000.0,
                    "Detail": 0.0,
                    "Distortion": 0.1,
                },
            ),
            "bump_strength": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("inputs", "Texture Strength"), 1: 0.025},
            ),
            "bump": (
                "ShaderNodeBump",
                {},
                {
                    "Strength": ("bump_strength", 0),
                    "Distance": 1e-3,
                    "Height": ("noise", 0),
                    "Normal": ("inputs", "Normal"),
                },
            ),
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 5e-5, "Normal": ("bump", 0)}),
            "shader": (
                "ShaderNodeBsdfPrincipled",
                {},
                {
                    "Base Color": ("inputs", "Color"),
                    "Roughness": ("inputs", "Roughness"),
                    "Coat Weight": 0.75,
                    "Normal": ("bevel", 0),
                },
            ),
        }

        outputs: OutputsDef = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        self.update_props(context)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Roughness"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Texture Strength"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Normal"]) -> NodeSocketVector: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        def __getitem__(self, key: Literal[0] | Literal["BSDF"]) -> NodeSocketShader: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


class ShaderNodeBsdfPcbBoardEdge(SharedCustomNodetreeNodeBase):
    bl_label = "Board Edge BSDF"
    bl_width_default = 180

    def init(self, context: bpy.types.Context):
        inputs = {
            "Base Color": (
                "NodeSocketColor",
                {"default_value": (*srgb2lin(PCB_COLORS["pcb_yellow"]), 1.0)},
            ),
            "Mask Color": (
                "NodeSocketColor",
                {"default_value": (*srgb2lin(hex2rgb("28a125")), 1.0)},
            ),
            "Mix": ("NodeSocketFloat", {"default_value": 0.95}),
            "Roughness": ("NodeSocketFloat", {"default_value": 0.6}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 0.5}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise": (
                "ShaderNodeTexVoronoi",
                {"feature": "SMOOTH_F1"},
                {"Vector": ("tex_coord", "Object"), "Scale": 15000.0},
            ),
            "noise_mapped": (
                "ShaderNodeMapRange",
                {},
                {"Value": ("noise", "Distance"), "From Max": 0.5, "To Min": 0.75, "To Max": -0.25},
            ),
            "noise_scaled": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("noise_mapped", 0), 1: ("inputs", "Texture Strength")},
            ),
            "bump": (
                "ShaderNodeBump",
                {},
                {
                    "Strength": 0.5,
                    "Distance": 1e-3,
                    "Height": ("noise_scaled", 0),
                    "Normal": ("inputs", "Normal"),
                },
            ),
            "roughness": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("noise_scaled", 0), 1: 0.2, 2: ("inputs", "Roughness")},
            ),
            "base_color": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("inputs", "Mix"),
                    "Color1": ("inputs", "Base Color"),
                    "Color2": ("inputs", "Mask Color"),
                },
            ),
            "bump_color": (
                "ShaderNodeHueSaturation",
                {},
                {"Saturation": 1.1, "Value": 1.5, "Color": ("inputs", "Base Color")},
            ),
            "color": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("noise_scaled", 0),
                    "Color1": ("base_color", 0),
                    "Color2": ("bump_color", 0),
                },
            ),
            "ssr": ("ShaderNodeBrightContrast", {}, {"Color": ("color", 0), "Bright": 0.1}),
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", 0)}),
            "shader": (
                "ShaderNodeBsdfPrincipled",
                {},
                {
                    "Base Color": ("color", 0),
                    "Subsurface Weight": 0.001,
                    "Subsurface Radius": ("ssr", 0),
                    "Roughness": ("roughness", 0),
                    "Normal": ("bevel", 0),
                },
            ),
        }

        outputs: OutputsDef = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Base Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Mask Color"]) -> NodeSocketColor: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Mix"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Roughness"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[4] | Literal["Texture Strength"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[5] | Literal["Normal"]) -> NodeSocketVector: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        def __getitem__(self, key: Literal[0] | Literal["BSDF"]) -> NodeSocketShader: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


class ShaderNodeBsdfSolder(SharedCustomNodetreeNodeBase):
    bl_label = "Solder BSDF"
    bl_width_default = 180

    def init(self, context: bpy.types.Context):
        inputs = {
            "Color": ("NodeSocketColor", {"default_value": (*srgb2lin(hex2rgb("aaaaa6")), 1.0)}),
            "Roughness": ("NodeSocketFloat", {"default_value": 0.25}),
            "Texture Strength": ("NodeSocketFloat", {"default_value": 1.0}),
            "Normal": ("NodeSocketVector", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "noise_coarse": (
                "ShaderNodeTexNoise",
                {},
                {
                    "Vector": ("tex_coord", "Generated"),
                    "Scale": 10.0,
                    "Detail": 5.0,
                    "Distortion": 0.1,
                },
            ),
            "noise_fine": (
                "ShaderNodeTexNoise",
                {},
                {
                    "Vector": ("tex_coord", "Generated"),
                    "Scale": 100.0,
                    "Detail": 5.0,
                    "Roughness": 0.75,
                    "Distortion": 0.5,
                },
            ),
            "noise_combined": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("noise_fine", "Fac"), 1: -0.2, 2: ("noise_coarse", "Fac")},
            ),
            "noise_scaled": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("noise_combined", 0), 1: ("inputs", "Texture Strength")},
            ),
            "bump": (
                "ShaderNodeBump",
                {},
                {
                    "Strength": 0.05,
                    "Distance": 1e-3,
                    "Height": ("noise_scaled", 0),
                    "Normal": ("inputs", "Normal"),
                },
            ),
            "roughness": (
                "ShaderNodeMapRange",
                {},
                {
                    "Value": ("noise_scaled", 0),
                    "To Min": ("inputs", "Roughness"),
                    "From Min": 0.4,
                    "From Max": 0.8,
                    "To Max": 0.05,
                },
            ),
            "bevel": ("ShaderNodeBevel", {}, {"Radius": 1e-4, "Normal": ("bump", 0)}),
            "shader": (
                "ShaderNodeBsdfPrincipled",
                {},
                {
                    "Base Color": ("inputs", "Color"),
                    "Metallic": 1.0,
                    "Roughness": ("roughness", 0),
                    "Normal": ("bevel", 0),
                },
            ),
        }

        outputs: OutputsDef = {
            "BSDF": ("NodeSocketShader", {}, ("shader", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Color"]) -> NodeSocketColor: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Roughness"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Texture Strength"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Normal"]) -> NodeSocketVector: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        def __getitem__(self, key: Literal[0] | Literal["BSDF"]) -> NodeSocketShader: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


class ShaderNodePcbShader(SharedCustomNodetreeNodeBase):
    bl_label = "PCB Shader"
    bl_width_default = 180

    def init(self, context: bpy.types.Context):
        inputs = {
            "Silkscreen Quality": ("NodeSocketFloat", {"default_value": 0.8}),
            "Base Material": ("NodeSocketShader", {}),
            "Exposed Copper": ("NodeSocketShader", {}),
            "Solder Mask": ("NodeSocketShader", {}),
            "Silkscreen": ("NodeSocketShader", {}),
            "Solder": ("NodeSocketShader", {}),
            "Board Edge": ("NodeSocketShader", {}),
            "F_Cu": ("NodeSocketFloat", {"hide_value": True}),
            "B_Cu": ("NodeSocketFloat", {"hide_value": True}),
            "F_Mask": ("NodeSocketFloat", {"hide_value": True}),
            "B_Mask": ("NodeSocketFloat", {"hide_value": True}),
            "F_SilkS": ("NodeSocketFloat", {"hide_value": True}),
            "B_SilkS": ("NodeSocketFloat", {"hide_value": True}),
            "F_Paste": ("NodeSocketFloat", {"hide_value": True}),
            "B_Paste": ("NodeSocketFloat", {"hide_value": True}),
        }

        nodes: NodesDef = {
            "tex_coord": ("ShaderNodeTexCoord", {}, {}),
            "separate_position": ("ShaderNodeSeparateXYZ", {}, {"Vector": ("tex_coord", "Object")}),
            "is_bottom_layer": (
                "ShaderNodeMath",
                {"operation": "LESS_THAN"},
                {0: ("separate_position", "Z"), 1: 0.0},
            ),
            "cu": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("is_bottom_layer", 0),
                    "Color1": ("inputs", "F_Cu"),
                    "Color2": ("inputs", "B_Cu"),
                },
            ),
            "mask": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("is_bottom_layer", 0),
                    "Color1": ("inputs", "F_Mask"),
                    "Color2": ("inputs", "B_Mask"),
                },
            ),
            "silks": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("is_bottom_layer", 0),
                    "Color1": ("inputs", "F_SilkS"),
                    "Color2": ("inputs", "B_SilkS"),
                },
            ),
            "paste": (
                "ShaderNodeMixRGB",
                {},
                {
                    "Fac": ("is_bottom_layer", 0),
                    "Color1": ("inputs", "F_Paste"),
                    "Color2": ("inputs", "B_Paste"),
                },
            ),
            "cu_exposed": (
                "ShaderNodeMath",
                {"operation": "SUBTRACT", "use_clamp": True},
                {0: ("cu", 0), 1: ("mask", 0)},
            ),
            "silks_cut": (
                "ShaderNodeMath",
                {"operation": "SUBTRACT", "use_clamp": True},
                {0: ("silks", 0), 1: ("cu_exposed", 0)},
            ),
            "edge_vc": ("ShaderNodeAttribute", {"attribute_name": LAYER_BOARD_EDGE}, {}),
            "hole_vc": ("ShaderNodeAttribute", {"attribute_name": LAYER_THROUGH_HOLES}, {}),
            "combined_vc": (
                "ShaderNodeMath",
                {"operation": "ADD", "use_clamp": True},
                {0: ("edge_vc", "Fac"), 1: ("hole_vc", "Fac")},
            ),
            "silks_cut2": (
                "ShaderNodeMath",
                {"operation": "SUBTRACT", "use_clamp": True},
                {0: ("silks_cut", 0), 1: ("combined_vc", 0)},
            ),
            "silks_noise": (
                "ShaderNodeTexNoise",
                {},
                {
                    "Vector": ("tex_coord", "Object"),
                    "Scale": 5000,
                    "Detail": 0.5,
                    "Roughness": 0.1,
                    "Distortion": 0.1,
                },
            ),
            "silks_quality": (
                "ShaderNodeMapRange",
                {"clamp": False},
                {"Value": ("inputs", "Silkscreen Quality"), "To Min": -0.65, "To Max": -0.95},
            ),
            "silks_combined": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("silks_cut2", 0), 1: ("silks_quality", 0), 2: ("silks_noise", "Fac")},
            ),
            "silks_damaged": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY", "use_clamp": True},
                {0: ("silks_combined", 0), 1: -15.0},
            ),
            "cu_invert": ("ShaderNodeInvert", {}, {"Color": ("cu", 0)}),
            "plated_edge": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("edge_vc", "Fac"), 1: ("cu_invert", 0)},
            ),
            "z_scaled": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("separate_position", "Z"), 1: 1000.0},
            ),
            "z_abs": ("ShaderNodeMath", {"operation": "ABSOLUTE"}, {0: ("z_scaled", 0)}),
            "mask_blend": ("ShaderNodeValToRGB", {}, {0: ("z_abs", 0)}),
            "board_edge": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("plated_edge", 0), 1: ("mask_blend", 0)},
            ),
            "mix_cu": (
                "ShaderNodeMixShader",
                {},
                {"Fac": ("cu", 0), 1: ("inputs", "Base Material"), 2: ("inputs", "Exposed Copper")},
            ),
            "mix_mask": (
                "ShaderNodeMixShader",
                {},
                {"Fac": ("mask", 0), 1: ("mix_cu", 0), 2: ("inputs", "Solder Mask")},
            ),
            "mix_silks": (
                "ShaderNodeMixShader",
                {},
                {"Fac": ("silks_damaged", 0), 1: ("mix_mask", 0), 2: ("inputs", "Silkscreen")},
            ),
            "mix_solder": (
                "ShaderNodeMixShader",
                {},
                {"Fac": ("paste", 0), 1: ("mix_silks", 0), 2: ("inputs", "Solder")},
            ),
            "mix_board_edge": (
                "ShaderNodeMixShader",
                {},
                {"Fac": ("board_edge", 0), 1: ("mix_solder", 0), 2: ("inputs", "Board Edge")},
            ),
            "multiply_mask": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("mask", 0), 1: 0.02},
            ),
            "multiply_cu": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY_ADD"},
                {0: ("cu", 0), 1: 0.03, 2: ("multiply_mask", 0)},
            ),
            "multiply_silks": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("silks_damaged", 0), 1: 0.0495},
            ),
            "max_silks": (
                "ShaderNodeMath",
                {"operation": "MAXIMUM"},
                {0: ("multiply_cu", 0), 1: ("multiply_silks", 0)},
            ),
            "multiply_scale": (
                "ShaderNodeMath",
                {"operation": "MULTIPLY"},
                {0: ("max_silks", 0), 1: 5e-3},
            ),
            "displacement": ("ShaderNodeDisplacement", {}, {"Height": ("multiply_scale", 0)}),
        }

        outputs: OutputsDef = {
            "Shader": ("NodeSocketShader", {}, ("mix_board_edge", 0)),
            "Displacement": ("NodeSocketVector", {}, ("displacement", 0)),
        }

        self.init_node_tree(inputs, nodes, outputs)
        assert self.node_tree

        color_ramp_node = cast(bpy.types.ShaderNodeValToRGB, self.node_tree.nodes["mask_blend"])
        assert (color_ramp := color_ramp_node.color_ramp)
        color_ramp.interpolation = "B_SPLINE"
        color_ramp.elements[0].color = (1, 1, 1, 1)
        color_ramp.elements[1].position = 0.75
        color_ramp.elements[1].color = (0, 0, 0, 0)
        color_ramp.elements.new(0.5).color = (1, 1, 1, 1)
        color_ramp.elements.new(0.7).color = (0, 0, 0, 0)

    class NodeInputs(bpy.types.NodeInputs):
        @overload
        def __getitem__(  # pyright: ignore[reportNoOverloadImplementation]
            self, key: Literal[0] | Literal["Silkscreen Quality"]
        ) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Base Material"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[2] | Literal["Exposed Copper"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[3] | Literal["Solder Mask"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[4] | Literal["Silkscreen"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[5] | Literal["Solder"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[6] | Literal["Board Edge"]) -> NodeSocketShader: ...
        @overload
        def __getitem__(self, key: Literal[7] | Literal["F_Cu"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[8] | Literal["B_Cu"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[9] | Literal["F_Mask"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[10] | Literal["B_Mask"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[11] | Literal["F_SilkS"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[12] | Literal["B_SilkS"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[13] | Literal["F_Paste"]) -> NodeSocketFloat: ...
        @overload
        def __getitem__(self, key: Literal[14] | Literal["B_Paste"]) -> NodeSocketFloat: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    inputs: NodeInputs  # pyright: ignore[reportIncompatibleVariableOverride]

    class NodeOutputs(bpy.types.NodeOutputs):
        @overload
        def __getitem__(self, key: Literal[0] | Literal["Shader"]) -> NodeSocketShader: ...  # pyright: ignore[reportNoOverloadImplementation]
        @overload
        def __getitem__(self, key: Literal[1] | Literal["Displacement"]) -> NodeSocketVector: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    outputs: NodeOutputs  # pyright: ignore[reportIncompatibleVariableOverride]


class NODE_MT_category_shader_pcb2blender(bpy.types.Menu):
    bl_idname = "NODE_MT_category_shader_pcb2blender"
    bl_label = "Pcb2Blender"

    def draw(self, context: bpy.types.Context):
        layout = self.layout

        node_add_menu.add_node_type(layout, "ShaderNodeBsdfPcbSurfaceFinish")
        node_add_menu.add_node_type(layout, "ShaderNodeBsdfPcbSolderMask")
        node_add_menu.add_node_type(layout, "ShaderNodeBsdfPcbSilkscreen")
        node_add_menu.add_node_type(layout, "ShaderNodeBsdfPcbBoardEdge")
        node_add_menu.add_node_type(layout, "ShaderNodeBsdfSolder")
        node_add_menu.add_node_type(layout, "ShaderNodePcbShader")

        node_add_menu.draw_assets_for_catalog(layout, self.bl_label)


def menu_draw(self: bpy.types.Menu, context: bpy.types.Context):
    assert self.layout
    self.layout.separator()
    self.layout.menu("NODE_MT_category_shader_pcb2blender")


classes = (
    ShaderNodeBsdfPcbSurfaceFinish,
    ShaderNodeBsdfPcbSolderMask,
    ShaderNodeBsdfPcbSilkscreen,
    ShaderNodeBsdfPcbBoardEdge,
    ShaderNodeBsdfSolder,
    ShaderNodePcbShader,
    NODE_MT_category_shader_pcb2blender,
)


def register():
    register_mat4cad()

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.NODE_MT_shader_node_add_all.append(menu_draw)


def unregister():
    bpy.types.NODE_MT_shader_node_add_all.remove(menu_draw)

    for cls in classes:
        bpy.utils.unregister_class(cls)

    unregister_mat4cad()
