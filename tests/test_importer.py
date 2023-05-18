import bpy

from itertools import product, chain
from pathlib import Path
from tempfile import gettempdir

import pytest

test_filepaths = list((Path(__file__).parent / "test_pcbs").resolve().glob("**/*.pcb3d"))

kwargs_test_permutations = {
    "import_components": (True, False),
    "center_boards": (True, False),
    "cut_boards": (True, False),
    "stack_boards": (True, False),
}

kwargs_test_once = {
    "add_solder_joints": ("NONE", "SMART", "ALL"),
    "merge_materials": (True, False),
    "enhance_materials": (False, True),
    "texture_dpi": (508, 1016),
}

test_parameters = chain((
    {key: value for key, value in zip(kwargs_test_permutations.keys(), permutation)}
    | {key: values[0] for key, values in kwargs_test_once.items()}
    for permutation in product(*kwargs_test_permutations.values())
), (
    {key: values[0] for key, values in (kwargs_test_permutations | kwargs_test_once).items()}
    | {key: value}
    for key, values in kwargs_test_once.items()
    for value in values[1:]
))

@pytest.mark.parametrize("path", test_filepaths)
@pytest.mark.parametrize("kwargs", test_parameters)
@pytest.mark.filterwarnings("ignore:.*U.*mode is deprecated:DeprecationWarning")
def test_importer(path, kwargs):
    bpy.ops.wm.read_homefile(use_empty=True)
    result = bpy.ops.pcb2blender.import_pcb3d(filepath=str(path), **kwargs)
    assert result == {"FINISHED"}
    assert len(bpy.context.view_layer.objects) > 0
    assert bpy.context.object is not None

def test_load_file():
    test_path = str(Path(gettempdir()) / "pcb2blender_test.blend")

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.pcb2blender.import_pcb3d(filepath=str(test_filepaths[0]))
    bpy.ops.wm.save_mainfile(filepath=test_path)

    assert not has_undefined_nodes()

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=test_path)

    assert not has_undefined_nodes()

def has_undefined_nodes():
    node_groups = (group for group in bpy.data.node_groups if group.type == "SHADER")
    return "NodeUndefined" in (node.bl_idname for group in node_groups for node in group.nodes)
