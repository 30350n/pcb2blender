from itertools import chain, product
from pathlib import Path
from tempfile import gettempdir

import pytest

import bpy

TEST_FILEPATHS = list((Path(__file__).parent / "test_pcbs").resolve().glob("**/*.pcb3d"))

KWARGS_TEST_PERMUTATIONS = {
    "import_components": (True, False),
    "center_boards": (True, False),
    "cut_boards": (True, False),
    "stack_boards": (True, False),
}

KWARGS_TEST_ONCE = {
    "add_solder_joints": ("NONE", "SMART", "ALL"),
    "merge_materials": (True, False),
    "enhance_materials": (False, True),
    "texture_dpi": (508, 1016),
}

TEST_PARAMETERS = chain(
    (
        {key: value for key, value in zip(KWARGS_TEST_PERMUTATIONS.keys(), permutation)}
        | {key: values[0] for key, values in KWARGS_TEST_ONCE.items()}
        for permutation in product(*KWARGS_TEST_PERMUTATIONS.values())
    ),
    (
        {key: values[0] for key, values in (KWARGS_TEST_PERMUTATIONS | KWARGS_TEST_ONCE).items()}
        | {key: value}
        for key, values in KWARGS_TEST_ONCE.items()
        for value in values[1:]
    ),
)


@pytest.mark.parametrize("path", TEST_FILEPATHS)
@pytest.mark.filterwarnings("ignore:.*U.*mode is deprecated:DeprecationWarning")
def test_importer(capsys, path):
    bpy.ops.wm.read_homefile(use_empty=True)

    result = bpy.ops.pcb2blender.import_pcb3d(filepath=str(path))

    if error := capsys.readouterr().err:
        raise Exception(error)
    assert result == {"FINISHED"}
    assert len(bpy.context.view_layer.objects) > 0
    assert bpy.context.object is not None


@pytest.mark.parametrize("kwargs", TEST_PARAMETERS)
@pytest.mark.filterwarnings("ignore:.*U.*mode is deprecated:DeprecationWarning")
def test_importer_parameters(capsys, kwargs):
    bpy.ops.wm.read_homefile(use_empty=True)

    result = bpy.ops.pcb2blender.import_pcb3d(filepath=str(TEST_FILEPATHS[0]), **kwargs)

    if error := capsys.readouterr().err:
        raise Exception(error)
    assert result == {"FINISHED"}
    assert len(bpy.context.view_layer.objects) > 0
    assert bpy.context.object is not None


def test_load_file(capsys):
    test_path = str(Path(gettempdir()) / "pcb2blender_test.blend")

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.pcb2blender.import_pcb3d(filepath=str(TEST_FILEPATHS[0]))
    bpy.ops.wm.save_mainfile(filepath=test_path)

    assert not has_undefined_nodes()
    if error := capsys.readouterr().err:
        raise Exception(error)

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=test_path)

    assert not has_undefined_nodes()
    if error := capsys.readouterr().err:
        raise Exception(error)


def has_undefined_nodes():
    node_groups = (group for group in bpy.data.node_groups if group.type == "SHADER")
    return "NodeUndefined" in (node.bl_idname for group in node_groups for node in group.nodes)
