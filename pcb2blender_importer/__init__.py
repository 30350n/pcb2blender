bl_info = {
    "name": "pcb2blender importer",
    "description": "Enables Blender to import .pcb3d files, exported from KiCad.",
    "author": "Bobbe",
    "version": (2, 3, 0),
    "blender": (3, 4, 0),
    "location": "File > Import",
    "category": "Import-Export",
    "support": "COMMUNITY",
    "doc_url": "https://github.com/30350n/pcb2blender",
    "tracker_url": "https://github.com/30350n/pcb2blender/issues",
}

from .blender_addon_utils import add_dependencies, register_modules_factory

deps = {
    "skia-python": "skia",
    "pillow": "PIL",
}
add_dependencies(deps, no_extra_deps=True)

modules = ("importer", "materials", "solder_joints")
register, unregister = register_modules_factory(modules)
