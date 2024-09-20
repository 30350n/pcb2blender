bl_info = {
    "name": "pcb2blender importer",
    "description": "Enables Blender to import .pcb3d files, exported from KiCad.",
    "author": "Bobbe",
    "version": (2, 12, 0),
    "blender": (4, 2, 0),
    "location": "File > Import",
    "category": "Import-Export",
    "support": "COMMUNITY",
    "doc_url": "https://github.com/30350n/pcb2blender",
    "tracker_url": "https://github.com/30350n/pcb2blender/issues",
}

import importlib

__version__ = "2.12"

MODULE_NAMES = ("importer", "materials", "solder_joints")
_modules = []

def register():
    for module_name in MODULE_NAMES:
        if module_name in locals():
            _modules.append(importlib.reload(locals()[module_name]))
        else:
            _modules.append(importlib.import_module(f".{module_name}", package=__package__))

    for module in _modules:
        module.register()

def unregister():
    for module in _modules:
        module.unregister()
