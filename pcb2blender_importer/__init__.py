bl_info = {
	"name": "pcb2blender importer",
	"description": "Enables Blender to import .pcb3d files, exported from KiCad.",
	"author": "Bobbe",
	"version": (1, 0, 0),
	"blender": (3, 0, 0),
	"location": "File > Import",
	"category": "Import-Export",
	"support": "COMMUNITY",
	"wiki_url": "", # todo: insert github url
}

import bpy
import importlib

module_names = ("import",)
modules = []

for module_name in module_names:
	if module_name in locals():
		modules.append(importlib.reload(locals()[module_name]))
	else:
		modules.append(importlib.import_module("." + module_name, package=__package__))

def register():
	for module in modules:
		module.register()

def unregister():
	for module in modules:
		module.unregister()
