import importlib

MODULE_NAMES = ("importer", "materials", "solder_joints")
_modules = []


def register():
    _modules.clear()
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
