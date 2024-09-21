from pathlib import Path

import pcbnew, wx

from .export import export_pcb3d, get_boarddefs
from .ui import SettingsDialog

class Pcb2BlenderExporter(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Export to Blender (.pcb3d)"
        self.category = "Export"
        self.show_toolbar_button = True
        self.icon_file_name = (
            Path(__file__).parent / "images" / "blender_icon_32x32.png").as_posix()
        self.description = "Export 3D Model to Blender."

    def Run(self):
        board = pcbnew.GetBoard()
        boarddefs, ignored = get_boarddefs(board)
        with SettingsDialog(None, boarddefs, ignored) as dialog:
            if dialog.ShowModal() == wx.OK:
                export_pcb3d(dialog.file_picker.GetPath(), boarddefs)

Pcb2BlenderExporter().register()
