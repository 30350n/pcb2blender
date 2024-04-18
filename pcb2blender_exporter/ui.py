from pathlib import Path

import wx

class SettingsDialog(wx.Dialog):
    def __init__(self, parent, boarddefs, ignored):
        wx.Dialog.__init__(self, parent, title="Export to Blender")

        panel = self.init_panel(boarddefs, ignored)
        sizer = wx.BoxSizer()
        sizer.Add(panel)
        self.SetSizerAndFit(sizer)
        self.SetMinSize((0, 1000))

        self.Center()
        self.Show()

    def on_export(self, event):
        path = Path(self.file_picker.GetPath())
        if path.parent.exists():
            self.EndModal(wx.OK)
        else:
            wx.MessageBox(
                f"Invalid path \"{path.parent}\"!", caption="Error",
                style=wx.CENTER | wx.ICON_ERROR | wx.OK
            )

    def init_panel(self, boarddefs, ignored):
        panel = wx.Panel(self)

        rows = wx.BoxSizer(orient=wx.VERTICAL)
        settings = wx.StaticBoxSizer(wx.StaticBox(panel, label="Settings"), orient=wx.VERTICAL)
        column = wx.BoxSizer()

        text_export_as = wx.StaticText(panel, label="Export as")
        column.Add(text_export_as, flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        self.file_picker = wx.FilePickerCtrl(
            panel, message="Export as",
            wildcard="PCB 3D Model (.pcb3d)|*.pcb3d",
            style=wx.FLP_SAVE | wx.FLP_USE_TEXTCTRL | wx.FLP_OVERWRITE_PROMPT,
            size=(300, 25)
        )
        column.Add(self.file_picker, proportion=1, flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        settings.Add(column, flag=wx.EXPAND | wx.ALL, border=5)
        rows.Add(settings, flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=5)

        info = wx.StaticBoxSizer(wx.StaticBox(panel, label="Info"), orient=wx.VERTICAL)

        n_boards = max(1, len(boarddefs))
        plural = "" if n_boards == 1 else "s"
        text_detected = wx.StaticText(panel, label=f"Detected {n_boards} Board{plural}.")
        info.Add(text_detected, flag=wx.ALL, border=5)

        for name, boarddef in sorted(boarddefs.items()):
            label = f"PCB {name}"\
                f" ({boarddef.bounds[2]:.2f}x{boarddef.bounds[3]:.2f}mm)"
            if boarddef.stacked_boards:
                label += " with "
                for stacked in boarddef.stacked_boards:
                    label += "front panel" if stacked.name == "FPNL" else stacked.name
                    stack_str = ", ".join(f"{f:.2f}" for f in stacked.offset)
                    label += f" stacked at ({stack_str}), "
                label = label[:-2] + "."

            info.Add(wx.StaticText(panel, label=label), flag=wx.ALL, border=5)

        rows.Add(info, flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=5)

        if ignored:
            warning = wx.StaticBoxSizer(
                wx.StaticBox(panel, label="Warning (failed to parse some identifiers)"),
                orient=wx.VERTICAL
            )

            for name in ignored:
                warning.Add(wx.StaticText(panel, label="    " + name), flag=wx.ALL, border=5)

            rows.Add(warning, flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=5)

        hint = wx.StaticBoxSizer(wx.StaticBox(panel, label="Hint"), orient=wx.VERTICAL)
        boarddef_hint = ""\
            "To define a board, specify its bounds by placing a Text Item with the text "\
            "PCB3D_TL_<boardname> at its top left corner and one with "\
            "PCB3D_BR_<boardname> at its bottom right corner.\n\n"\
            "To stack a board A to another board B, add a Text Item with the text "\
            "PCB3D_STACK_<boardA>_ONTO_<boardB>_<zoffset>\n"\
            "at the location (relative to the top left corner of board B), "\
            "where you want the top left corner of A to be.\n"\
            "(zoffset is given in mm, 10.0 is a good default for 2.54mm headers and sockets)"
        boarddef_hint_text = wx.StaticText(panel, label=boarddef_hint)
        boarddef_hint_text.Wrap(400)
        hint.Add(boarddef_hint_text, flag=wx.ALL, border=5)
        rows.Add(hint, flag=wx.TOP | wx.LEFT | wx.RIGHT, border=5)

        buttons = wx.BoxSizer()
        button_cancel = wx.Button(panel, id=wx.ID_CANCEL, label="Cancel", size=(85, 26))
        buttons.Add(button_cancel, flag=wx.ALL | wx.ALIGN_CENTER, border=5)
        button_export = wx.Button(panel, id=wx.ID_OK, label="Export", size=(85, 26))
        button_export.Bind(wx.EVT_BUTTON, self.on_export)
        buttons.Add(button_export, flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        rows.Add(buttons, flag=wx.ALL | wx.ALIGN_RIGHT, border=5)

        panel.SetSizer(rows)

        return panel
