[![blender](https://img.shields.io/badge/Blender-4.2_LTS-orange)](https://www.blender.org/)
[![blender](https://img.shields.io/badge/Blender-4.3-orange)](https://www.blender.org/)
[![kicad](https://img.shields.io/badge/KiCad-8.0-blue)](https://www.kicad.org/)
[![gplv3](https://img.shields.io/badge/License-GPLv3-lightgrey)](https://www.gnu.org/licenses/gpl-3.0.txt)

<img src="images/header.jpg"/>

The pcb2blender workflow lets you create professionally looking product renders of all your
KiCad projects in minutes! Simply export your board as a `.pcb3d` file in KiCad, import it into
Blender and start creating!

It lets you focus all your time on actually composing and lighting an interesting scene,
skipping all the boring cleanup work that would be required without it.

Its quick export/import times also make it viable to create renders of WIP boards or to update
them, after last minute board revisions.

<img src="images/e201_soldered.jpg"/>

## Usage

1. Hit the `Export to Blender (.pcb3d)` button in pcbnew.
2. Import the file in Blender via `File -> Import -> PCB (.pcb3d)`
3. Start creating awesome renders!

## Installation

### Exporter (KiCad)

- `Tools -> Plugin and Content Manager -> Plugins -> pcb2blender`

- (manual) Download the `pcb2blender_exporter_<version>.zip` from the
  [latest release](https://github.com/30350n/pcb2blender/releases/latest),
  unpack the `plugins` directory from it into your
  [KiCad Plugin directory](https://dev-docs.kicad.org/en/apis-and-binding/pcbnew/) and rename it to
  `pcb2blender_exporter`.

### Importer (Blender)

- `Edit -> Preferences -> Get Extensions -> PCB 3D Importer`

- (manual) Download the `pcb2blender_importer_<version>.zip` from the
  [latest release](https://github.com/30350n/pcb2blender/releases/latest),
  install it in Blender via<br>
  `Edit -> Preferences -> Add-ons -> Install from Disk... (Top Right Menu)`<br> and enable it.
  (You need to install the actual zip file, don't unpack it!)

## Other Projects

- The [protorack-kicad](https://github.com/30350n/protorack-kicad) KiCad library, contains
  all the custom symbols and footprints I use for eurorack module development.

- The [svg2blender](https://github.com/30350n/svg2blender) workflow enables you to export
  2D graphical designs from [Inkscape](https://inkscape.org/) to Blender. It's mainly intended
  for use with front panel designs, but could be used for other things as well.

## Credits

- The name of this project is inspired by the awesome
  [svg2shenzhen](https://github.com/badgeek/svg2shenzhen) Inkscape extension by
  [badgeek](https://github.com/badgeek).

- The PCB Shader node setup this addon comes with is inspired by the
  [stylized-blender-setup](https://github.com/PCB-Arts/stylized-blender-setup)
  repository by [PCB-Arts](https://www.pcb-arts.com).

## License

- This project is licensed under
  [GPLv3](https://github.com/30350n/pcb2blender/blob/master/LICENSE).
