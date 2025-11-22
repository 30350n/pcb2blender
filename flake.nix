{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

        self.submodules = true;
    };

    outputs = {nixpkgs, ...}: let
        system = "x86_64-linux";
        pkgs = nixpkgs.legacyPackages.${system};
        blender = pkgs.blender.overrideAttrs (finalAttrs: prevAttrs: {
            pythonPath =
                prevAttrs.pythonPath
                ++ (with pkgs.python311Packages; [
                    debugpy
                    flask
                    pytest
                    requests
                    werkzeug
                ]);
        });
    in {
        devShells.${system}.default = pkgs.mkShell {
            packages = with pkgs; [
                python3
                uv
            ];

            shellHook = let
                kicadPythonVersion = pkgs.lib.versions.majorMinor pkgs.python3.version;
                kicadPythonSitePackages = "lib/python${kicadPythonVersion}/site-packages";
            in ''
                mkdir -p .vscode/
                [[ -f .vscode/settings.json ]] || echo "{}" > .vscode/settings.json

                prev_settings=$(mktemp)
                cp .vscode/settings.json "$prev_settings"

                ${pkgs.jq}/bin/jq '
                    .["blender.executables"] = [
                        {
                            "name": "Blender ${blender.version}",
                            "path": "${blender}/bin/blender",
                            "isDefault": false
                        }
                    ]
                    |
                    .["basedpyright.analysis.extraPaths"] = [
                        "${pkgs.kicad-small.src}/${kicadPythonSitePackages}",
                        "${pkgs.python3Packages.wxpython}/${kicadPythonSitePackages}"
                    ]
                ' "$prev_settings" > .vscode/settings.json
            '';

            BLENDER_EXECUTABLE = "${blender}/bin/blender";
        };
    };
}
