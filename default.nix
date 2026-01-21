{pkgs ? import (fetchTarball "https://nixos.org/channels/nixos-unstable/nixexprs.tar.xz") {} }:
  pkgs.mkShell {
    buildInputs = with pkgs; [
      marimo
      cargo
      gdb
      cargo-watch
      rust-analyzer
  ] ++ (with pkgs.python3.pkgs; [
      black # auto formatting
      flake8 # annoying "good practice" annotations
      mypy # static typing
      pkgs.ruff # language server ("linting")

      numpy
      matplotlib
      seaborn
      tqdm
      scipy
      pandas
      pyarrow
      #   (ortools.overrideAttrs (final: prev: {
      #    buildInputs = prev.buildInputs ++ [ pkgs.scipopt-scip ];
      #    propagatedBuildInputs = prev.propagatedBuildInputs ++ [ pkgs.scipopt-scip ];
      #    cmakeFlags = prev.cmakeFlags ++ [ (lib.cmakeBool "USE_SCIP" true) ];
      #    NIX_CFLAGS_COMPILE = [ "-Wno-format-security" ];
      #  }))
    ]);
  }
