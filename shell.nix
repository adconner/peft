let 
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.05") {};
  pkgs = import <nixpkgs> {}; 
in
pkgs.mkShell {
  buildInputs = with pkgs.python312Packages; [
    # tensorflow
    # ipython
    # jax
    jax-cuda12-plugin
    transformers
    datasets
    flax
    pytorch
    evaluate
    scikit-learn
    accelerate
    # optax
    # optimistix
    # venvShellHook
    ipython
    # pytest
    # equinox
  ] ++ (with pkgs; [
  ]);
  # venvDir = ".venv";
}
  
