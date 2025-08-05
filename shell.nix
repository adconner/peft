let 
  # https://status.nixos.org/ nixos-unstable-small 8/5/25
  # https://hydra.nix-community.org/job/nixpkgs/cuda/python3Packages.torch.x86_64-linux
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ebe222ec7ef9de52478f76cba3f0324c1d1119f") { 
    # config.allowUnfree = true;
    # config.cudaSupport = true;
  };
in
pkgs.mkShell {
  buildInputs = with pkgs.python312Packages; [
    flax
    jax-cuda12-plugin
    optax
    optimistix
    equinox
      
    datasets
    evaluate
    scikit-learn
    accelerate
    ipython
    transformers
      
    # torch
      
    # venvShellHook
  ];
  # venvDir = ".venv";
}
  
