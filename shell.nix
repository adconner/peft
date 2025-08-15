let 
  # https://status.nixos.org/ nixos-unstable-small 8/5/25
  # https://hydra.nix-community.org/job/nixpkgs/cuda/python3Packages.torch.x86_64-linux
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ebe222ec7ef9de52478f76cba3f0324c1d1119f") { 
    # config.allowUnfree = true;
    # config.cudaSupport = true;
  };
  pythonPackages = pkgs.python3Packages;
  # jaxonnxruntime = let
  #   pname = "jaxonnxruntime";
  #   rev = "a1650c26b393e58b780371c030f9290247cf6cc5";
  # in pythonPackages.buildPythonPackage {
  #   inherit pname;
  #   version = rev;
  #   src = pkgs.fetchFromGitHub {
  #     owner = "google";
  #     repo = pname;
  #     inherit rev;
  #     sha256 = "sha256-aLl5RM63pp4R+0edL8NwmEcbD0j1OfGzniYw8jIpyK4=";
  #   };
  #   pyproject = true;
  #   build-system = with pythonPackages; [ setuptools-scm ];
  #   propagatedBuildInputs = with pythonPackages; [ jaxtyping chex onnx ];
  # };
  torch2jax = let
    pname = "torch2jax";
    rev = "814b35ff9fa1b6be93bf07a14a4dd6f2bf50685f";
  in pythonPackages.buildPythonPackage {
    inherit pname;
    version = rev;
    src = ../torch2jax;
    # src = pkgs.fetchFromGitHub {
    #   owner = "samuela";
    #   repo = pname;
    #   inherit rev;
    #   sha256 = "sha256-Ekk0Vh3f5ksxII04Wkd5YcLHP07zXeQqnjxZgGbWqWY=";
    # };
    pyproject = true;
    build-system = with pythonPackages; [ setuptools ];
    propagatedBuildInputs = with pythonPackages; [ torch jax ];
  };
in
pkgs.mkShell {
  buildInputs = with pythonPackages; [
    jax-cuda12-plugin
    jax
    optax
    equinox
    jaxtyping
    chex
    torch2jax
      
    datasets
    # evaluate
    # scikit-learn
      
    # accelerate
    transformers
    torch
      
    ipython
    # venvShellHook
  ];
  # venvDir = ".venv";
}
  
