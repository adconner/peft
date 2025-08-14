let 
  # https://status.nixos.org/ nixos-unstable-small 8/5/25
  # https://hydra.nix-community.org/job/nixpkgs/cuda/python3Packages.torch.x86_64-linux
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ebe222ec7ef9de52478f76cba3f0324c1d1119f") { 
    # config.allowUnfree = true;
    # config.cudaSupport = true;
  };
  pythonPackages = pkgs.python3Packages;
  jaxonnxruntime = let
    pname = "jaxonnxruntime";
    rev = "a1650c26b393e58b780371c030f9290247cf6cc5";
  in pythonPackages.buildPythonPackage {
    inherit pname;
    version = rev;
    src = pkgs.fetchFromGitHub {
      owner = "google";
      repo = pname;
      inherit rev;
      sha256 = "sha256-aLl5RM63pp4R+0edL8NwmEcbD0j1OfGzniYw8jIpyK4=";
    };
    pyproject = true;
    build-system = with pythonPackages; [ setuptools-scm ];
    propagatedBuildInputs = with pythonPackages; [ jaxtyping chex onnx ];
  };
in
pkgs.mkShell {
  buildInputs = with pythonPackages; [
    jax-cuda12-plugin
    jax
    optax
    optimistix
    equinox
    jaxtyping
    chex
    jaxonnxruntime
      
    datasets
    # evaluate
    # scikit-learn
      
    # accelerate
    # transformers
    # torch

    # jax2onnx
      
    ipython
    # venvShellHook
  ];
  # venvDir = ".venv";
}
  
