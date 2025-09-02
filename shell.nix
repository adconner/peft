let 
  # # https://status.nixos.org/ nixos-unstable-small 8/5/25
  # # https://hydra.nix-community.org/job/nixpkgs/cuda/python3Packages.torch.x86_64-linux
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ebe222ec7ef9de52478f76cba3f0324c1d1119f") { 
  #   config.allowUnfree = true;
  #   config.cudaSupport = true;
  # };
  pkgs = import <nixpkgs> {
    overlays = [
      (self: super: {
        pythonPackagesExtensions = super.pythonPackagesExtensions ++ [
          (python-self: python-super: {
            flax = python-super.flax.overridePythonAttrs (_: { doCheck = false; });
            torch = python-super.torch.override { cudaSupport = true; };

            # aqtp = python-self.buildPythonPackage rec {
            #   pname = "aqtp";
            #   version = "0.9.0";
            #   src = pkgs.fetchFromGitHub {
            #     owner = "google";
            #     repo = "aqt";
            #     rev = "e781c13f63d359027922b31e7e227cca4fab4dc8";
            #     sha256 = "sha256-FvqMD6OLvcK1umCRW8skAMjUHw5koy/a7xIn76mG9wU=";
            #   };
            #   pyproject = true;
            #   build-system = with python-self; [ setuptools ];
            #   propagatedBuildInputs = with python-self; [ jax jaxlib absl-py flax ];
            #   doCheck = false;
            #   patchPhase = "cat setup.py | sed 's/0.1.0/${version}/' > tmp && mv tmp setup.py";
            # };
              
            # haliax = let
            # in python-self.buildPythonPackage rec {
            #   pname = "haliax";
            #   version = "";
            #   src = pkgs.fetchFromGitHub {
            #     owner = "stanford-crfm";
            #     repo = pname;
            #     rev = "30066d0a1eea3d282580a5a5fef815211d005abb";
            #     sha256 = "sha256-/gzDORDv3GWWxMxQO7OE7C+9kn0v2dE+ojvdvhy/pKk=";
            #   };
            #   pyproject = true;
            #   build-system = with python-self; [ hatchling ];
            #   propagatedBuildInputs = with python-self; [ equinox jaxtyping jmp safetensors aqtp ];
            # };
            
            torch2jax = pythonPackages.buildPythonPackage rec {
              pname = "torch2jax";
              version = "0.1.0";
              src = pkgs.fetchFromGitHub {
                owner = "samuela";
                repo = pname;
                rev = "505769d32fc20b95e23f53c36dd320db31066282";
                sha256 = "sha256-+1AICchWfmRou93iMq53Ai+1KkxYS8rBU14A9pjt3a0=";
              };
              pyproject = true;
              build-system = with pythonPackages; [ setuptools ];
              propagatedBuildInputs = with pythonPackages; [ torch jax ];
            };
            
          })
        ];
      })
    ];
  };
  pythonPackages = pkgs.python3Packages;
in
pkgs.mkShell {
  buildInputs = with pythonPackages; [
    jax-cuda12-plugin
    jax
    optax
    equinox
    jaxtyping
    chex
    flax
    torch2jax
      
    datasets
    # evaluate
    # scikit-learn
      
    accelerate
    transformers
    torch
      
    ipython
    # venvShellHook
  ];
  # venvDir = ".venv";
}
