{
  description = "ratio-manager: TorrentLeech ratio building across home and rented seedboxes";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      cargo = builtins.fromTOML (builtins.readFile ./Cargo.toml);
      version = cargo.package.version;

      ratio-manager = pkgs.rustPlatform.buildRustPackage {
        pname = "ratio-manager";
        inherit version;
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type: !(pkgs.lib.hasInfix "/target" path) && !(pkgs.lib.hasInfix "/dev/" path);
        };
        cargoLock.lockFile = ./Cargo.lock;
        meta.mainProgram = "ratio-manager";
      };

      # The runtime user (uid 1000, the fleet's) needs a passwd entry: ssh
      # refuses to run without one.
      nss = pkgs.dockerTools.fakeNss.override {
        extraPasswdLines = [ "ratio:x:1000:1000:ratio-manager:/data:/bin/false" ];
        extraGroupLines = [ "ratio:x:1000:" ];
      };

      image = pkgs.dockerTools.buildLayeredImage {
        name = "ghcr.io/idanreed/ratio-manager";
        tag = version;
        contents = [
          ratio-manager
          nss
          pkgs.cacert
          pkgs.tzdata
          # Host pressure (PSI) on a box with shell access.
          pkgs.openssh
          pkgs.sshpass
        ];
        extraCommands = ''
          mkdir -m 1777 tmp
        '';
        config = {
          Entrypoint = [ "${ratio-manager}/bin/ratio-manager" ];
          Cmd = [ "serve" ];
          User = "1000:1000";
          Env = [
            "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            "HOME=/data"
            "TZ=UTC"
          ];
          ExposedPorts."8080/tcp" = { };
          Labels."org.opencontainers.image.source" = "https://github.com/IdanReed/ratio-manager";
          Labels."org.opencontainers.image.description" = "TorrentLeech ratio building across home and rented seedboxes";
        };
      };
    in
    {
      packages.${system} = {
        default = ratio-manager;
        inherit image;
      };
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cargo
          pkgs.rustc
          pkgs.clippy
          pkgs.rustfmt
          pkgs.qbittorrent-nox
          pkgs.python3
        ];
      };
    };
}
