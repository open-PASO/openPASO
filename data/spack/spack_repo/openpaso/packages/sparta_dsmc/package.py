# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import datetime as dt

from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class SpartaDsmc(CMakePackage):
    """SPARTA is a parallel Direct Simulation Monte Carlo (DSMC) code for
    performing simulations of low-density gases in 2d or 3d domains,
    developed by Sandia National Laboratories."""

    homepage = "https://sparta.github.io"
    url = "https://github.com/sparta/sparta/archive/27Aug2026.tar.gz"
    git = "https://github.com/sparta/sparta.git"

    license("GPL-2.0-only")

    version(
        "2026.08.27", sha256="9f02e90d46c4ce44dca761d1ebe630d529d569829316b5875f4d2519ff38d1c0"
    )

    variant("mpi", default=True, description="Build with MPI support")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("cmake@3.16:", type="build")
    depends_on("mpi", when="+mpi")

    root_cmakelists_dir = "cmake"

    def url_for_version(self, version):
        vdate = dt.datetime.strptime(str(version), "%Y.%m.%d").replace(tzinfo=dt.timezone.utc)
        tag = vdate.strftime("%d%b%Y").lstrip("0")
        return f"https://github.com/sparta/sparta/archive/{tag}.tar.gz"

    def cmake_args(self):
        args = [
            self.define("BUILD_MPI", self.spec.satisfies("+mpi")),
            self.define("PKG_MPI_STUBS", self.spec.satisfies("~mpi")),
            self.define("SPARTA_MACHINE", "mpi" if self.spec.satisfies("+mpi") else "serial"),
        ]

        if self.spec.satisfies("+mpi"):
            args.append(self.define("CMAKE_C_COMPILER", self.spec["mpi"].mpicc))
            args.append(self.define("CMAKE_CXX_COMPILER", self.spec["mpi"].mpicxx))

        return args

    @run_after("install")
    def install_data_and_examples(self):
        """Input decks reference species/VSS/reaction/surface files shipped
        in data/ and surface files shipped alongside the examples/ input
        decks; ship both so installed decks can be run standalone."""
        share = join_path(self.prefix.share, "sparta")
        install_tree(join_path(self.stage.source_path, "data"), join_path(share, "data"))
        install_tree(
            join_path(self.stage.source_path, "examples"), join_path(share, "examples")
        )
