# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class _4c(CMakePackage):
    """4C ("Comprehensive Computational Community Code") is a parallel
    multiphysics research code to analyze and solve a plethora of physical
    problems described by ordinary or partial differential equations."""

    homepage = "https://www.4c-multiphysics.org"
    url = "https://github.com/4C-multiphysics/4C/archive/refs/tags/v2026.3.0.tar.gz"
    git = "https://github.com/4C-multiphysics/4C.git"

    license("LGPL-3.0-or-later")

    version("2026.3.0", sha256="d8fa2ca8a3815f8050f6d7f1c428ed65c4e4fea83a7079beffe4d6e11eddfb84")

    variant(
        "build_type",
        default="Release",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo"),
    )
    # From 2026.3.0 on, beam interaction searches only with ArborX's bounding
    # volumes; without it a beam contact or beam-to-solid run stops with "The
    # struct 'Core::GeometricSearch::BoundingVolume' can only be used with
    # ArborX". 4C's own CMake default is OFF.
    variant("arborx", default=True, description="Build with ArborX (needed for beam interaction)")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on("cmake@3.30:", type="build")
    depends_on("patchelf", type="build")

    depends_on("mpi")
    depends_on("hdf5+mpi+hl")
    depends_on("boost+graph")
    depends_on("cln")
    depends_on("zlib-api")
    depends_on("blas")
    depends_on("lapack")

    # Trilinos configuration from dependencies/current/trilinos/install.sh
    depends_on(
        "trilinos+mpi+shared+explicit_template_instantiation"
        "+amesos+amesos2+aztec+belos+epetra+epetraext+exodus+ifpack+ifpack2+intrepid2"
        "+kokkos+ml+muelu+nox+sacado+shards+stratimikos+teko+thyra+tpetra+zoltan+zoltan2"
        "+mumps+suite-sparse+superlu-dist gotype=int cxxstd=17"
    )
    depends_on("trilinos@16.2.1", when="@2026.3.0")
    depends_on("superlu-dist@9.2.1", when="@2026.3.0")
    depends_on("suite-sparse@5.4.0", when="@2026.3.0")
    depends_on("parmetis")

    # C++20 support (see doc/documentation/src/installation/installation.rst)
    conflicts("%gcc@:12", msg="4C requires at least GCC 13")
    conflicts("%clang@:17", msg="4C requires at least Clang 18")

    # Header/source dependencies that 4C otherwise downloads via FetchContent
    # at configure time; pinned to the commits used by the release.
    with when("@2026.3.0"):
        resource(
            name="ryml",
            git="https://github.com/biojppm/rapidyaml.git",
            commit="47ec2fa184209687c20fd5bc05621e1cb1200311",  # v0.9.0
            submodules=True,
            placement="_deps/ryml",
        )
        resource(
            name="magic_enum",
            git="https://github.com/Neargye/magic_enum.git",
            commit="e046b69a3736d314fad813e159b1c192eaef92cd",  # v0.9.7
            placement="_deps/magic_enum",
        )
        resource(
            name="cli11",
            git="https://github.com/CLIUtils/CLI11.git",
            commit="bfffd37e1f804ca4fae1caae106935791696b6a9",  # v2.6.1
            placement="_deps/cli11",
        )
        resource(
            name="arborx",
            git="https://github.com/arborx/ArborX.git",
            commit="f9244ba03904cc518a54d99e9f87bb42dc9ecaf3",  # v2.0.1
            placement="_deps/arborx",
            when="+arborx",
        )

    # post_processor converts 4C's native output (.control, .result.*,
    # .mesh.*) to VTU. 4C marks it EXCLUDE_FROM_ALL and gives it no install
    # rule, so it is built as an extra target and copied into bin, beside 4C.
    build_targets = ["all", "post_processor"]

    @run_after("install")
    def install_post_processor(self):
        # The copy would keep the build tree's RPATH, whose first entries point
        # into Spack's stage under /tmp. 4C's CMakeLists.txt sets
        # CMAKE_BUILD_WITH_INSTALL_RPATH to FALSE itself, so a -D cannot change
        # that. The copy gets the RPATH CMake installed 4C with instead, kept an
        # RPATH (not a RUNPATH) like 4C's own.
        install(join_path(self.build_directory, "post_processor"), self.prefix.bin)
        patchelf = Executable(join_path(self.spec["patchelf"].prefix.bin, "patchelf"))
        rpath = patchelf("--print-rpath", join_path(self.prefix.bin, "4C"), output=str).strip()
        patchelf("--force-rpath", "--set-rpath", rpath, join_path(self.prefix.bin, "post_processor"))

    def cmake_args(self):
        spec = self.spec
        args = [
            self.define("FOUR_C_BUILD_SHARED_LIBS", True),
            self.define("FOUR_C_ENABLE_WARNINGS_AS_ERRORS", False),
            self.define("FOUR_C_ENABLE_NATIVE_OPTIMIZATIONS", False),
            self.define("FOUR_C_ENABLE_DOCUMENTATION", False),
            self.define("FOUR_C_ENABLE_METADATA_GENERATION", False),
            self.define("FOUR_C_ENABLE_PYTHON_BINDINGS", False),
            self.define("FOUR_C_WITH_PYTHON", False),
            self.define("FOUR_C_WITH_PYBIND11", False),
            self.define("FOUR_C_WITH_GOOGLETEST", False),
            self.define("FOUR_C_WITH_GOOGLE_BENCHMARK", False),
            self.define("FOUR_C_TRILINOS_ROOT", spec["trilinos"].prefix),
            self.define("FOUR_C_HDF5_ROOT", spec["hdf5"].prefix),
            self.define("FOUR_C_BOOST_ROOT", spec["boost"].prefix),
            self.define("FOUR_C_CLN_ROOT", spec["cln"].prefix),
            self.define("Kokkos_ROOT", spec["kokkos"].prefix),
            self.define("MPI_CXX_COMPILER", spec["mpi"].mpicxx),
            self.define("MPI_C_COMPILER", spec["mpi"].mpicc),
        ]

        # Spack's Trilinos is built from a release tarball without the
        # TrilinosRepoVersion.txt that 4C uses to identify the version, so
        # map it to 4C's internal version (dependencies/supported_version).
        if spec.satisfies("^trilinos@16.2.1"):
            args.append(self.define("FOUR_C_TRILINOS_INTERNAL_VERSION", "2026.1"))
        elif spec.satisfies("^trilinos@16.2.0"):
            args.append(self.define("FOUR_C_TRILINOS_INTERNAL_VERSION", "2025.6"))

        if spec.satisfies("@2026.3.0"):
            deps = join_path(self.stage.source_path, "_deps")
            args.extend(
                [
                    self.define("FETCHCONTENT_FULLY_DISCONNECTED", True),
                    self.define("FETCHCONTENT_SOURCE_DIR_RYML", join_path(deps, "ryml")),
                    self.define(
                        "FETCHCONTENT_SOURCE_DIR_MAGIC_ENUM", join_path(deps, "magic_enum")
                    ),
                    self.define("FETCHCONTENT_SOURCE_DIR_CLI11", join_path(deps, "cli11")),
                ]
            )
            if spec.satisfies("+arborx"):
                args.append(
                    self.define("FETCHCONTENT_SOURCE_DIR_ARBORX", join_path(deps, "arborx"))
                )
        args.append(self.define_from_variant("FOUR_C_WITH_ARBORX", "arborx"))
        return args
