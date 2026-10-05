# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.packages.dealii.package import Dealii as BuiltinDealii

from spack.package import *


class Dealii(BuiltinDealii):
    """deal.II: the builtin recipe plus the two fixes a deal.II used outside
    Spack needs.

    1. A project built against deal.II takes its compiler from
       deal.IIConfig.cmake. The builtin recipe leaves Spack's compiler wrapper
       there, and the wrapper refuses to run outside a Spack build ("Spack
       compiler must be run from Spack! Input 'SPACK_COMPILER_WRAPPER_PATH'
       is missing."). The wrapper paths are replaced with the real compilers
       after install.
    2. The builtin recipe passes DEAL_II_WITH_OPENCASCADE only with
       +opencascade, so with ~opencascade deal.II's own detection can take a
       system OpenCASCADE (it took Ubuntu's 7.3.0 from /usr). The switch is
       passed either way.
    """

    filter_compiler_wrappers(
        "deal.IIConfig.cmake", relative_root=join_path("lib", "cmake", "deal.II")
    )

    def cmake_args(self):
        args = super().cmake_args()
        if self.spec.satisfies("~opencascade"):
            args.append(self.define("DEAL_II_WITH_OPENCASCADE", False))
        return args
