"""
Comprehensive 4C Multiphysics knowledge catalogue.

Based on systematic reading of ALL 4C source code.
73 modules, 41 problem types, 120+ materials, 130+ conditions, 20+ cell types.

This is the single source of truth for 4C domain knowledge in openPASO.
"""

FOURC_KNOWLEDGE = {
    # ═══════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    "overview": {
        "description": "4C is a large-scale parallel C++20 multiphysics FEM code developed at TU Munich",
        "source": "$FOURC_ROOT/src/ (73 modules) — set FOURC_ROOT env var",
        "input_format": "YAML (.4C.yaml) with inline mesh or Exodus mesh references",
        "execution": "mpirun -np N $FOURC_BINARY input.4C.yaml (or just 4C if on PATH)",
        "output": "VTU via IO/RUNTIME VTK OUTPUT sections",
        "build": "CMake (cd build && cmake --build . -j$(nproc))",
        "modules": 73,
        "problem_types": 41,  # 4C 2026.1.0 has 40 (no Reduced_Lung_1D_Pipe_Flow)
        "material_models": "120+",
        "condition_types": "130+",
        "entrypoint_dispatch": {
            "source": "apps/global_full/4C_global_full_entrypoint_switch.cpp",
            "function": "entrypoint_switch()",
            "description": (
                "Authoritative Core::ProblemType -> solver-driver mapping. "
                "Every 4C input YAML's PROBLEM TYPE section selects ONE of "
                "these 41 problem types, written as the PROBLEMTYPE string key "
                "of Core::string_to_problem_type_map() (e.g. "
                "Fluid_Structure_Interaction -> fsi), not as the enum name "
                "that keys this table (porofluid_pressure_based is the one "
                "name that is both); a misspelled value, or an enum name such "
                "as fsi, stops input reading with 'Could not match this input' "
                "before entrypoint_switch() runs."
            ),
            "problem_types": {
                "structure": "caldyn_drt()",
                "polymernetwork": "caldyn_drt()",
                "fluid": "dyn_fluid_drt(restart)",
                "fluid_redmodels": "dyn_fluid_drt(restart)",
                "lubrication": "lubrication_dyn(restart)",
                "ehl": "ehl_dyn()",
                "scatra": "scatra_dyn(restart)",
                "cardiac_monodomain": "scatra_cardiac_monodomain_dyn(restart)",
                "sti": "sti_dyn(restart)",
                "fluid_xfem": "fluid_xfem_drt(problem)",
                "fluid_ale": "fluid_ale_drt(problem)",
                "fsi": "fsi_ale_drt(problem)",
                "fsi_redmodels": "fsi_ale_drt(problem)",
                "fsi_xfem": "xfsi_drt(problem)",
                "fpsi_xfem": "xfpsi_drt(problem)",
                "gas_fsi": "fs3i_dyn()",
                "biofilm_fsi": "fs3i_dyn()",
                "thermo_fsi": "fs3i_dyn()",
                "fps3i": "fs3i_dyn()",
                "fbi": "fsi_immersed_drt(problem)",
                "ale": "dyn_ale_drt()",
                "thermo": "thermo_dyn_drt()",
                "tsi": "tsi_dyn_drt()",
                "loma": "loma_dyn(restart)",
                "elch": "elch_dyn(restart)",
                "art_net": "dyn_art_net_drt()",
                "red_airways": "dyn_red_airways_drt()",
                "reduced_lung": "ReducedLung::reduced_lung_main()",
                "one_d_pipe_flow": "ReducedLung1dPipeFlow::main()",
                "poroelast": "poroelast_drt()",
                "poroscatra": "poro_scatra_drt()",
                "porofluid_pressure_based": "porofluid_pressure_based_dyn(restart)",
                "porofluid_pressure_based_elast": "porofluid_elast_dyn(restart)",
                "porofluid_pressure_based_elast_scatra": "porofluid_pressure_based_elast_scatra_dyn(restart)",
                "fpsi": "fpsi_drt()",
                "ssi": "ssi_drt()",
                "ssti": "ssti_drt()",
                "particle": "particle_drt()",
                "pasi": "pasi_dyn()",
                "level_set": "levelset_dyn(restart)",
                "np_support": "MultiScale::np_support_drt()",
            },
            "Signal": (
                "[Input] A mis-typed PROBLEMTYPE is rejected while the input "
                "is read, before entrypoint_switch() runs: 'Could not match "
                "this input', with the line \"[!] Candidate "
                "deprecated_selection 'PROBLEMTYPE' has wrong value, possible "
                "values: Ale|ArterialNetwork|...\" listing the accepted "
                "strings. Those are the string keys of "
                "Core::string_to_problem_type_map() (e.g. "
                "Fluid_Structure_Interaction, Thermo_Structure_Interaction, "
                "Poroelasticity), not the enum names that key this table "
                "(fsi, tsi, poroelast; porofluid_pressure_based is both): "
                "PROBLEMTYPE: \"fsi\" is rejected the same way. The default "
                "arm 'solution of unknown problemtype <X> requested' cannot "
                "be reached from input. (Measured on a "
                "development build before 4C 2026.2.0, on 4C 2026.2.0 and on "
                "4C 2026.3.0.)"
            ),
        },
        "cli_arguments": {
            "source": "apps/global_full/4C_global_full_main.cpp (parse_command_line) + apps/global_full/4C_global_full_io.cpp",
            "description": (
                "Command-line arguments for nested-group parallelism and "
                "I/O configuration. Parsed with CLI11 in parse_command_line() "
                "(4C_global_full_main.cpp), which then calls "
                "validate_argument_cross_compatibility() and "
                "assign_group_layout() (4C_global_full_io.cpp); "
                "setup_global_problem() receives the parsed arguments."
            ),
            "flags": {
                "--ngroup=N":           "Number of nested-parallelism groups (default 1).",
                "--glayout=N1,N2,...":  "Explicit per-group MPI rank counts. If omitted with --ngroup>1, ranks are split equally and FAIL if num_procs%n_groups!=0.",
                "--nptype=<type>":      "Nested parallelism type (mandatory when --ngroup>1).",
                "<input> <output>":     "Positional pair(s). Multiple pairs allowed when --nptype=separateInputFiles or nestedMultiscale.",
                "--parameters":         "Parameters-dump mode (skips io_pairs validation). When set, the cross-compat validator does NOT require <input> <output> count to match --ngroup.",
                "--nptype=diffgroup0|diffgroup1": "Diff mode; there is no --diffgroup flag (4C rejects it: 'The following argument was not expected'). Sets the diff group (default -1 = disabled) to 0 or 1 so that two separate (serial/parallel) 4C runs can compare vectors/matrices/results; another suffix stops with 'Only diffgroup0 and diffgroup1 are allowed.'",
                "--interactive":        "Interactive mode (default false).",
                "--restart=N":          "Restart step (default 0 = no restart).",
                "--restartfrom=<id>":   "Output identifier to restart from. With nested parallelism, can be specified per-group via repeated flag.",
            },
            "commandline_arguments_struct_defaults": {
                # Source: apps/global_full/4C_global_full_io.hpp:38 CommandlineArguments
                "n_groups":                       1,
                "parameters":                     "false (set by --parameters)",
                "group_layout":                   "empty (auto-split if n_groups>1)",
                "nptype":                         "no_nested_parallelism",
                "diffgroup":                      -1,
                "restart":                        0,
                "restart_file_identifier":        "''",
                "restart_per_group":              "empty",
                "restart_identifier_per_group":   "empty",
                "interactive":                    False,
                "io_pairs":                       "empty",
                "input_file_name":                "''",
                "output_file_identifier":         "''",
            },
            "nptype_enum_values": {
                # snake_case (C++ enum) -> CLI-string alias (camelCase)
                "no_nested_parallelism":        "noNestedParallelism (default if --nptype omitted)",
                "every_group_read_input_file":  "everyGroupReadInputFile — single input, output gets _group_<N> suffix",
                "separate_input_files":         "separateInputFiles — N input/output pairs required",
                "nested_multiscale":            "nestedMultiscale — N input/output pairs required",
                "diffgroup0":                   "Sets nptype=no_nested_parallelism + diffgroup=0; first of two paired runs whose vectors/matrices/results will be diff'd",
                "diffgroup1":                   "Sets nptype=no_nested_parallelism + diffgroup=1; second of paired runs. Suffixes other than '0' or '1' rejected by CLI11 ValidationError",
            },
            "restart_special_values": {
                "last_possible":  "Sentinel string accepted by --restart; passes -1 internally, meaning 'restart from the last available checkpoint'",
                "<a>,<b>,<c>":    "Comma-separated per-group restart steps. Only meaningful with --nptype=separateInputFiles (one entry per group)",
            },
            "legacy_cli_syntax": {
                "description": (
                    "adapt_legacy_cli_arguments rewrites old-style invocations before CLI11 sees them — "
                    "two compat sets:"),
                "single_dash_legacy_names":  ["ngroup", "glayout", "nptype"],
                "nodash_legacy_names":       ["restart", "restartfrom"],
                "explanation": (
                    "single_dash: `4C -ngroup 4 ...` rewritten to `4C --ngroup=4 ...`. "
                    "nodash: `4C restart=10 input.yaml output.pre` rewritten to `4C --restart=10 input.yaml output.pre`. "
                    "Lets old scripts keep working but is invisible in --help."),
            },
            "build_options_affecting_runtime": {
                "FOUR_C_ENABLE_FE_TRAPPING": (
                    "If defined at compile time, main.cpp calls "
                    "feenableexcept(FE_INVALID | FE_DIVBYZERO) — the OS "
                    "kills the process via SIGFPE on the first NaN or "
                    "division-by-zero (informative message). Useful for "
                    "debugging numerical drift; will crash production runs "
                    "that intentionally use NaN sentinels."),
                "FOUR_C_ENABLE_CORE_DUMP": (
                    "If defined: run() is called WITHOUT a try/catch around "
                    "Core::Exception, so any throw triggers a core dump "
                    "(for post-mortem in gdb). If NOT defined (default): "
                    "Core::Exception is caught, stack trace printed via "
                    "err.what_with_stacktrace(), and MPI_Abort(MPI_COMM_WORLD, "
                    "EXIT_FAILURE) is called."),
            },
            "cmake_build_config_options": {
                "description": (
                    "Configure-time CMake cache variables defined in "
                    "cmake/setup_global_options.cmake. All take the "
                    "FOUR_C_* prefix and are surfaced via "
                    "four_c_process_global_option."),
                "options": {
                    "FOUR_C_BUILD_SHARED_LIBS": (
                        "bool, default ON. Force-syncs the legacy CMake "
                        "BUILD_SHARED_LIBS via FORCE cache writes if "
                        "only the legacy var is set — emits a CMake "
                        "WARNING in that case. Both names point to "
                        "the same value."),
                    "FOUR_C_ENABLE_DEVELOPER_MODE": (
                        "bool, default OFF. Optimizes the setup for "
                        "iterative development cycles."),
                    "FOUR_C_ENABLE_WARNINGS_AS_ERRORS": (
                        "bool, default OFF. Adds -Werror to the "
                        "private compile interface when ON."),
                    "FOUR_C_ENABLE_NATIVE_OPTIMIZATIONS": (
                        "bool, default OFF. Adds -march=native; "
                        "incompatible with portable binaries / "
                        "container images run on heterogeneous "
                        "hardware."),
                    "FOUR_C_ENABLE_ADDRESS_SANITIZER": (
                        "bool, default OFF. Adds -fsanitize=address "
                        "to both compile and link. FATAL_ERRORs at "
                        "configure time if the compiler+linker "
                        "don't accept the flag."),
                    "FOUR_C_ENABLE_COVERAGE": (
                        "bool, default OFF. LLVM source-based "
                        "coverage: -fprofile-instr-generate + "
                        "-fcoverage-mapping + -Wl,--build-id=sha1. "
                        "FATAL_ERROR at configure time if compiler "
                        "doesn't support."),
                    "FOUR_C_ENABLE_CORE_DUMP": (
                        "bool, default OFF. See "
                        "build_options_affecting_runtime."),
                    "FOUR_C_ENABLE_FE_TRAPPING": (
                        "bool, DEFAULT ON. Adds -ftrapping-math to "
                        "the compile interface. FATAL_ERRORs at "
                        "configure time if the compiler does not "
                        "support -ftrapping-math (most GCC/Clang "
                        "do; some Intel ICX / older Clang versions "
                        "don't). When OFF, instead adds "
                        "-fno-trapping-math. See "
                        "build_options_affecting_runtime for the "
                        "runtime behavior."),
                    "FOUR_C_ENABLE_IWYU": (
                        "bool, default OFF. Enables include-what-"
                        "you-use linter. FATAL_ERROR if iwyu "
                        "binary not found; user can override via "
                        "FOUR_C_IWYU_EXECUTABLE CMake variable."),
                    "FOUR_C_ENABLE_PYTHON_BINDINGS": (
                        "bool, default OFF. Builds the py4C "
                        "pybind11 bindings. Gated by cmake/"
                        "setup_py4C.cmake: requires BOTH "
                        "FOUR_C_BUILD_SHARED_LIBS=ON AND "
                        "FOUR_C_WITH_PYBIND11=ON, and is "
                        "INCOMPATIBLE with "
                        "FOUR_C_ENABLE_ADDRESS_SANITIZER=ON — "
                        "each violation FATAL_ERRORs at "
                        "configure time. The Python package "
                        "name written into the build dir is "
                        "literally 'py4C' (set via "
                        "FOUR_C_PYTHON_BINDINGS_PROJECT_NAME); "
                        "pyproject.toml.in and __init__.py.in "
                        "are configure_file'd from "
                        "utilities/py4C/src/config/ into "
                        "${PROJECT_BINARY_DIR}/py4C/."),
                    "FOUR_C_WITH_PYBIND11": (
                        "bool. Toggles the project-level "
                        "pybind11 dependency. Hard precondition "
                        "for FOUR_C_ENABLE_PYTHON_BINDINGS."),
                    "FOUR_C_ENABLE_ASSERTIONS": (
                        "bool, default OFF — but FORCE-set to ON "
                        "when build type is DEBUG (line 252-255: "
                        "explicit FORCE cache write). Adds "
                        "-D_GLIBCXX_ASSERTIONS for libstdc++ "
                        "assertions in addition to 4C's own "
                        "assertions."),
                    "FOUR_C_ENABLE_METADATA_GENERATION": (
                        "bool, default ON. Invokes Python after "
                        "build to generate metadata; requires "
                        "Python on the build host."),
                    "FOUR_C_ENABLE_LINKER_DETECTION": (
                        "bool, default ON. Defined in "
                        "cmake/checks/01_detect_linkers.cmake. "
                        "Probes `ld.mold`, `ld.lld`, `ld.gold`, "
                        "`ld.bfd` in that preference order via "
                        "find_program + four_c_check_compiles "
                        "with -fuse-ld=<name>. First linker that "
                        "passes the link-test wins. Populates "
                        "cache vars FOUR_C_LINKER_PROGRAM_<name> "
                        "(absolute path to ld.<name>) and "
                        "FOUR_C_LINKER_FUNCTIONAL_<name> (bool) "
                        "for each candidate tried. When OFF, the "
                        "user's manually-supplied linker flags "
                        "are used unchanged. If no linker is "
                        "functional, FATAL_ERROR with text "
                        "'Failed to find any working linker. "
                        "Please check your compiler and any "
                        "manually added flags.' OpenMPI / Ubuntu "
                        "20.04 quirk: faster linkers can fail "
                        "with mpic++; cmake then retries each "
                        "linker with `-lopen-pal` added "
                        "(populating FOUR_C_LINKER_FUNCTIONAL_"
                        "WITH_OPEN_PAL_<name>) before falling "
                        "back to the next candidate."),
                    "FOUR_C_CXX_FLAGS": (
                        "string, default empty. Expert setting; "
                        "additional C++ compile flags appended at "
                        "the END of the compile interface (so they "
                        "DO override earlier defaults). "
                        "separate_arguments-split."),
                    "FOUR_C_CXX_LINKER_FLAGS": (
                        "string, default empty. Expert setting; "
                        "additional linker flags appended at the "
                        "end."),
                },
                "build_type_optimization_flags": {
                    "DEBUG":          "-O0 -g (+ forces ENABLE_ASSERTIONS=ON)",
                    "RELEASE":        "-O3 -funroll-loops",
                    "RELWITHDEBINFO": "-O3 -g -funroll-loops",
                },
                "Signal": (
                    "[Performance] FOUR_C_ENABLE_FE_TRAPPING defaults "
                    "to ON in setup_global_options.cmake:195. Compilers "
                    "that don't accept -ftrapping-math (some Intel "
                    "ICX builds, older Clang on certain platforms) "
                    "FATAL_ERROR at CMake configure time with "
                    "'Option FOUR_C_ENABLE_FE_TRAPPING is ON but the "
                    "compiler does not support this feature. "
                    "Specifically, the compiler does not support "
                    "-ftrapping-math, which is necessary to generate "
                    "code that can safely use the floating-point "
                    "trapping mechanism.' Users on such compilers "
                    "must explicitly cmake -DFOUR_C_ENABLE_FE_TRAPPING="
                    "OFF. Plus three other configure-time pitfalls: "
                    "(a) DEBUG build type silently FORCEs "
                    "FOUR_C_ENABLE_ASSERTIONS=ON even if the user "
                    "explicitly passed -DFOUR_C_ENABLE_ASSERTIONS=OFF "
                    "(lines 251-255: explicit FORCE cache write with "
                    "'Forced ON due to build type DEBUG' help text); "
                    "(b) RELEASE / RELWITHDEBINFO build types add "
                    "-O3 + -funroll-loops directly to "
                    "four_c_private_compile_interface BEFORE "
                    "user FOUR_C_CXX_FLAGS — but FOUR_C_CXX_FLAGS is "
                    "appended at the END so it wins; CMAKE_CXX_FLAGS "
                    "by contrast is added in FRONT and cannot "
                    "override (file's own comment at lines 240-242 "
                    "spells this out); "
                    "(c) BUILD_SHARED_LIBS → FOUR_C_BUILD_SHARED_LIBS "
                    "migration emits a CMake WARNING but does NOT "
                    "fail — users following older 4C docs that "
                    "reference BUILD_SHARED_LIBS get a warning, "
                    "their value is force-synced into the new name, "
                    "and the build proceeds. "
                    "(d) [Integration] FOUR_C_ENABLE_PYTHON_BINDINGS=ON "
                    "has three configure-time prerequisites checked "
                    "by cmake/setup_py4C.cmake (NOT by setup_global_"
                    "options.cmake — easy to miss when reading only "
                    "the option declaration): "
                    "(i) FOUR_C_BUILD_SHARED_LIBS must be ON "
                    "(FATAL_ERROR: '4C Python bindings require to "
                    "build 4C with shared libraries (FOUR_C_BUILD_"
                    "SHARED_LIBS).'); "
                    "(ii) FOUR_C_WITH_PYBIND11 must be ON "
                    "(FATAL_ERROR: '4C Python bindings require to "
                    "build 4C with pybind11 (FOUR_C_WITH_PYBIND11).'); "
                    "(iii) FOUR_C_ENABLE_ADDRESS_SANITIZER must be "
                    "OFF (FATAL_ERROR: '4C Python bindings are "
                    "currently not compatible with an address "
                    "sanitizer build. Either set FOUR_C_ENABLE_"
                    "ADDRESS_SANITIZER=OFF or FOUR_C_ENABLE_PYTHON_"
                    "BINDINGS=OFF.'). The bindings build outputs a "
                    "pip-installable package at "
                    "${PROJECT_BINARY_DIR}/py4C/, with pyproject.toml "
                    "and __init__.py generated from "
                    "utilities/py4C/src/config/*.in templates. "
                    "(File walk cmake/setup_global_options.cmake + "
                    "cmake/setup_py4C.cmake 2026-06-03.) "
                    "(e) [Performance] When FOUR_C_ENABLE_LINKER_"
                    "DETECTION=ON (default), cmake/checks/"
                    "01_detect_linkers.cmake probes linkers in the "
                    "literal preference order mold > lld > gold > "
                    "bfd; first one that passes "
                    "four_c_check_compiles with "
                    "-fuse-ld=<name> wins. Source comment line "
                    "47-49 documents an OpenMPI / Ubuntu 20.04 "
                    "mpic++ wrapper bug: faster linkers can fail "
                    "with a missing-symbol error from "
                    "libopen-pal — cmake retries each linker with "
                    "`-lopen-pal` added (populating "
                    "FOUR_C_LINKER_FUNCTIONAL_WITH_OPEN_PAL_<name>) "
                    "before falling back to the next candidate. "
                    "To FORCE a specific linker, set "
                    "FOUR_C_ENABLE_LINKER_DETECTION=OFF and pass "
                    "-fuse-ld=<name> via FOUR_C_CXX_LINKER_FLAGS — "
                    "the detection block is skipped entirely. "
                    "When all four candidates fail, configure "
                    "aborts with FATAL_ERROR 'Failed to find any "
                    "working linker. Please check your compiler "
                    "and any manually added flags.' (File walk "
                    "cmake/checks/01_detect_linkers.cmake "
                    "2026-06-03.)"
                ),
            },
            "cmake_test_setup_options": {
                "description": (
                    "Configure-time CMake options + derived "
                    "constants defined in cmake/setup_tests.cmake "
                    "— controls GoogleTest unit-test fetching, "
                    "Google Benchmark micro-benchmark fetching, "
                    "and the test-timeout scaling system."),
                "options": {
                    "FOUR_C_TEST_TIMEOUT_SCALE": (
                        "STRING (integer-valued), default 4 when "
                        "FOUR_C_BUILD_TYPE_UPPER==DEBUG else 1. "
                        "Multiplier applied to all test timeouts. "
                        "Silent 4× scaling in Debug builds — easy "
                        "to miss when comparing CI durations "
                        "between Debug and Release."),
                    "FOUR_C_WITH_GOOGLETEST": (
                        "bool, default ON. Toggles GoogleTest "
                        "v1.15.2 FetchContent (pinned commit "
                        "b514bdc898e2951020cbdca1304b75f5950d1f59) "
                        "and the `unittests` custom target. The "
                        "`full` target depends on `unittests`."),
                    "FOUR_C_WITH_GOOGLE_BENCHMARK": (
                        "bool, default OFF. Toggles Google "
                        "Benchmark v1.9.2 FetchContent (pinned "
                        "commit afa23b7699c17f1e26c88cbf95257b20d"
                        "78d6247) and the `benchmarktests` custom "
                        "target. Implicitly sets "
                        "BENCHMARK_ENABLE_TESTING=OFF to skip "
                        "google-benchmark's own internal tests."),
                    "FOUR_C_ENABLE_FULL_BENCHMARK_TESTS": (
                        "In 4C 2026.3.0; not in cmake/setup_tests.cmake of 4C 2026.2.0 and earlier. "
                        "bool, default OFF, ONLY visible when "
                        "FOUR_C_WITH_GOOGLE_BENCHMARK=ON. OFF "
                        "means dry-run mode (10s timeout); ON "
                        "means real benchmark execution (600s "
                        "timeout × FOUR_C_TEST_TIMEOUT_SCALE)."),
                    "FOUR_C_BENCHMARK_TESTS_COLLECTION_FILE": (
                        "In 4C 2026.3.0; not in cmake/setup_tests.cmake of 4C 2026.2.0 and earlier. "
                        "PATH, default ${PROJECT_BINARY_DIR}/"
                        "benchmark_test_results.json. Output JSON "
                        "where 4C aggregates benchmark results via "
                        "four_c_collect_benchmark_test_results."),
                    "FOUR_C_ENABLE_FULL_PERFORMANCE_TESTS": (
                        "In 4C 2026.3.0; not in cmake/setup_tests.cmake of 4C 2026.2.0 and earlier. "
                        "bool, default OFF. Switches performance "
                        "tests between full and minimal execution. "
                        "Distinct from FOUR_C_ENABLE_FULL_BENCHMARK_"
                        "TESTS — performance tests are 4C-internal, "
                        "benchmark tests use Google Benchmark."),
                    "FOUR_C_PERFORMANCE_TESTS_COLLECTION_FILE": (
                        "In 4C 2026.3.0; not in cmake/setup_tests.cmake of 4C 2026.2.0 and earlier. "
                        "PATH, default ${PROJECT_BINARY_DIR}/"
                        "performance_test_results.json."),
                },
                "derived_constants": {
                    "FOUR_C_TEST_GLOBAL_TIMEOUT": (
                        "120 * FOUR_C_TEST_TIMEOUT_SCALE — global "
                        "ctest timeout floor."),
                    "UNITTEST_TIMEOUT": (
                        "10 * FOUR_C_TEST_TIMEOUT_SCALE — per-"
                        "unit-test timeout (set when "
                        "FOUR_C_WITH_GOOGLETEST=ON)."),
                    "BENCHMARK_TEST_TIMEOUT": (
                        "In 4C 2026.3.0; not in cmake/setup_tests.cmake of 4C 2026.2.0 and earlier. "
                        "10 (dry-run) or 600 (full) * "
                        "FOUR_C_TEST_TIMEOUT_SCALE."),
                    "FOUR_C_INSTALL_PREFIX": (
                        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_"
                        "DATADIR}/cmake/4C — where the install-"
                        "test harness expects 4CConfig.cmake."),
                },
                "install_test_configure_files": (
                    "Three .in templates configure_file'd at "
                    "configure time into ${PROJECT_BINARY_DIR}/"
                    "tests/install_test/: main.cpp, CMakeLists.txt, "
                    "test_install.sh — used by CI to verify the "
                    "installed 4CConfig.cmake works for downstream "
                    "consumers."),
                "Signal": (
                    "[Integration] cmake/setup_tests.cmake "
                    "FetchContent's GoogleTest at PINNED COMMIT "
                    "b514bdc898e2951020cbdca1304b75f5950d1f59 "
                    "(v1.15.2) and Google Benchmark at PINNED "
                    "COMMIT afa23b7699c17f1e26c88cbf95257b20d78d6247 "
                    "(v1.9.2). Two FATAL_ERROR guards catch "
                    "TARGET-name clashes when 4C is embedded in a "
                    "larger CMake project that has already pulled "
                    "in either library: "
                    "  if(TARGET gtest) → FATAL_ERROR 'A target "
                    "<gtest> has already been included by another "
                    "library. This is not supported.' "
                    "  if(TARGET benchmark_main) → FATAL_ERROR 'A "
                    "target <benchmark_main> has already been "
                    "included by another library. This is not "
                    "supported.' "
                    "Workarounds when integrating 4C into an "
                    "umbrella project: (a) set "
                    "FOUR_C_WITH_GOOGLETEST=OFF and/or "
                    "FOUR_C_WITH_GOOGLE_BENCHMARK=OFF to skip the "
                    "fetch entirely (the parent project must then "
                    "not link against 4C's unit-test executables); "
                    "(b) override the FetchContent source via "
                    "FETCHCONTENT_SOURCE_DIR_GOOGLETEST and "
                    "FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK to "
                    "point at the parent project's pre-included "
                    "copies — note that the FATAL_ERROR fires "
                    "BEFORE the FetchContent_MakeAvailable call, "
                    "so source-dir override alone is not sufficient. "
                    "[Performance] FOUR_C_TEST_TIMEOUT_SCALE "
                    "defaults to 4 in DEBUG builds (line 8-12) "
                    "vs 1 in Release/RelWithDebInfo. This silently "
                    "quadruples ALL test timeouts (UNITTEST_TIMEOUT "
                    "40s, GLOBAL_TIMEOUT 480s, and in 4C 2026.3.0 "
                    "BENCHMARK_TEST_TIMEOUT 2400s in full-benchmark "
                    "mode) when the user "
                    "switches between Debug and Release without "
                    "changing CMakeCache.txt. CI run-time diffs "
                    "between Debug and Release jobs often surface "
                    "this. (File walk cmake/setup_tests.cmake "
                    "2026-06-03.)"
                ),
            },
            "cmake_dependency_configure_pattern": {
                "description": (
                    "Pattern shared by cmake/configure/configure_"
                    "<Dep>.cmake files (one per external "
                    "dependency: ArborX, Backtrace, Boost, "
                    "CLI11, CLN, FFTW, HDF5, MIRCO, MPI, Qhull, "
                    "Trilinos, VTK, ZLIB, gmsh, deal.II, "
                    "magic_enum, pybind11, ryml). Each file "
                    "controls HOW to obtain that dependency at "
                    "configure time; the higher-level "
                    "FOUR_C_WITH_<Dep> toggle (cmake_install_"
                    "export.dependency_toggles_FOUR_C_WITH_*) "
                    "controls WHETHER to use it at all."),
                "shape": (
                    "There are THREE sub-shapes. "
                    "HEAVY (e.g. configure_ArborX.cmake, plus "
                    "Trilinos / VTK / HDF5 / deal.II): "
                    "(1) Declares a FOUR_C_<DEP>_FIND_INSTALLED "
                    "boolean option (default usually OFF) via "
                    "four_c_process_global_option. "
                    "(2) When ON: find_package(<Dep> HINTS "
                    "${FOUR_C_<DEP>_ROOT}); FATAL_ERROR with a "
                    "per-dep message if find fails. "
                    "(3) When OFF: fetchcontent_declare/"
                    "_makeavailable from a PINNED commit hash, "
                    "then sets FOUR_C_<DEP>_ROOT to "
                    "${CMAKE_INSTALL_PREFIX}. "
                    "(4) four_c_remember_variable_for_install "
                    "on both FOUR_C_<DEP>_FIND_INSTALLED + "
                    "FOUR_C_<DEP>_ROOT. "
                    "LIGHT-FIND (e.g. configure_Backtrace.cmake, "
                    "plus Boost / ZLIB / typical system "
                    "libraries): "
                    "(1) NO FOUR_C_<DEP>_FIND_INSTALLED toggle "
                    "— no fetch fallback. "
                    "(2) Direct find_package(<Dep> REQUIRED) — "
                    "configure aborts if not found. "
                    "(3) target_link_libraries(four_c_all_"
                    "enabled_external_dependencies INTERFACE "
                    "<Dep>::<Dep>) attaches the dep. "
                    "(4) four_c_remember_variable_for_install "
                    "on the relevant FindPackage cache vars "
                    "(e.g. <Dep>_INCLUDE_DIR, <Dep>_LIBRARY) "
                    "for downstream replay. "
                    "LIGHT-FETCH (e.g. configure_CLI11.cmake, "
                    "plus magic_enum / pybind11): "
                    "(1) NO FOUR_C_<DEP>_FIND_INSTALLED toggle "
                    "and NO find_package — fetch is UNCONDITIONAL. "
                    "(2) fetchcontent_declare with GIT_REPOSITORY "
                    "+ pinned GIT_TAG commit; flags like "
                    "<DEP>_BUILD_DOCS=OFF, <DEP>_BUILD_TESTS=OFF, "
                    "<DEP>_BUILD_EXAMPLES=OFF, <DEP>_INSTALL=ON "
                    "set as CACHE BOOLs before "
                    "fetchcontent_makeavailable. "
                    "(3) Some (CLI11) temporarily swap "
                    "CMAKE_PROJECT_NAME to trick the dep into "
                    "exporting install rules, then restore the "
                    "original. "
                    "(4) four_c_add_external_dependency on the "
                    "<Dep>::<Dep> imported target + "
                    "four_c_remember_variable_for_install on "
                    "FOUR_C_<DEP>_ROOT=${CMAKE_INSTALL_PREFIX}. "
                    "User-impact: with no FIND_INSTALLED toggle, "
                    "a system pybind11/CLI11/magic_enum CANNOT "
                    "be used — the build always vendors its own "
                    "pinned copy."),
                "Signal": (
                    "[Integration] Two-layer toggle structure to "
                    "be aware of: FOUR_C_WITH_<Dep>=ON enables "
                    "the dependency at all, then FOUR_C_<DEP>_"
                    "FIND_INSTALLED=ON/OFF controls find-vs-fetch. "
                    "Wanting to use a system-installed library "
                    "but forgetting to set FIND_INSTALLED=ON "
                    "silently triggers a fetch+build of a pinned "
                    "vendored version, doubling configure time "
                    "and producing two copies of the dep in the "
                    "build tree. Pinned-commit fetch fallback "
                    "uses fetchcontent_declare + "
                    "fetchcontent_makeavailable; each dep's "
                    "configure_<Dep>.cmake has its own "
                    "GIT_REPOSITORY + GIT_TAG commit hash hard-"
                    "coded (e.g. ArborX is pinned to "
                    "f9244ba03904cc518a54d99e9f87bb42dc9ecaf3 = "
                    "v2.0.1, ARBORX_ENABLE_MPI=ON unconditionally "
                    "forced; CLI11 pinned to "
                    "bfffd37e1f804ca4fae1caae106935791696b6a9 = "
                    "v2.6.1). To switch fetch source: override "
                    "FETCHCONTENT_SOURCE_DIR_<dep> before "
                    "fetchcontent_makeavailable. LIGHT-FETCH "
                    "deps (CLI11/magic_enum/pybind11) have NO "
                    "FIND_INSTALLED escape hatch — system "
                    "installs are ignored, the build always "
                    "vendors. (File walk cmake/configure/"
                    "configure_ArborX.cmake + configure_CLI11.cmake "
                    "2026-06-03; LIGHT-FETCH shape verified "
                    "across CLI11 + magic_enum + pybind11.)"
                ),
            },
            "cmake_install_export": {
                "description": (
                    "Configure-time surfaces defined in "
                    "cmake/setup_install.cmake — install rules, "
                    "exported 4CTargets, and the 4CConfig.cmake "
                    "consumer file."),
                "exported_targets_namespace": "4C::",
                "config_file_destination": (
                    "${CMAKE_INSTALL_DATADIR}/cmake/4C/4CConfig.cmake"
                    " (plus 4CConfigVersion.cmake)"),
                "version_compatibility": "ExactVersion",
                "dependency_toggles_FOUR_C_WITH_*": [
                    "HDF5", "MPI", "Qhull", "Trilinos", "VTK",
                    "gmsh", "deal.II", "Boost", "ArborX", "FFTW",
                    "CLN", "MIRCO", "Backtrace", "ryml",
                    "magic_enum", "ZLIB", "pybind11", "CLI11",
                ],
                "rolled_up_dependency_target": (
                    "four_c_all_enabled_external_dependencies — "
                    "single CMake target rolling up every "
                    "FOUR_C_WITH_<X>=ON external; downstream "
                    "consumers link via 4C::lib4C only."),
                "Signal": (
                    "[Input] 4CConfig.cmake exports the package "
                    "with COMPATIBILITY ExactVersion (line 100 of "
                    "setup_install.cmake). Downstream "
                    "find_package(4C <version> EXACT) requires "
                    "EXACT FOUR_C_VERSION_MAJOR.MINOR match — "
                    "find_package(4C 1.3) when 4C is installed at "
                    "1.4 FAILS with 'incompatible version' even "
                    "though they may be API-compatible. Use "
                    "find_package(4C) (no version pin) to fall "
                    "back to whatever is installed, or pin to the "
                    "EXACT installed MAJOR.MINOR. The 18-package "
                    "FOUR_C_WITH_<X> boolean surface (HDF5 / MPI "
                    "/ Qhull / Trilinos / VTK / gmsh / deal.II / "
                    "Boost / ArborX / FFTW / CLN / MIRCO / "
                    "Backtrace / ryml / magic_enum / ZLIB / "
                    "pybind11 / CLI11) is set by the parent "
                    "build's FOUR_C_WITH_<X> CMake cache values "
                    "and baked into the exported config — "
                    "downstream cannot RE-enable a dep that was "
                    "OFF at 4C install time. The "
                    "four_c_all_enabled_external_dependencies "
                    "rolled-up target is the canonical downstream "
                    "link edge; downstream projects do "
                    "target_link_libraries(myapp PRIVATE "
                    "4C::lib4C) and inherit the dependency "
                    "transitively. (File walk "
                    "cmake/setup_install.cmake 2026-06-03.)"
                ),
            },
            "additional_io_input_keys": {
                "WRITE_TIMINGS": (
                    "bool — when true, run() writes "
                    "`<output_file_identifier>-timings.yaml` via export_timings() "
                    "after the simulation completes. Read from io_params."),
            },
            "memory_high_water_mark_summary": (
                "After run() completes, main calls get_memory_high_water_mark(comm) "
                "which reads /proc/self/status for VmHWM, MPI-gathers, and prints "
                "'Memory High Water Mark Summary: MinOverProcs [PID] / MeanOverProcs / "
                "MaxOverProcs [PID] / SumOverProcs' in GB. LINUX-ONLY — guarded by "
                "#if defined(__linux__); macOS/Windows runs print 'Memory High Water "
                "Mark summary not available on this operating system.' instead. "
                "Failure to open /proc/self/status (e.g. namespace restrictions in "
                "containers) prints a friendlier 'status file could not be opened "
                "on every proc.' rather than failing — does NOT abort the run."),
            "io_input_keys": {
                # YAML/dat keys read from the input file's I/O block by setup_parallel_output
                "WRITE_TO_SCREEN":     "bool — stream Core::IO::cout to stdout",
                "WRITE_TO_FILE":       "bool — stream Core::IO::cout to log file",
                "PREFIX_GROUP_ID":     "bool — prepend group ID to each line",
                "LIMIT_OUTP_TO_PROC":  "int — limit per-rank output to this MPI rank only",
                "VERBOSITY":           "Core::IO::Verbositylevel enum (e.g. verbose, standard, minimal)",
            },
            "Signal": (
                "[Input] CLI validation in validate_argument_cross_compatibility() "
                "raises FOUR_C_THROW with literal text messages — these are "
                "the verbatim error strings:\n"
                "  - 'When --glayout is provided its number of entries must "
                "equal --ngroup.'\n"
                "  - 'when --ngroup > 1, a nested parallelism type must be "
                "specified via --nptype.'\n"
                "  - 'when using \\'no_nested_parallelism\\' or "
                "\\'everyGroupReadInputFile\\' the number of <input> <output> "
                "pairs must be exactly 1.'\n"
                "  - 'when using \\'separateInputFiles\\' or "
                "\\'nestedMultiscale\\' the number of <input> <output> pairs "
                "must equal --ngroup ...'\n"
                "  - 'When using --nptype other than \\'separateInputFiles\\', "
                "only one restart step and one restartfrom identifier must be given.'\n"
                "  - 'You need to specify a restart step when using restartfrom.'\n"
                "  - 'Positional arguments must be provided as pairs: <input> <output>.'\n"
                "Mixed naming: ENUM values are snake_case in C++ "
                "(no_nested_parallelism) but the CLI string parses the "
                "camelCase ALIAS (everyGroupReadInputFile / separateInputFiles "
                "/ nestedMultiscale). The error messages quote BOTH forms in "
                "the same sentence — user-facing inconsistency. (File walk "
                "apps/global_full/4C_global_full_io.cpp 2026-06-02.)"
            ),
            "output_naming_under_groups": (
                "When --nptype=everyGroupReadInputFile is set, output_identifier "
                "gets _group_<N> suffix appended. If the user's identifier "
                "already ends with -<num> (e.g. 'mysim-001'), the suffix is "
                "inserted as 'mysim_group_<N>_001'. Restart identifier follows "
                "the same convention. Source: update_io_identifiers() switch case."
            ),
        },
        "post_monitor_tool": {
            "description": (
                "The standalone post_monitor CLI binary extracts time-history "
                "of a single node into an ASCII .mon file. Source: "
                "apps/post_monitor/4C_post_monitor.cpp main()."),
            "cli_arguments": {
                "--node": "Required int. Global node id whose history to dump.",
                "--field": ("String, default 'fluid'. Selects which "
                            "discretization the node belongs to. Valid "
                            "vocabulary per ProblemType is hard-coded "
                            "in the main() switch — see "
                            "supported_field_per_problem below."),
            },
            "output_file_suffixes": {
                ".mon": "primary fields per write_mon_file()",
                ".stress.mon": "Cauchy + 2nd-PK stresses",
                ".strain.mon": "Green-Lagrange / Euler-Almansi / Log strains",
                ".plasticstrain.mon": "plastic GL / plastic EA strains",
                ".heatflux.mon": "thermo current + initial heatfluxes (thermo only)",
                ".tempgrad.mon": "thermo current + initial tempgrads (thermo only)",
            },
            "stresstype_straintype_heatfluxtype_enum": [
                "none",
                "ndxyz",
            ],
            "supported_field_per_problem": {
                "fsi / fsi_redmodels": ["fluid", "structure"],
                "structure / loma / fluid / fluid_redmodels / fps3i":
                    ["scatra", "fluid", "structure"],
                "ale": ["ale"],
                "thermo": ["thermo"],
                "red_airways": ["red_airway"],
                "poroelast": ["fluid", "structure"],
                "porofluid_pressure_based": ["porofluid"],
                "porofluid_pressure_based_elast":
                    ["structure", "porofluid"],
                "porofluid_pressure_based_elast_scatra":
                    ["structure", "porofluid", "scatra", "artery_scatra"],
            },
            "stress_strain_groupnames_at_write_time": {
                "stress": ["gauss_cauchy_stresses_xyz", "gauss_2PK_stresses_xyz"],
                "strain": ["gauss_GL_strains_xyz", "gauss_EA_strains_xyz",
                           "gauss_LOG_strains_xyz"],
                "plastic_strain": ["gauss_pl_GL_strains_xyz",
                                   "gauss_pl_EA_strains_xyz"],
                "heatflux": ["gauss_current_heatfluxes_xyz",
                             "gauss_initial_heatfluxes_xyz"],
                "tempgrad": ["gauss_initial_tempgrad_xyz",
                             "gauss_current_tempgrad_xyz"],
            },
            "Signal": (
                "[Output] Seven sharp edges users routinely hit running "
                "post_monitor: "
                "(1) SERIAL-ONLY tool. The source-file header comment says "
                "'Works in seriell version only! Requires to read one "
                "instance of the discretisation!!'; the body counts node "
                "owners across all ranks and FOUR_C_THROW('Found more than "
                "one owner of node {}: {}') if more than one rank owns the "
                "node. Running with mpirun -n>1 errors out at the first "
                "node lookup. "
                "(2) The stresstype / straintype / heatfluxtype enum is "
                "EXACTLY {'none', 'ndxyz'} — any other value (e.g. 'cxyz', "
                "'averaged', '123') triggers FOUR_C_THROW('Cannot deal "
                "with requested <kind> output type: {}'). Common confusion: "
                "4C output formats elsewhere use 'cxyz' for cell-based, but "
                "post_monitor accepts only the nodal 'ndxyz' form. "
                "(3) FSI + --field=ale is explicitly REJECTED with "
                "FOUR_C_THROW('There is no ALE output. Displacements of "
                "fluid nodes can be printed.') — there's even a leftover "
                "FsiAleMonWriter ctor call after the throw that is dead "
                "code. Use --field=fluid to get the fluid-side ALE "
                "displacements. "
                "(4) ProblemType red_airways silently NO-OPS if "
                "--field != 'red_airway'. The main() case has an if-check "
                "and no else clause — wrong field value writes no .mon "
                "file and prints no error. "
                "(5) Stress / strain time-history is DEAD CODE in this "
                "tool. write_mon_stress_file, write_mon_strain_file, "
                "write_mon_pl_strain_file are defined on MonWriter but "
                "main() never calls them — only thermo's heatflux and "
                "tempgrad are auto-invoked. The .stress.mon / .strain.mon "
                "files are produced only if the user calls the methods "
                "programmatically, NOT from the CLI. "
                "(6) CLI default --field=fluid is silently applied to "
                "structural / thermo / scatra runs where it would be "
                "rejected — easy oversight: the first FOUR_C_THROW the "
                "user sees is 'Node {} does not belong to fluid field!' "
                "even though they're running a structure problem. "
                "(7) ProblemType gas_fsi / biofilm_fsi / thermo_fsi are "
                "in the dispatch but throw FOUR_C_THROW('not implemented "
                "yet') — the tool's coverage is narrower than the full "
                "ProblemType set. Default unknown ProblemType triggers "
                "FOUR_C_THROW('problem type {} not yet supported'). "
                "(File walk apps/post_monitor/4C_post_monitor.cpp "
                "2026-06-03.)"
            ),
        },
        "post_processor_tool": {
            "description": (
                "The standalone post_processor CLI binary — the bigger "
                "sibling of post_monitor. Reads native 4C output and "
                "writes per-field visualization files (Ensight .case or "
                "ParaView VTU/VTI). Source: apps/post_processor/"
                "4C_post_processor.cpp main() + run_ensight_vtu_filter()."),
            "cli_arguments": {
                "--filter": ("String, default 'ensight'. CASE-SENSITIVE "
                             "enum: {'ensight', 'vtu', 'vtu_node_based', "
                             "'vti'}. Any other value FOUR_C_THROWs "
                             "'Unknown filter {} given, supported "
                             "filters: [ensight|vtu|vti]'."),
            },
            "supported_problem_types_in_dispatch": [
                "fsi", "fsi_redmodels", "gas_fsi", "thermo_fsi",
                "biofilm_fsi", "structure", "polymernetwork",
                "fluid", "fluid_redmodels", "fluid_ale",
                "particle", "pasi", "ale", "lubrication",
                "cardiac_monodomain", "scatra",
                "fsi_xfem", "fpsi_xfem", "fluid_xfem",
                "loma", "elch", "art_net", "thermo",
                "red_airways", "poroelast", "poroscatra",
                "fpsi", "fbi", "fps3i", "ehl", "none",
            ],
            "filter_writer_classes": [
                "StructureFilter (also used for art_net, red_airways)",
                "FluidFilter (also used for porofluid)",
                "XFluidFilter (XFEM-only)",
                "AleFilter",
                "MortarFilter (structure problem with do_mortar_interfaces)",
                "InterfaceFilter (fsi_xfem boundary discretizations)",
                "ThermoFilter (uses heatfluxtype + tempgradtype)",
                "LubricationFilter",
                "AnyFilter (ProblemType::none — write whatever vectors exist)",
            ],
            "filter_result_tags_per_writer": {
                "StructureFilter": (
                    "~50 tags. Primary: displacement, prolongated_"
                    "gauss_2PK_stresses_xyz, prolongated_gauss_GL_"
                    "strains_xyz, material_displacement (if struct_"
                    "mat_disp='yes'). Contact: activeset, contact"
                    "owner, nor/tan contactstress, slave/master"
                    "forces (+lm/g suffixes), interfacetraction, "
                    "wear, poronopen_lambda. Spring/dashpot: gap, "
                    "curnormals, springstress. Error norms: L2_norm, "
                    "H1_norm, Energy_norm. 1D artery: one_d_artery_"
                    "{pressure,flow,area}, forward/backward speed[0]. "
                    "Reduced airway: pnp/p_nonlin, NodeIDs, radii, "
                    "scatraO2np, PO2, dVO2, AcinarPO2, acini_vnp, "
                    "qin_np/qout_np, x_np, open, p_extnp/n, generations, "
                    "elemVolume[0]np, elemRadius_current. FSI: "
                    "Add_Forces, fsilambda, fpilambda_ps/pf. Biofilm: "
                    "str_growth_displ. Poro: porosity_p1. SSI: nodal_"
                    "stresses_xyz. EHL: fluid_force, normal/tangential_"
                    "contact, active, slip. Plus Gauss-point post-stress "
                    "and rotation R."),
                "FluidFilter": (
                    "~40 tags. Primary: velnp (-> 'velocity'), pressure, "
                    "scalar_field, residual. Averaged: averaged_pressure/"
                    "velnp/scanp. Filtered: filteredvel, fsvelaf. ALE: "
                    "dispnp, idispnfull, traction. Wall-shear: wss, "
                    "wss_mean. XWall: xwall_enrvelnp, xwall_tauw, par_vel. "
                    "FSI volume-constraint: Add_Forces. Poro: convel, "
                    "gridv. Adjoint: adjoint_velnp/pressure. Meshfree: "
                    "velatmeshfreenodes, pressureatmeshfreenodes. FSI "
                    "Lagrange mul.: fsilambda. Biofilm: fld_growth_displ. "
                    "HDG: velnp_hdg, pressure_hdg, tracevel_hdg. XFluid "
                    "level-set: fluid_levelset_boundary + phinp_0..19."),
                "XFluidFilter": (
                    "5 tags ONLY (XFEM-specific naming): velocity_"
                    "smoothed, pressure_smoothed, averaged_velnp, "
                    "averaged_pressure, fsvelocity."),
                "MortarFilter": (
                    "8 tags: displacement, nor/tan contactstress, "
                    "interface traction, slave/master forces (+nor/tan "
                    "suffixes)."),
                "InterfaceFilter": (
                    "7 tags (interface-side FSI accessors): idispnp/n, "
                    "ivelnp/n/nm, iaccn, itrueresnp."),
                "AleFilter": (
                    "3 tags: dispnp (-> 'displacement'), det_j, "
                    "element_quality."),
                "LubricationFilter": (
                    "5 tags: prenp (-> 'pressure'), height, no_gap_DBC, "
                    "dispnp, viscosity."),
                "ThermoFilter": (
                    "Primary: temperature (NOT 'tempnp'). Optional "
                    "Gauss-point post: gauss_{current,initial}_heatfluxes_"
                    "xyz → 'heatflux' (nodebased), gauss_{current,initial}_"
                    "tempgrad_xyz → 'tempgrad' (nodebased). Plus "
                    "displacement (TSI), and SLM-specific: phase, "
                    "conductivity, capacity."),
                "AnyFilter": (
                    "Writes all dof + node + element results blindly "
                    "(no tag whitelist)."),
            },
            "structure_filter_one_time_step_subset": (
                "StructureFilter::write_all_results_one_time_step "
                "(line 173) writes ONLY displacement + node results, "
                "NOT the full ~50-tag set. Used for partial restart-"
                "style writes. Users expecting stresses/strains/contact "
                "tags from a per-step call get only displacement."),
            "structure_stress_filter_internals": {
                "post_stress_stresstype_enum": [
                    "ndxyz",
                    "cxyz",
                    "cxyz_ndxyz",
                    "nd123",
                    "c123",
                    "c123_nd123",
                ],
                "post_stress_dispatch": {
                    "ndxyz": "write_stress(..., nodebased)",
                    "cxyz": "write_stress(..., elementbased)",
                    "cxyz_ndxyz": (
                        "write_stress(..., nodebased) then PostResult "
                        "reset then write_stress(..., elementbased) "
                        "— dual write"),
                    "nd123": "write_eigen_stress(..., nodebased)",
                    "c123": "write_eigen_stress(..., elementbased)",
                    "c123_nd123": (
                        "write_eigen_stress(..., nodebased) then "
                        "PostResult reset then write_eigen_stress("
                        "..., elementbased) — dual write"),
                },
                "special_field_subclasses_in_file": [
                    "WriteNodalStressStep (6-component nodal stress, "
                    "via Core::FE::extrapolate_gauss_point_quantity_"
                    "to_nodes)",
                    "WriteElementCenterStressStep (6-component element-"
                    "center stress, via Core::FE::evaluate_gauss_point_"
                    "quantity_at_element_center)",
                    "WriteElementCenterRotation (9-component element-"
                    "center rotation tensor R, only triggered when "
                    "groupname=='rotation'; comment 'pfaller may17')",
                    "WriteNodalEigenStressStep (num_df_map = "
                    "{1,1,1,3,3,3} — 3 eigenvalues + 3 eigenvector "
                    "columns × 3 components; uses symmetric_eigen_"
                    "problem)",
                    "WriteElementCenterEigenStressStep (same shape as "
                    "WriteNodalEigenStressStep but at element centers)",
                ],
                "write_stress_groupname_vocab": [
                    "gauss_2PK_stresses_xyz",
                    "gauss_cauchy_stresses_xyz",
                    "gauss_GL_strains_xyz",
                    "gauss_EA_strains_xyz",
                    "gauss_LOG_strains_xyz",
                    "gauss_pl_GL_strains_xyz",
                    "gauss_pl_EA_strains_xyz",
                    "rotation",
                ],
                "write_eigen_stress_groupname_vocab": [
                    "gauss_2PK_stresses_xyz",
                    "gauss_cauchy_stresses_xyz",
                    "gauss_GL_strains_xyz",
                    "gauss_EA_strains_xyz",
                    "gauss_LOG_strains_xyz",
                    "gauss_pl_GL_strains_xyz",
                    "gauss_pl_EA_strains_xyz",
                ],
                "eigen_output_naming_pattern": (
                    "For each groupname, write_eigen_stress emits 6 "
                    "names: <base>_eigenval{1,2,3} (1 component each) "
                    "and <base>_eigenvec{1,2,3} (3 components each). "
                    "Both nodal_ and element_ prefixes applied via "
                    "stresskind dispatch."),
            },
            "thermo_heatflux_filter_internals": {
                "post_heatflux_heatfluxtype_enum": [
                    "ndxyz",
                    "cxyz",
                    "cxyz_ndxyz",
                ],
                "post_heatflux_dispatch": {
                    "ndxyz": "write_heatflux(..., nodebased)",
                    "cxyz": "write_heatflux(..., elementbased)",
                    "cxyz_ndxyz": (
                        "write_heatflux(..., nodebased) then "
                        "PostResult reset then write_heatflux("
                        "..., elementbased) — dual write"),
                },
                "special_field_subclasses_in_file": [
                    "WriteNodalHeatfluxStep (numdf-aware 1/2/3 "
                    "components; uses Thermo::postproc_thermo_"
                    "heatflux action via dis->evaluate; "
                    "components averaged across adjoining elements "
                    "via /adjele)",
                    "WriteElementCenterHeatfluxStep (numdf-aware "
                    "1/2/3 components; passes 'eleheatflux' "
                    "vector to elements; FOUR_C_THROW if returned "
                    "vector is nullptr)",
                ],
                "write_heatflux_groupname_vocab": [
                    "gauss_initial_heatfluxes_xyz",
                    "gauss_current_heatfluxes_xyz",
                    "gauss_initial_tempgrad_xyz",
                    "gauss_current_tempgrad_xyz",
                ],
                "element_action_routed": (
                    "Thermo::postproc_thermo_heatflux — routed via "
                    "Teuchos::ParameterList p.set<Thermo::Action>("
                    "'action', ...). 'heatfluxtype' is passed AGAIN "
                    "as a parameter string ('ndxyz' or 'cxyz') to "
                    "tell the element which output shape to fill."),
                "numdf_per_dim": (
                    "WriteNodalHeatfluxStep + WriteElementCenter"
                    "HeatfluxStep::numdf() return 1/2/3 from "
                    "problem()->num_dim(). FOUR_C_THROW('Cannot "
                    "handle dimension {}') for any other dim. "
                    "Average is per-element-incidence (adjele = "
                    "lnode->num_element() in the nodal averaging "
                    "loop) — boundary nodes with fewer adjacent "
                    "elements get the same /adjele divisor as "
                    "interior nodes."),
            },
            "Signal": (
                "[Output] Six sharp edges in post_processor most users "
                "hit at least once: "
                "(1) --filter is case-sensitive enum {'ensight', 'vtu', "
                "'vtu_node_based', 'vti'}. Common mistakes: 'VTU' "
                "(uppercase), 'paraview', 'vtkhdf' — all FOUR_C_THROW "
                "'Unknown filter <X> given'. "
                "(2) On problemtype scatra / cardiac_monodomain / elch "
                "with num_discr() == 1, the tool SILENTLY DOES NOTHING "
                "(comment in source: 'runtime output is used for scatra'). "
                "No .case file appears, no error, no warning. The "
                "runtime VTU output written during the solve is the only "
                "result; users running post_processor expecting "
                "additional output get nothing. "
                "(3) The fluid case has [[fallthrough]] to fluid_redmodels "
                "which has [[fallthrough]] to fluid_ale. A `fluid` "
                "problem with num_discr()==2 and disc[1].name()=='xfluid' "
                "writes THREE filters (XFluidFilter for disc[1] + the "
                "fluid_redmodels artery branch's StructureFilter on the "
                "same disc + FluidFilter for disc[0]). If disc[1] is NOT "
                "named 'xfluid', the fluid_redmodels artery branch still "
                "writes StructureFilter on disc[1], which is wrong for a "
                "pure ALE fluid. "
                "(4) fsi_xfem / fpsi_xfem branch has an INVERTED-LOGIC "
                "test: `disname.compare(1, 12, \"boundary_of_\")` "
                "returns 0 (falsy) when the substring at offset 1 IS "
                "'boundary_of_', so the InterfaceFilter branch runs ONLY "
                "for discretizations whose name does NOT match. Discs "
                "literally named like '_boundary_of_fluid' fall to the "
                "FOUR_C_THROW 'You try to postprocess a discretization "
                "with name {X}, maybe you should add it here?'. The fix "
                "would be `== 0` or `starts_with` — upstream bug worth "
                "reporting. "
                "(5) fsi_xfem ALE branch (case Core::ProblemType::fsi_xfem "
                "around line 305-310) is DEAD CODE: 'ale' name matches "
                "and prints the header but the AleFilter constructor + "
                "WriteFiles call are COMMENTED OUT. ALE fields in "
                "fsi_xfem problems produce no output despite the visual "
                "indicator. "
                "(6) ProblemType::none uses AnyFilter and writes "
                "'whatever vectors exist' in the first discretization. "
                "This is the right escape hatch for ad-hoc field dumps "
                "but offers no diagnostic if the discretization is "
                "missing — user sees an empty .case. "
                "(7) FluidFilter::write_all_results has a HARDCODED "
                "`int num_levelsets = 20;` (line 271 of "
                "single_field_writers.cpp) which unconditionally loops "
                "writing phinp_0..phinp_19 from the XFluid level-set "
                "store. Users with fewer than 20 level-sets get "
                "silent no-ops for the missing tags; users with MORE "
                "than 20 lose level-sets 20+ from .case output. "
                "Recompile-only knob — no runtime override. "
                "(8) XFluidFilter uses the `_smoothed` naming "
                "convention (velocity_smoothed, pressure_smoothed) "
                "for its 4-DOF-per-node fixed-size Paraview vectors, "
                "NOT the raw XFEM runtime names ('velocity', "
                "'pressure'). Source comment (line 286-294) "
                "explains XFEM has changing DOF counts so restart "
                "vectors and Paraview vectors are different sizes. "
                "Users grepping a runtime XFEM .out for 'velocity' "
                "see the raw name; users opening the post_processor "
                ".case in ParaView see velocity_smoothed — and "
                "looking for the wrong name gives 'field not found'. "
                "(9) StructureFilter + ThermoFilter post-stress / "
                "post-heatflux paths call BOTH alternatives "
                "(gauss_cauchy_stresses_xyz AND gauss_2PK_stresses_xyz; "
                "{current,initial}_heatfluxes / tempgrad pairs) even "
                "though only ONE is present in the result archive at "
                "a time. Comments at lines 142-159 / 362-379 spell "
                "this out: 'only one function call to PostStress is "
                "really postprocessing Gauss point stresses'. The "
                "non-present tag's write_result is silently a no-op. "
                "Plus 5 strain types are tried "
                "(GL/EA/LOG/pl_GL/pl_EA) — users counting writes "
                "from log lines see 'attempted N writes, got M' "
                "without an error indicator. "
                "(12) Dead wrapper script: apps/post_processor/"
                "scripts/post_gid ships in every build "
                "(installed alongside post_ensight / post_vti / "
                "post_vtu by create_post_scripts.cmake's "
                "copy_script() invocations) but invokes "
                "`post_processor --filter=gid $@` — and 'gid' is "
                "NOT in the post_processor filter enum (which is "
                "{'ensight', 'vtu', 'vtu_node_based', 'vti'} per "
                "edge 1). Every invocation of `./post_gid <file>` "
                "FOUR_C_THROWs 'Unknown filter gid given, "
                "supported filters: [ensight|vtu|vti]' at the "
                "filter-dispatch line in apps/post_processor/"
                "4C_post_processor.cpp. The wrapper is dead "
                "code — likely a leftover from when 4C had a "
                "Ciarlet-Geuzaine GiD output backend that was "
                "removed without updating the CMake glue. Users "
                "expecting GiD output get no help text, just the "
                "filter-enum error. Workaround: drop the GiD "
                "format entirely and use ensight / vtu / vti, OR "
                "edit post_processor source to add a 'gid' "
                "branch (requires an actual GiD writer "
                "implementation which the 4C codebase no longer "
                "ships). "
                "(11) ThermoFilter::post_heatflux dispatches on a "
                "3-VALUE heatfluxtype enum {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz'} — NO eigen variants. Three-tool "
                "asymmetry users routinely confuse: "
                "post_monitor accepts {'none', 'ndxyz'} (2 values, "
                "tick #54); post_processor structure_stress accepts "
                "{'ndxyz', 'cxyz', 'cxyz_ndxyz', 'nd123', 'c123', "
                "'c123_nd123'} (6 values, edge 10); post_processor "
                "thermo_heatflux accepts {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz'} (3 values, this edge). All three "
                "tools share the literal 'ndxyz' / 'cxyz' tokens "
                "but ONLY post_processor structure_stress accepts "
                "'nd123' / 'c123' eigen variants. Unknown enum "
                "values in thermo_heatflux trigger FOUR_C_THROW("
                "'Unknown heatflux/tempgrad type'). The 4 "
                "groupnames write_heatflux accepts are "
                "{gauss_{initial,current}_{heatfluxes,tempgrad}_xyz} "
                "and any other groupname FOUR_C_THROWs 'trying to "
                "write something that is not a heatflux or a "
                "temperature gradient'. The nodal-averaging loop "
                "(WriteNodalHeatfluxStep::operator()) divides each "
                "summed Gauss-point contribution by "
                "lnode->num_element() — boundary nodes (fewer "
                "adjacent elements) get the same divisor as "
                "interior nodes, which means the implied "
                "consistent-projection weights are wrong at the "
                "domain boundary. Result: visualized boundary "
                "heatfluxes are biased; the bias is largest for "
                "coarse meshes. "
                "(10) StructureFilter::post_stress dispatches on a "
                "6-VALUE stresstype enum {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz', 'nd123', 'c123', 'c123_nd123'} — "
                "WIDER than post_monitor's {'none', 'ndxyz'}. The two "
                "tools have ASYMMETRIC vocabularies and users "
                "routinely cross-confuse them. Any value outside the "
                "6-set FOUR_C_THROWs 'Unknown stress/strain type'. "
                "The '*_ndxyz' / '*_123' compound forms are DUAL-"
                "WRITE paths: write nodal first, PostResult reset, "
                "then element-center. Eigen variants (nd123/c123/"
                "c123_nd123) route to write_eigen_stress which emits "
                "<base>_eigenval{1,2,3} + <base>_eigenvec{1,2,3} per "
                "groupname (6 outputs per call). The eigen path is "
                "missing the 'rotation' groupname that write_stress "
                "supports — asking for principal rotation tensors is "
                "silently undefined. Also note write_eigen_stress's "
                "final else clause throws 'Unknown heatflux type' "
                "(line 636) — a verbatim copy-paste error from "
                "ThermoFilter, never updated; the message is "
                "misleading for structure dispatch. "
                "(File walks apps/post_processor/4C_post_processor.cpp + "
                "4C_post_processor_single_field_writers.cpp + "
                "4C_post_processor_structure_stress.cpp + "
                "4C_post_processor_thermo_heatflux.cpp + "
                "scripts/create_post_scripts.cmake + scripts/post_gid "
                "2026-06-03.)"
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL MECHANICS (structure, structure_new, solid_3D_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "structural_mechanics": {
        # ---- READ THIS FIRST ----
        "description": (
            "Linear and finite-deformation solid mechanics.\n"
            "  PROBLEMTYPE:      Structure\n"
            "  control section:  STRUCTURAL DYNAMIC\n"
            "  element section:  STRUCTURE ELEMENTS\n"
            "  3D element line:  <eid> SOLID <CELLTYPE> <nodes...> MAT <m> "
            "KINEM linear|nonlinear\n"
            "  2D element line:  4C before 2026.2.0: <eid> WALL <CELLTYPE> <nodes...> "
            "MAT <m> KINEM <k> EAS <e> THICK <t> STRESS_STRAIN <s> GP <a> <b>\n"
            "                    (ALL SIX keys; SOLID owns no 2D cell type there)\n"
            "                    4C 2026.2.0 and later: <eid> SOLID <CELLTYPE> <nodes...> "
            "MAT <m> KINEM <k> THICKNESS <t> PLANE_ASSUMPTION <s>\n"
            "                    (no WALL; `4C -p` of an older build lists WALL)\n"
            "  material:         MAT_Struct_StVenantKirchhoff with YOUNG, "
            "NUE, DENS (DENS required even under Statics)\n"
            "  convergence:      TOLDISP (update norm) and TOLRES (residual "
            "norm), both must be met\n"
            "  result check:     field group STRUCTURE, DIS: \"structure\", "
            "QUANTITY: dispx|dispy|dispz|...\n"
            "The deck below is complete and runs as written."
        ),
        "problemtype": "Structure",
        "yaml_section": "STRUCTURAL DYNAMIC",

        # ---- COMPLETE RUNNABLE DECK, 3D, no external files ----
        "minimal_working_input_3d": """\
# 3D cantilever under an end traction. The mesh is GENERATED - there is not
# a single node coordinate in this file.
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURE DOMAIN:
  bottom_corner_point: [0.0, 0.0, 0.0]   # REQUIRED
  top_corner_point: [10.0, 1.0, 1.0]     # REQUIRED
  subdivisions: [10, 2, 2]               # REQUIRED, elements per direction
  elements:                              # REQUIRED
    SOLID:
      HEX8:
        MAT: 1
        KINEM: nonlinear
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"    # Statics|GenAlpha|OneStepTheta|CentrDiff|...
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0              # truncates: stops at min(NUMSTEP*TIMESTEP, MAXTIME)
  TOLDISP: 1.0e-10          # update-norm tolerance
  TOLRES: 1.0e-09           # residual-norm tolerance
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3              # validated to lie in [-1, 0.5)
      DENS: 1.0             # REQUIRED even for Statics
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, -1.0]
    FUNCT: [0, 0, 1]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:        # symbolic faces of the generated box
  - "SIDE structure x- DSURFACE 1"
  - "SIDE structure x+ DSURFACE 2"
IO/RUNTIME VTK OUTPUT:      # OPTIONAL - drop both VTK sections and the
  INTERVAL_STEPS: 1         # deck still runs, it just writes no .vtu.
IO/RUNTIME VTK OUTPUT/STRUCTURE:   # But for .vtu you need BOTH sections AND
  OUTPUT_STRUCTURE: true           # at least one field flag; any one of the
  DISPLACEMENT: true               # three alone writes nothing.
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e30     # record mode: abs(diff) prints the true value
""",

        # ---- COMPLETE RUNNABLE DECK, 2D ----
        "minimal_working_input_2d": """\
# 4C before 2026.2.0 (WALL). For 4C 2026.2.0 and later use minimal_working_input_2d_solid.
# 2D plane-strain square, hand-written mesh. The box generator cannot make
# 2D cells, so 2D always uses the inline route.
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2               # 2 in 2D, not 3
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2                  # right edge: pulled in +x
    NUMDOF: 2
    ONOFF: [1, 0]
    VAL: [10.0, 0.0]
    FUNCT: [1, 0]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:              # in a 2D problem every z MUST be 0.0
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  # WALL, not SOLID, and all six keys are required
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispx"
      VALUE: 0.0
      TOLERANCE: 1.0e30   # record mode: abs(diff) prints the true value
""",

        # ---- THE SAME 2D DECK FOR 4C 2026.3.0 ----
        # Run on 4C 2026.2.0 and 2026.3.0: both record the same displacement as
        # the WALL deck on a 2026.2.0 development build (8.52459016393e-3,
        # relative difference 1e-14).
        "minimal_working_input_2d_solid": """\
# 4C 2026.2.0 and later (no WALL; the 2D element is SOLID with THICKNESS and
# PLANE_ASSUMPTION). 2D plane-strain square, hand-written mesh. The box generator cannot make
# 2D cells, so 2D always uses the inline route.
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2               # 2 in 2D, not 3
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2                  # right edge: pulled in +x
    NUMDOF: 2
    ONOFF: [1, 0]
    VAL: [10.0, 0.0]
    FUNCT: [1, 0]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:              # in a 2D problem every z MUST be 0.0
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  # 4C 2026.2.0 and later: SOLID with THICKNESS and PLANE_ASSUMPTION
  - "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM linear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispx"
      VALUE: 0.0
      TOLERANCE: 1.0e30   # record mode: abs(diff) prints the true value
""",

        # ---- CONVERGENCE: the keys and what they print ----
        "convergence_control": {
            "_note": (
                "Newton convergence in STRUCTURAL DYNAMIC is governed by a "
                "PAIR of tolerances, and by default BOTH must be satisfied "
                "(NORMCOMBI_RESFDISP: 'And'). Set 'Or' to stop as soon as "
                "either is met."
            ),
            "TOLDISP": (
                "Default 1e-10. Tolerance on the solution-update norm. "
                "Printed in the NOX status block as "
                "'Structure-Update-Norm = <value> < <your TOLDISP>'."
            ),
            "TOLRES": (
                "Default 1e-08. Tolerance on the residual force norm. "
                "Printed as 'Structure-F-Norm = <value> < <your TOLRES>'."
            ),
            "NORM_DISP": "Default 'Abs'. Choices: Abs, Rel, Mix. How TOLDISP is measured.",
            "NORM_RESF": "Default 'Abs'. Choices: Abs, Rel, Mix. How TOLRES is measured.",
            "NORMCOMBI_RESFDISP": "Default 'And'. Choices: And, Or. How the two are combined.",
            "ITERNORM": "Default 'L2'. Choices: L1, L2, Inf, Rms.",
            "MAXITER": "Default 50. Exhausting it aborts the run, it does not continue. NEVER set it to 1.",
            "LINEAR_SOLVER": (
                "REQUIRED IN PRACTICE. `4C --parameters` reports "
                "required:false with default -1, but -1 is not a solver id "
                "and setup rejects it. Verified by omission. Signal: 'no "
                "linear solver defined for structural field. Please set "
                "LINEAR_SOLVER in STRUCTURAL DYNAMIC to a valid number!' "
                "Every field has its own variant of this message naming its "
                "own section — ALE DYNAMIC, FLUID DYNAMIC, THERMAL DYNAMIC, "
                "CONTACT DYNAMIC, TSI DYNAMIC and so on — so the section it "
                "names is the section to fix."
            ),
            "MINITER": "Default 0.",
            "NLNSOL": (
                "Default 'fullnewton'. Choices: fullnewton, modnewton, "
                "lsnewton, ptc, noxnln, newtonlinuzawa, augmentedlagrange, "
                "singlestep."
            ),
            "DIVERCONT": (
                "Default 'stop' — a non-converged step aborts. Set "
                "'halve_step' or 'adapt_step' to let 4C retry with a smaller "
                "step instead of dying. Other choices: continue, repeat_step, "
                "repeat_simulation, rand_adapt_step, adapt_penaltycontact."
            ),
            "PREDICT": "Default 'ConstDis'. Choices: ConstDis, ConstVel, ConstAcc, TangDis, TangDisConstFext, ConstDisVelAcc.",
            "_failure_signal": (
                "When Newton runs out of iterations 4C prints "
                "'Failed.......Number of Iterations = <n> < <MAXITER>' in "
                "the final status block and then aborts with 'The nonlinear "
                "solver did not converge!' from "
                "solver_nonlin_nox/4C_solver_nonlin_nox_problem.cpp, exit 1."
            ),
        },

        # ---- STRUCT NOX: the section whose keys are not SCREAMING_SNAKE ----
        "struct_nox": {
            "_note": (
                "STRUCT NOX and its subsections configure the Trilinos NOX "
                "nonlinear solver. THEIR KEYS ARE CAPITALISED ENGLISH WORDS "
                "WITH SPACES, not SCREAMING_SNAKE like the rest of a 4C "
                "deck: 'Nonlinear Solver', 'Method', 'Max Iters', 'Outer "
                "Iteration'. Writing 'METHOD' instead of 'Method' is fatal. "
                "The section is optional; omit it entirely and 4C uses a "
                "line-search-based Newton with a full step."
            ),
            "STRUCT NOX": "Nonlinear Solver: Line Search Based | Trust Region Based | Inexact Trust Region Based | Tensor Based | Pseudo Transient | Single Step. Default 'Line Search Based'.",
            "STRUCT NOX/Direction": "Method: Newton | Steepest Descent | NonlinearCG | Broyden | User Defined. Default 'Newton'.",
            "STRUCT NOX/Line Search": "Method: Full Step | Backtrack | Polynomial | More'-Thuente | User Defined. Default 'Full Step'.",
            "STRUCT NOX/Line Search/Backtrack": "Default Step (1), Minimum Step (1e-12), Recovery Step (1), Max Iters (50), Reduction Factor (0.5), Allow Exceptions (false).",
            "STRUCT NOX/Printing": "Outer Iteration, Inner Iteration, Outer Iteration StatusTest, Details, Debug, Error, Warning, Parameters, Linear Solver Details, Test Details — all booleans. Set the first three false to quieten a long run.",
            "STRUCT NOX/Pseudo Transient": "deltaInit, deltaMax, deltaMin, Max Number of PTC Iterations, SER_alpha, ScalingFactor, Time Step Control, Norm Type for TSC, Scaling Type, Build scaling operator.",
            "STRUCT NOX/Solver Options": "Merit Function ('Sum of Squares'), Status Test Check Type (Complete | Minimal | None).",
            "STRUCT NOX/Status Test": "XML File: path to a NOX status-test XML.",
            "_example": (
                "STRUCT NOX:\\n"
                "  Nonlinear Solver: \"Line Search Based\"\\n"
                "STRUCT NOX/Direction:\\n"
                "  Method: \"Newton\"\\n"
                "STRUCT NOX/Line Search:\\n"
                "  Method: \"Backtrack\"\\n"
                "STRUCT NOX/Line Search/Backtrack:\\n"
                "  Max Iters: 30\\n"
                "  Reduction Factor: 0.5"
            ),
        },

        # ---- ELEMENT VOCABULARY, required keys stated ----
        "structure_elements_section": {
            "_note": (
                "Section 'STRUCTURE ELEMENTS' is a list of quoted strings, "
                "one per element:\n"
                "  <eid> <ELETYPE> <CELLTYPE> <node ids...> <KEY value>...\n"
                "It goes together with 'NODE COORDS' and the "
                "'D*-NODE TOPOLOGY' sections. Use it OR "
                "'STRUCTURE GEOMETRY' (Exodus) OR 'STRUCTURE DOMAIN' (box "
                "generator) — never two of them."
            ),
            "SOLID": {
                "cell_types": ["HEX8", "HEX18", "HEX20", "HEX27", "TET4",
                               "TET10", "WEDGE6", "PYRAMID5", "NURBS27"],
                "required_keys": ["MAT", "KINEM"],
                "KINEM": "linear | nonlinear",
                "optional_keys": {
                    "TECH": (
                        "ONLY on HEX8, WEDGE6, PYRAMID5, and with different "
                        "choices each: HEX8 -> none|fbar|eas_mild|eas_full|"
                        "shell_ans|shell_eas|shell_eas_ans; WEDGE6 -> none|"
                        "shell_ans|shell_eas_ans; PYRAMID5 -> none|fbar. "
                        "Writing TECH on any other cell type is fatal."
                    ),
                    "PRESTRESS_TECH": "none | mulf",
                    "RAD / AXI / CIR": "3-vectors, cylindrical fibre frame",
                    "FIBER1 / FIBER2 / FIBER3": "3-vectors, anisotropy directions",
                    "INTEGRATION": (
                        "Sub-group with RESIDUUM and MASS Gauss rules. "
                        "USABLE ONLY where the element is written as a YAML "
                        "MAP — i.e. inside 'ELEMENT_BLOCKS' (Exodus route) "
                        "or inside 'STRUCTURE DOMAIN: elements:' (box "
                        "generator). NEVER on a whitespace-separated "
                        "'STRUCTURE ELEMENTS' string line, in any syntax. "
                        "YAML-map form: 'INTEGRATION: {RESIDUUM: "
                        "hex_27point, MASS: hex_8point}'. The Gauss-rule "
                        "names are validated per cell family (hex_*, tet_*, "
                        "wedge_*, pyramid_*) and a mismatched one is "
                        "rejected."
                    ),
                },
                "example": '"1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear TECH eas_full"',
            },
            "WALL": {
                "_note": "The 2D solid element of 4C before 2026.2.0; SOLID owns no 2D cell type there. 4C 2026.2.0 and later have no WALL: see 'SOLID (2D, 4C 2026.2.0 and later)'.",
                "cell_types": ["QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6"],
                "_cell_types_that_do_NOT_work": (
                    "`4C --parameters` also lists NURBS4 and NURBS9 under "
                    "WALL, but they are not registered: the element type "
                    "for those is WALLNURBS, and asking for them under WALL "
                    "gives the misleading \"Unknown type 'WALL' of finite "
                    "element\"."
                ),
                "required_keys": ["MAT", "KINEM", "EAS", "THICK",
                                  "STRESS_STRAIN", "GP"],
                "KINEM": "linear | nonlinear",
                "EAS": "none | full. 'full' is 4-node only; on TRI3 it gives 'eas-technology not implemented for tri3 elements'.",
                "STRESS_STRAIN": "plane_strain | plane_stress",
                "GP": (
                    "Two integers. For QUADs it is Gauss points per "
                    "direction, e.g. 'GP 2 2' (QUAD9 wants 'GP 3 3'). For "
                    "TRI3/TRI6 the SECOND number must be 0 — 'GP 3 0', not "
                    "'GP 3 3', which aborts with 'Unknown number of Gauss "
                    "points for tri element'."
                ),
                "example": '"1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"',
            },
            "SOLID (2D, 4C 2026.2.0 and later)": {
                "_note": "The 2D solid element of 4C 2026.2.0 and later (earlier builds have WALL instead). Each cell type below ran on 4C 2026.2.0 and on 2026.3.0.",
                "cell_types": ["QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6"],
                "required_keys": ["MAT", "KINEM", "THICKNESS", "PLANE_ASSUMPTION"],
                "KINEM": "linear | nonlinear",
                "THICKNESS": "Out-of-plane thickness.",
                "PLANE_ASSUMPTION": "plane_strain | plane_stress",
                "TECH": (
                    "QUAD4 only; the other 2D cells take no TECH key. none "
                    "(default) or eas_full, and eas_full needs KINEM "
                    "nonlinear. KINEM linear with eas_full, and eas_mild or "
                    "fbar with either KINEM, stop with 'Your element "
                    "formulation with cell type QUAD4, kinematic type <k>, "
                    "element technology <t> and prestress type none does not "
                    "exist'."
                ),
                "example": '"1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"',
            },
            "SOLIDSCATRA": {
                "cell_types": ["HEX8", "HEX27", "TET4", "TET10", "NURBS27"],
                "required_keys": ["MAT", "KINEM", "TYPE"],
                "_note": "Structure element carrying a scalar field; used by TSI and SSI.",
            },
            "_error_signals": {
                "_note": (
                    "These are what 4C PRINTS. In the binary they are fmt "
                    "templates with '{}' where the value goes, so a "
                    "`strings | grep -F` check must be run against the "
                    "TEMPLATE (\"Required value '{}' not found in input "
                    "line\"), not against the rendered message. Note also "
                    "that a near-identical sibling exists for the one_of "
                    "case, \"Required '{}' not found in input line\"."
                ),
                "missing required key": "Required value 'KINEM' not found in input line",
                "cell type not owned by that element": "Element 'SOLID' does not seem to know cell type 'quad4'. (cell type echoed LOWERCASE)",
                "element type does not exist": "Unknown type 'BOGUS' of finite element",
                "surplus / unsupported key on the line": "After parsing, the line still contains '<token>'. — followed by a 'Parsed parameters:' dump of what it did accept",
                "bad value for an enum key": "Could not parse parameter 'TECH': invalid value 'fbar'. Valid options are: none|shell_ans|shell_eas_ans",
                "INTEGRATION on an inline line, both sub-keys given": "Key 'INTEGRATION' cannot be found in the container.",
                "INTEGRATION on an inline line, PARTIAL (one sub-key or bare)": (
                    "NO 4C diagnostic at all. The process aborts with "
                    "\"terminate called after throwing an instance of "
                    "'std::bad_any_cast'\" and shell exit status 134. "
                    "There is no 'PROC 0 ERROR' block to grep for. Note "
                    "that this line comes from the C++ RUNTIME's "
                    "terminate handler in libstdc++, not from 4C: "
                    "`strings` on the 4C binary will NOT find it, and "
                    "that absence is expected rather than evidence the "
                    "message cannot occur."
                ),
                "cell type parses but the physics module has no implementation": "Element shape TRI6 (6 nodes) not activated. Just do it.",
                "triangle Gauss rule written as <n> <n>": "Unknown number of Gauss points for tri element",
            },
        },

        "time_integration": {
            "Statics": "Static analysis (one step, equilibrium)",
            "GenAlpha": "Generalized-alpha implicit time integration",
            "GenAlphaLieGroup": "Generalized-alpha for SO(3) rotation group (beams/shells)",
            "OneStepTheta": "One-step-theta implicit scheme",
            "ExplicitEuler": "Explicit forward Euler (the enum is ExplicitEuler; 'ExplEuler' is rejected)",
            "CentrDiff": "Explicit central differences (wave propagation)",
            "AdamsBashforth2": "Explicit 2nd order Adams-Bashforth",
            "AdamsBashforth4": "Explicit 4th order Adams-Bashforth",
        },

        "kinematics": {
            "linear": "Small strain / linear kinematics (ε = sym(∇u))",
            # The KINEM key on an element line takes 'linear' or
            # 'nonlinear' and NOTHING else. 'nonlinearTotLag' is what 4C
            # echoes back internally after parsing 'nonlinear'; writing it
            # is rejected.
            "nonlinear": "Total Lagrangian / finite deformation (F = I + grad u). Write 'KINEM nonlinear'; 'nonlinearTotLag' is rejected.",
        },

        "element_types": {
            "3D_solid": {
                "SOLID HEX8": "8-node hexahedron (Q1, standard or F-bar or EAS)",
                "SOLID HEX20": "20-node hexahedron (serendipity Q2)",
                "SOLID HEX27": "27-node hexahedron (full Q2)",
                "SOLID TET4": "4-node tetrahedron (P1)",
                "SOLID TET10": "10-node tetrahedron (P2)",
                "SOLID WEDGE6": "6-node wedge/prism",
                "SOLID HEX18": "18-node hexahedron",
                                "SOLID PYRAMID5": "5-node pyramid",
                "SOLIDSCATRA HEX8": "8-node hex with scalar transport coupling (for TSI)",
            },
            "2D_wall": {
                "_which_element_type": (
                    "WHICH element type owns 2D structural cells is "
                    "VERSION-DEPENDENT and the two spellings share no "
                    "keywords, so they cannot be mixed:\n"
                    "  WALL  <cell> <nodes> MAT m KINEM k EAS e THICK t "
                    "STRESS_STRAIN s GP a b\n"
                    "  SOLID <cell> <nodes> MAT m KINEM k THICKNESS t "
                    "PLANE_ASSUMPTION p\n"
                    "Exactly one of them is registered in any given build. "
                    "Decide it, do not guess it: `4C --parameters` lists the "
                    "cell types each element factory owns. If SOLID's cell "
                    "list is 3D-only (HEX/TET/WEDGE/PYRAMID), 2D belongs to "
                    "WALL and 'SOLID QUAD4' aborts with \"Element 'SOLID' "
                    "does not seem to know cell type 'quad4'.\"; if SOLID "
                    "lists QUAD4/TRI3 then 'WALL' aborts with \"Unknown "
                    "type 'WALL' of finite element\". Measured: a 2026.2.0 "
                    "development build registers WALL for 2D, the 2026.2.0 and "
                    "2026.3.0 releases the 2D SOLID and no WALL."
                ),
                "WALL QUAD4": "4C before 2026.2.0: 4-node quadrilateral. Six required keys: MAT, KINEM, EAS, THICK, STRESS_STRAIN, GP.",
                "WALL QUAD8": "4C before 2026.2.0: 8-node serendipity quad, same six required keys.",
                "WALL QUAD9": "4C before 2026.2.0: 9-node full biquadratic quad, same six required keys.",
                "WALL TRI3":  "4C before 2026.2.0: 3-node triangle, same six required keys.",
                "WALL TRI6":  "4C before 2026.2.0: 6-node quadratic triangle, same six required keys.",
                "SOLID QUAD4 / QUAD8 / QUAD9 / TRI3 / TRI6": "4C 2026.2.0 and later: the same cells under SOLID. Four required keys: MAT, KINEM, THICKNESS, PLANE_ASSUMPTION.",
                "WALL NURBS4 / NURBS9": "4C before 2026.2.0: DO NOT USE. `4C --parameters` lists these under WALL but they are not registered; the element type is WALLNURBS, and asking for them under WALL gives the misleading \"Unknown type 'WALL' of finite element\".",
            },
            "1D_beam": {
                "BEAM3R": "Simo-Reissner beam (shear-deformable, geometrically exact)",
                "BEAM3K": "Kirchhoff beam (shear-rigid, inextensible option)",
                "BEAM3EB": "Euler-Bernoulli beam (classical)",
            },
            "shell": {
                "SHELL7P": "7-parameter shell (EAS, ANS options, thickness locking-free)",
                "SHELL_KIRCHHOFF_LOVE_NURBS": "Kirchhoff-Love NURBS shell (isogeometric). Spelled out in full; 'SHELL_KL_NURBS' is not registered.",
            },
            "other": {
                "MEMBRANE3 / MEMBRANE4 / MEMBRANE6 / MEMBRANE9": "Membrane elements, no bending stiffness. The node count is part of the element NAME; a bare 'MEMBRANE' is not registered.",
                "TRUSS3": "Truss element (axial force only)",
                "TORSION3": "Torsional spring element",
                "RIGIDSPHERE": "Rigid sphere for DEM contact",
            },
        },

        "element_technologies": {
            "none": "Standard displacement-based formulation",
            "fbar": "F-bar method (volumetric locking treatment for hex8)",
            "eas_mild": "Enhanced Assumed Strain (mild enrichment, 9 modes for hex8)",
            "eas_full": "Enhanced Assumed Strain (full enrichment, 21 modes for hex8)",
            "shell_ans": "Assumed Natural Strain for shells (shear locking treatment)",
            "shell_eas": "EAS for shells",
            "shell_eas_ans": "Combined EAS + ANS for shells",
        },

        # STRUCTURAL DYNAMIC/NLNSOL. These are the EXACT enum spellings;
        # the plausible-looking variants newtonfull / newtonmod / newtonls /
        # newtonuzawalin / nox_nln are NOT accepted.
        "nonlinear_solvers": {
            "fullnewton": "Full Newton-Raphson (assemble tangent every iteration). The default.",
            "modnewton": "Modified Newton (reuse tangent, cheaper per iteration)",
            "lsnewton": "Newton with line search (backtracking)",
            "newtonlinuzawa": "Linear Uzawa for constrained problems",
            "augmentedlagrange": "Augmented-Lagrange solver",
            "ptc": "Pseudo-transient continuation (robust for difficult convergence)",
            "noxnln": "NOX nonlinear solver framework (Trilinos); configure it via the STRUCT NOX sections",
            "singlestep": "Single-step (no iteration)",
            "vague": "Unset sentinel; do not select",
        },

        # Full 2D element line, so it can be copied rather than assembled.
        # See structure_elements_section/WALL above for the key rules.
        "wall_element_params": (
            "4C before 2026.2.0. A complete WALL line, in section STRUCTURE ELEMENTS:\n"
            '  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none '
            'THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"\n'
            "All six keys are required. KINEM linear|nonlinear; EAS "
            "none|full (full is 4-node only); THICK is the out-of-plane "
            "thickness; STRESS_STRAIN plane_strain|plane_stress; GP is two "
            "integers, '2 2' for QUAD4, '3 3' for QUAD9, and '<n> 0' for "
            "TRI3/TRI6."
        ),
        "solid_2d_element_params": (
            "4C 2026.2.0 and later (no WALL). A complete 2D SOLID line, in section "
            "STRUCTURE ELEMENTS:\n"
            '  - "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear THICKNESS 1.0 '
            'PLANE_ASSUMPTION plane_strain"\n'
            "All four keys are required. KINEM linear|nonlinear; THICKNESS "
            "is the out-of-plane thickness; PLANE_ASSUMPTION "
            "plane_strain|plane_stress; QUAD4 alone takes TECH, none or "
            "eas_full (eas_full needs KINEM nonlinear)."
        ),

        "pitfalls": [
            (
                "[Input] STRUCTURE ELEMENTS: The 2D structural element type is "
                "VERSION-DEPENDENT and the two spellings share no keywords, "
                "so you cannot hedge by writing both:\n"
                "  WALL  QUAD4 <n..> MAT m KINEM k EAS e THICK t "
                "STRESS_STRAIN s GP a b\n"
                "  SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t "
                "PLANE_ASSUMPTION p\n"
                "Determine which one this build registers before writing "
                "anything: `4C --parameters` lists, per element type, the "
                "cell types it owns. If SOLID's list is 3D-only, 2D is "
                "WALL's (measured: a 2026.2.0 development build registers WALL, "
                "the 2026.2.0 and 2026.3.0 releases the 2D SOLID and no WALL). "
                "Signal: both were confirmed by "
                "triggering them - the "
                "wrong 2D element type gives \"Element 'SOLID' does not "
                "seem to know cell type 'quad4'.\" (note the cell type is "
                "echoed in LOWERCASE), and an element type this build does "
                "not register at all gives \"Unknown type 'WALL' of finite "
                "element\". A keyword the element does not own gives "
                "\"After parsing, the line still contains '<token>'\" — not "
                "'unknown parameter', which is not the wording 4C uses "
                "here. See Tier-2 fixture "
                "structural_2d_solid_quad4_not_wall, which probes both "
                "spellings and asserts the era-agnostic invariant rather "
                "than hard-coding either."
            ),
            (
                "[Input] STRUCTURE ELEMENTS (the WALL of 4C before 2026.2.0): WALL TRI3 and TRI6 take GP as "
                "'<n> 0', NOT '<n> <n>'. 'GP 3 3' on a triangle is fatal. "
                "Signal: 'Unknown number of Gauss points for tri element' "
                "from w1/4C_w1_input.cpp. 'GP 3 0' on the same element runs. "
                "EAS is likewise 4-node-only: 'EAS full' on TRI3 gives "
                "'eas-technology not implemented for tri3 elements'."
            ),
            (
                "[Input] STRUCTURE ELEMENTS (4C before 2026.2.0): WALL does NOT own NURBS4/NURBS9 even "
                "though `4C --parameters` lists them under WALL. The "
                "registered element type for isogeometric 2D cells is "
                "WALLNURBS. Signal: \"Unknown type 'WALL' of finite "
                "element\" — a misleading message, since WALL itself is a "
                "perfectly known element type; what is unregistered is the "
                "WALL-plus-NURBS combination. Any NURBS "
                "element additionally needs PROBLEM TYPE/SHAPEFCT: "
                "\"Nurbs\", a '<DIS> KNOTVECTORS' section, and control "
                "points written as 'CP <id> COORD x y z <weight>' inside "
                "NODE COORDS. Writing plain 'NODE' lines with a NURBS "
                "element SEGFAULTS — exit status 139, no 4C error block, no "
                "message at all."
            ),
            (
                "[Input] STRUCTURE ELEMENTS: KINEM takes exactly two values on "
                "this build: 'linear' or 'nonlinear'. Do NOT write "
                "'nonlinearTotLag' — it is what 4C reports INTERNALLY once "
                "it has parsed 'nonlinear' (you see it echoed in the "
                "'Parsed parameters:' dump), but it is not accepted as "
                "input. Signal: \"Could not parse parameter 'KINEM': "
                "invalid value 'nonlinearTotLag'. Valid options are: "
                "linear|nonlinear\". So 'KINEM nonlinear' already IS the "
                "total-Lagrangian formulation; there is no separate "
                "updated-Lagrangian spelling to avoid."
            ),
            (
                "[Input] SOLIDSCATRA element is REQUIRED for "
                "TSI coupling — plain SOLID cannot couple "
                "with the thermal field. Signal: 'Unsupported "
                "solid element type!' from tsi/4C_tsi_utils.cpp. "
                "(An earlier version of this entry quoted 'no SCATRA "
                "discretisation found' from '4C_tsi_factory.cpp'; "
                "THAT STRING IS NOT IN THE BINARY and there is no file "
                "of that name. Corrected against a real run that swapped "
                "SOLIDSCATRA for SOLID in a working TSI deck.) the "
                "structure has no SCATRA-side mass matrix "
                "to clone into a thermal discretisation. "
                "Replace 'SOLID' with 'SOLIDSCATRA' for "
                "all elements in the structural mesh. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] NEVER set MAXITER = 1, not even for a "
                "linear problem. Exhausting MAXITER is an ABORT, not an "
                "early exit, and the counter reaches 1 before the "
                "convergence test is credited — so MAXITER: 1 kills a "
                "perfectly converged linear deck whose residual is already "
                "at 1e-12. Leave it at the default 50, or set 10-30. There "
                "is no cost to a generous MAXITER: Newton stops at the "
                "tolerance, not at the cap. Signal: 'Failed.......Number of "
                "Iterations = 1 < 1' in the final status block followed by "
                "'The nonlinear solver did not converge!' and exit 1. "
                "(An earlier version of this entry RECOMMENDED MAXITER = 1 "
                "for linear problems; both KINEM linear and KINEM nonlinear "
                "decks were run with it and both aborted. Corrected by "
                "execution.)"
            ),
            (
                "[Numerical] PREDICT: TangDis is "
                "RECOMMENDED for nonlinear Newton "
                "convergence — uses the tangent "
                "stiffness to predict the next "
                "iterate. Signal: PREDICT: ConstDis "
                "(constant displacement) on a "
                "geometrically-nonlinear problem "
                "costs a few more Newton iterations per "
                "step than TangDis, because the tangent predictor starts "
                "closer to equilibrium. The gap is modest — a step or two — "
                "not a change of order, so reach for TangDis to trim "
                "iterations, not to rescue a diverging solve. (Audit "
                "2026-06-02; the original entry claimed 5-10 versus 2-3, "
                "which a direct comparison on a 3D deck did not support.)"
            ),
            (
                "[Input] Body forces go in DESIGN SURF NEUMANN (2D) or "
                "DESIGN VOL NEUMANN (3D), with 3 components in 3D and 2 in "
                "2D; NUMDOF = 6 is the beam/shell case (3 forces + 3 "
                "moments). But 4C does NOT enforce that count on a solid: a "
                "3D Neumann block written with NUMDOF: 6 and six-entry "
                "arrays runs to completion, the extra entries simply unused. "
                "What IS enforced is internal consistency — ONOFF, VAL and "
                "FUNCT must each have exactly NUMDOF entries. Signal: an "
                "inconsistent block is rejected at parse with 'Could not "
                "match this input' plus an echo of the block; an oversized "
                "but self-consistent block produces NO diagnostic at all, so "
                "getting the count wrong is a silent modelling error rather "
                "than a caught one. (An earlier version quoted 'NUMDOF "
                "mismatch — expected 3 got 6'; THAT STRING IS NOT IN THE "
                "BINARY and the deck exits 0. Corrected by execution.)"
            ),
            (
                "[API] Beam elements need SPECIAL BEAM3* "
                "type — NOT SOLID or WALL. Signal: "
                "writing 'SOLID LINE2' or 'WALL LINE2' "
                "for beam elements raises \"Element 'SOLID' does not seem "
                "to know cell type 'line2'.\" from "
                "fem/general/element/4C_fem_general_element_definition.cpp "
                "— the SOLID/WALL (WALL: 4C before 2026.2.0) factories register volume/surface cell "
                "types only, so the ELEMENT TYPE is recognised and the CELL "
                "TYPE is not. Use BEAM3R / BEAM3K / BEAM3EB with the "
                "appropriate LINE2/LINE3 cell type and TRIADS. (An earlier "
                "version attributed this to 'Unknown type' from "
                "parobjectfactory.cpp; that path is never reached here, "
                "because the element type itself does exist. Corrected by "
                "execution.)"
            ),

            # ───────────────────────────────────────────────────
            # 2026-08-03 EXECUTION SWEEP on the DEPLOYED binary
            # deployed 4C binary = 4C 2026.2.0-dev,
            # git 89519cf. Where these disagree with the entries
            # above (which were written against 4C 2026.3.0-dev)
            # the disagreement is a VERSION boundary, not a
            # correction — both spellings are kept and the probe
            # that tells them apart is given.
            # ───────────────────────────────────────────────────
            (
                "[API] The 2D structural element name is "
                "VERSION-DEPENDENT and the two spellings share no "
                "keywords. On a development build of March 2026 "
                "(it reports 2026.2.0-dev; the 2026.2.0 release "
                "already has the 2-D SOLID) the SOLID factory "
                "registers 3D cell types ONLY (hex8, hex18, hex20, "
                "hex27, tet4, tet10, wedge6, pyramid5, nurbs27) and "
                "2D lives in the separate WALL factory, whose input "
                "line is '<id> WALL QUAD4 <n1..n4> MAT <m> KINEM "
                "<linear|nonlinear> EAS <none|full> THICK <t> "
                "STRESS_STRAIN <plane_strain|plane_stress> GP <a> "
                "<b>'. All six of MAT / KINEM / EAS / THICK / "
                "STRESS_STRAIN / GP are required, GP is a "
                "two-integer vector, and there is no THICKNESS or "
                "PLANE_ASSUMPTION keyword. From the 4C 2026.2.0 release on, the 2D "
                "families were folded into SOLID with the "
                "THICKNESS / PLANE_ASSUMPTION spelling instead. "
                "Decide which build you are on before writing a 2D "
                "deck — `4C --parameters` lists the element under "
                "legacy_element_specs with its exact required "
                "parameters. Signal: 'Element 'SOLID' does not seem "
                "to know cell type 'quad4'.' from "
                "fem_general_element_definition means you are on a "
                "WALL-era build; \"Unknown type 'WALL' of finite "
                "element\" from parobjectfactory means you are on a "
                "SOLID-era build; \"Required value 'GP' not found in "
                "input line\" from input_spec_builders means WALL "
                "was right but the GP pair was omitted. (Verified by "
                "execution 2026-08-03: on that development build the full "
                "WALL QUAD4 line ran to exit 0, 'SOLID QUAD4' failed "
                "with the cell-type message with and without DIM: 2, "
                "and the upstream deck tests/input_files/"
                "test_struct.4C.yaml — one of only 3 of the 1974 "
                "decks that use SOLID QUAD4, against 109 that use "
                "WALL — fails on this build for the same reason.)"
            ),
            (
                "[Numerical] On the WALL element EAS and "
                "STRESS_STRAIN are not cosmetic — both change the "
                "answer by percent-level amounts. STRESS_STRAIN "
                "does so SILENTLY, with no warning at all. EAS is "
                "different and the distinction matters: EAS is "
                "legal ONLY with KINEM nonlinear. 'EAS full' "
                "together with 'KINEM linear' is a HARD ERROR, not "
                "a silent change — so the EAS trap is a run that "
                "stops, while the STRESS_STRAIN trap is a run that "
                "lies. Signal: for a fixed 2D cantilever deck at "
                "KINEM nonlinear, switching EAS none -> full moved "
                "the tip deflection by 23% with no diagnostic, and "
                "plane_strain -> plane_stress moved it by 6.7%; if "
                "a 2D result disagrees with a reference by a stable "
                "few percent with no convergence trouble, compare "
                "these two keywords before refining the mesh. The "
                "loud branch is 'ERROR: No EAS for geometrically "
                "linear WALL element' from w1_input (4C_w1_input.cpp), "
                "exit 1. (Verified by execution 2026-08-03 on 4C "
                "2026.2.0-dev git 89519cf, single WALL QUAD4 "
                "1x1 unit square, YOUNG 1000 NUE 0.3 THICK 1.0 "
                "GP 2 2, line Neumann VAL 1 in y, dispy at the "
                "loaded corner node 2 — all comparisons at the SAME "
                "node and the SAME kinematics: KINEM nonlinear EAS "
                "none / plane_strain 4.33567955849997223e-03 vs "
                "EAS full / plane_strain 5.33749404753981662e-03, "
                "i.e. +23.1%, both exit 0; KINEM linear EAS none / "
                "plane_strain 4.33333333333345890e-03 vs "
                "plane_stress 4.62222222222249089e-03, i.e. +6.7%, "
                "both exit 0; KINEM linear with EAS full exited 1 "
                "with the w1_input throw for both STRESS_STRAIN "
                "settings.)"
            ),
            (
                "[Input] DENS is a REQUIRED parameter of "
                "MAT_Struct_StVenantKirchhoff in every analysis "
                "type, including Statics — it has no default in "
                "global_legacy_module_validmaterials. The often-"
                "repeated advice that density 'is only needed for "
                "dynamics' is about whether the value matters, not "
                "about whether the key may be omitted. Signal: "
                "omitting DENS gives \"Failed to match "
                "specification in section 'MATERIALS'\" from "
                "global_data_read plus \"Expected parameter "
                "'DENS'\" from input_spec_builders, exit 1, even "
                "for DYNAMICTYPE: Statics. (Verified by execution "
                "2026-08-03: a Statics HEX8 deck whose "
                "MAT_Struct_StVenantKirchhoff carried only YOUNG "
                "and NUE.)"
            ),
            (
                "[Input] NUE is validated against the half-open "
                "range [-1, 0.5) — the incompressible limit NUE: "
                "0.5 is REJECTED, not merely ill-conditioned. Use "
                "0.4999 with a locking-free element technology, or "
                "a MAT_ElastHyper split with a volumetric summand, "
                "for near-incompressible behaviour. Signal: "
                "\"Candidate parameter 'NUE' does not pass "
                "validation: in_range[-1,0.5)\" inside the "
                "'Failed to match specification in section "
                "'MATERIALS'' report from global_data_read, exit 1. "
                "(Verified by execution 2026-08-03: NUE 0.49 ran to "
                "exit 0; NUE 0.5 and NUE 0.6 were both rejected at "
                "parse.)"
            ),
            (
                "[Numerical] MAT_ElastHyper combined with KINEM "
                "linear is accepted by both the parser and the "
                "element factory, and it fails in two different "
                "ways depending on how far the structure deforms: "
                "at moderate strain it just returns quietly wrong "
                "numbers, and past roughly unit stretch it dies "
                "with SIGFPE because the linearised right "
                "Cauchy-Green tensor stops being admissible for "
                "the invariant-based strain-energy evaluation. "
                "Neither failure names KINEM. Signal: a "
                "hyperelastic run that differs from its KINEM "
                "nonlinear twin by a few percent and grows with "
                "load is the quiet branch; the loud branch is "
                "'Signal: Floating point exception (8)' / 'Signal "
                "code: Invalid floating point operation (7)' with "
                "shell exit status 136 and a stack whose top "
                "frames are Mat::ElastHyper::evaluate and "
                "DisplacementBasedLinearKinematicsFormulation — "
                "the FE-trapping build option turns the NaN into a "
                "hard kill instead of a NaN-filled result file. "
                "(Verified by execution 2026-08-03, HEX8 unit cube, "
                "CoupNeoHooke + VolSussmanBathe: tip dispy at load "
                "50 was 2.04240098889810623e-01 linear vs "
                "2.07913934875450096e-01 nonlinear (1.8% low), at "
                "load 100 4.08487004522754327e-01 vs "
                "4.20495844658128337e-01 (2.9% low), both exit 0; "
                "at load 300 the linear variant was killed by "
                "SIGFPE while the nonlinear variant finished at "
                "1.19488815226590517e+00.)"
            ),
            (
                "[Numerical] KINEM linear on an ordinary "
                "St-Venant-Kirchhoff solid never errors — it "
                "simply drops the geometric nonlinearity and "
                "over-predicts compliance, and the run looks "
                "perfectly healthy. There is no diagnostic to grep "
                "for, so the only defence is a converged "
                "comparison. Signal: run the identical deck with "
                "KINEM nonlinear; a stable relative difference "
                "that grows with load magnitude (and vanishes as "
                "load -> 0) is the signature of a wrongly linear "
                "kinematics choice rather than of a mesh or solver "
                "problem. Note the difference is not monotone in "
                "sign: at very small load it can go either way and "
                "it is node-dependent, so compare at a load that "
                "actually produces finite strain, and compare the "
                "SAME node. (Verified by execution 2026-08-03 on 4C "
                "2026.2.0-dev git 89519cf, single-HEX8 unit cube, "
                "YOUNG 1000 NUE 0.3, surface Neumann in y, dispy at "
                "node 6: at a small load the two agreed to 0.06% "
                "(linear 4.48190476190476333e-03 vs nonlinear "
                "4.48471241704061219e-03 — linear is the SMALLER "
                "one here); at a 300x load KINEM linear "
                "over-predicted by 5.7% (1.34457142857142875e+00 vs "
                "1.27200961795990475e+00). Both exited 0 with no "
                "warning. Reading the same runs at node 3 instead "
                "gives +81.8% at the large load, which is why the "
                "probe node has to be stated for the number to mean "
                "anything.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MATERIALS (120+ models)
    # ═══════════════════════════════════════════════════════════════════════
    "materials": {
        "description": "120+ material models spanning all physics disciplines",

        "basic_structural": {
            "MAT_Struct_StVenantKirchhoff": {
                "params": "YOUNG, NUE, DENS",
                "use": "Linear elastic (small strain) or geometric nonlinear",
            },
            "MAT_Struct_ThermoStVenantK": {
                "params": "YOUNGNUM (int, required: the number of YOUNG values), YOUNG (list of YOUNGNUM values), NUE, DENS, THEXPANS, INITTEMP (4C before 2026.3.0 also: THERMOMAT, optional, default -1)",
                "use": "Linear elastic with thermal expansion coupling (for TSI)",
                "notes": "Before 4C 2026.3.0 only: THERMOMAT can link a MAT_Fourier; 4C 2026.3.0 removed the key. The thermal field takes its MAT_Fourier from the CLONING MATERIAL MAP in both",
            },
        },

        "hyperelastic": {
            "MAT_ElastHyper": "Toolbox: combine summands (NeoHooke + volumetric, etc.)",
            # Each summand is its own MATERIALS entry, listed in the MATIDS of
            # MAT_ElastHyper / MAT_ViscoElastHyper.
            "summands": {
                "ELAST_CoupNeoHooke": "Neo-Hooke (coupled form; YOUNG, NUE): W = c*(I1-3) + c/beta*(J^(-2*beta)-1), beta = nu/(1-2*nu)",
                "ELAST_CoupLogNeoHooke": "Logarithmic Neo-Hooke: W = mu/2*(I1-3) - mu*ln(J) + lam/2*ln(J)^2",
                "ELAST_CoupMooneyRivlin": "Mooney-Rivlin (coupled; C1, C2, C3): W = c1*(I1-3) + c2*(I2-3) - (2*c1+4*c2)*ln(J) + c3*(J-1)^2",
                "ELAST_IsoNeoHooke": "Isochoric Neo-Hooke (incompressible split)",
                "ELAST_IsoOgden": "Isochoric Ogden (stretch-based)",
                "ELAST_IsoYeoh": "Isochoric Yeoh (polynomial in I1)",
                "ELAST_CoupBlatzKo": "Blatz-Ko (compressible rubber-like)",
                "ELAST_CoupSimoPister": "Simo-Pister model",
                "ELAST_CoupAnisoExpo": "Anisotropic exponential fiber model (soft tissue)",
                "ELAST_CoupAnisoNeoHooke": "Anisotropic Neo-Hooke fiber",
            },
            "volumetric": {
                "ELAST_VolOgden": "Ogden volumetric penalty",
                "ELAST_VolPenalty": "Penalty (EPSILON, GAMMA): W = eps*(J^gamma + J^(-gamma) - 2)",
                "ELAST_VolSussmanBathe": "Sussman-Bathe volumetric",
            },
        },

        "viscoelastic": {
            "MAT_ViscoElastHyper": "Viscohyperelastic with Maxwell branches",
            "VISCO_GenMax": "Standard Linear Solid visco summand (before 4C 2026.3.0)",
            "VISCO_GeneralizedGenMax": "Generalized Maxwell branches (before 4C 2026.3.0; 4C 2026.3.0 has VISCO_GeneralizedMaxwell with VISCO_GeneralizedMaxwellBranch entries instead)",
            "VISCO_Fract": "Fractional Standard Linear Solid (before 4C 2026.3.0; 4C 2026.3.0 has VISCO_FSLS instead)",
        },

        "plasticity": {
            "MAT_Struct_PlasticLinElast": "Small-strain von Mises plasticity (YOUNG, NUE, DENS, YIELD, ISOHARD, KINHARD, TOL)",
            "MAT_Struct_PlasticNlnLogNeoHooke": "Finite strain von Mises + logarithmic Neo-Hooke",
            "MAT_Struct_DruckerPrager": "Drucker-Prager plasticity (pressure-dependent yield)",
            "MAT_Struct_PlasticGTN": "Gurson-Tvergaard-Needleman (ductile damage)",
            "MAT_crystal_plasticity": "Crystal plasticity (single crystal, multiple slip systems)",
            "MAT_PlasticElastHyper": "Hyperelastic + finite strain von Mises (semi-smooth Newton)",
        },

        "biological": {
            "MAT_ConstraintMixture": "Constrained mixture model for arterial growth/remodeling",
            "MAT_GrowthRemodel_ElastHyper": "Growth and remodeling hyperelastic",
            "MAT_Muscle_Combo": "Active strain muscle model (combo)",
            "MAT_Muscle_Giantesio": "Giantesio active strain muscle",
            "MAT_myocard": "Myocardial tissue with electrophysiology (FHN, TenTusscher, etc.)",
        },

        "fluid": {
            "MAT_fluid": "Newtonian fluid (DYNVISCOSITY, DENSITY)",
            "MAT_carreauyasuda": "Carreau-Yasuda shear-thinning",
            "MAT_herschelbulkley": "Herschel-Bulkley yield stress fluid",
            "MAT_sutherland": "Temperature-dependent viscosity (Sutherland law)",
        },

        "thermal": {
            "MAT_Fourier": "Fourier heat conduction (CAPA=heat capacity, CONDUCT=conductivity)",
            "MAT_soret": "Soret effect (thermodiffusion coupling)",
        },

        "scalar_transport": {
            "MAT_scatra": "General scalar transport (DIFFUSIVITY parameter)",
            "MAT_scatra_reaction": "Reactive scalar transport",
            "MAT_scatra_chemotaxis": "Chemotactic scalar transport",
        },

        "porous_media": {
            "MAT_FluidPoro": "Darcy fluid in porous medium",
            "MAT_StructPoro": "Structural skeleton for poroelasticity",
            "phase_laws": "Linear, tangent, constraint, by-function",
            "permeability_laws": "Constant, exponential",
        },

        "particle": {
            "MAT_ParticleSPHFluid": "SPH fluid particle",
            "MAT_ParticleDEM": "DEM particle",
            "MAT_ParticlePD": "Peridynamic particle (bond-based)",
        },

        # Beam material names from 4C's MATERIALS grammar (MATERIALS
        # section enum). Catalog previously had wrong delimiter:
        # 'MAT_Beam_Reissner_ElastHyper' (underscore-separated)
        # is NOT a valid 4C material name — real format is
        # 'MAT_BeamReissnerElastHyper' (CamelCase, only one
        # underscore between MAT and the beam family). Wrong
        # names fail at YAML parse with input_spec_builders.cpp
        # 'Could not match this input'. Verified 2026-06-01; the
        # same names in 4C 2026.2.0 and 2026.3.0 (2026-09-30).
        "beam": {
            "MAT_BeamReissnerElastHyper": "Simo-Reissner beam hyperelastic (default for BEAM3R)",
            "MAT_BeamReissnerElastHyper_ByModes": "Reissner beam, parametrized by deformation modes",
            "MAT_BeamReissnerElastPlastic": "Reissner beam with plasticity",
            "MAT_BeamKirchhoffElastHyper": "Kirchhoff beam hyperelastic (default for BEAM3K)",
            "MAT_BeamKirchhoffElastHyper_ByModes": "Kirchhoff beam, parametrized by deformation modes",
            "MAT_BeamKirchhoffTorsionFreeElastHyper": "Kirchhoff torsion-free hyperelastic (default for BEAM3EB)",
            "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes": "Torsion-free Kirchhoff, parametrized by modes",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FLUID MECHANICS (fluid, fluid_ele, fluid_turbulence)
    # ═══════════════════════════════════════════════════════════════════════
    "fluid": {
        "description": "Incompressible Navier-Stokes with stabilized FEM",
        "problemtype": "Fluid",
        "yaml_section": "FLUID DYNAMIC",

        # FLUID DYNAMIC/TIMEINTEGR enum (4C's input grammar —
        # 4C_schema.json). NOTE: different from STRUCTURAL
        # DYNAMIC/DYNAMICTYPE — the fluid enum has Af_/Np_
        # prefixes on Gen_Alpha and underscores on
        # One_Step_Theta. Verified 2026-06-01; the same five values
        # in 4C 2026.2.0 and 2026.3.0 (2026-09-30).
        "time_integration": {
            "Af_Gen_Alpha": "Alpha-form generalized-alpha (alpha-f weighting on the residual)",
            "Np_Gen_Alpha": "N+1 generalized-alpha (the TIMEINTEGR default is One_Step_Theta)",
            "BDF2": "2nd order backward difference formula",
            "One_Step_Theta": "One-step-theta (note underscores — NOT 'OneStepTheta')",
            "Stationary": "Steady state (stationary Navier-Stokes or Stokes; 4C has no RANS model)",
        },

        "stabilization": {
            "SUPG": "Streamline upwind Petrov-Galerkin",
            "GLS": "Galerkin least squares",
            "VMS": "Variational multiscale (recommended)",
            "PSPG": "Pressure stabilization Petrov-Galerkin",
        },

        # FLUID DYNAMIC/TURBULENCE MODEL PHYSICAL_MODEL values; 4C has no RANS or
        # k-epsilon model.
        "turbulence_models": [
            "Smagorinsky / Smagorinsky_with_van_Driest_damping (LES)",
            "Dynamic_Smagorinsky (LES)",
            "Vreman / Dynamic_Vreman (LES)",
            "Multifractal_Subgrid_Scales",
        ],

        "ale": "ALE formulation for moving meshes (ale2, ale3 elements)",

        "pitfalls": [
            "[Syntax] Fluid uses its own element section "
            "'FLUID ELEMENTS' (NOT 'STRUCTURE'). The dynamics-"
            "control section is 'FLUID DYNAMIC' (not 'FLUID' or "
            "'FLUID_DYN'). Wrong section name is rejected at "
            "YAML parse with 'PROC 0 ERROR ... Section ... is "
            "not a valid section name.' from "
            "core/io/src/4C_io_input_file.cpp. Signal: stderr "
            "contains the offending section name + 'not a valid "
            "section name'. The bare word 'STRUCTURE' is caught "
            "the same way, but 'STRUCTURE ELEMENTS' is NOT: it "
            "is a real section, so the elements are read into a "
            "field this problem does not solve, the fluid "
            "discretization stays empty and the abort blames "
            "something else entirely — 'Pressure map empty. "
            "Wrong DIM value in input file?' from "
            "src/fluid/4C_fluid_implicit_integration.cpp, with "
            "no mention of FLUID ELEMENTS anywhere. Do not chase "
            "PROBLEM SIZE/DIM when you see that; check which "
            "section your elements are under. (Verified by "
            "execution 2026-08-06.)",
            "[Numerical] Stabilization choice really does move "
            "the answer at high Reynolds — but every identifier "
            "an earlier version of this entry gave for it is "
            "wrong. There is NO section 'FLUID DYNAMIC/"
            "STABILIZATION' and there are NO keys TAU_TYPE or "
            "TAU_DEF. The section is 'FLUID DYNAMIC/RESIDUAL-"
            "BASED STABILIZATION'; the master switch is STABTYPE "
            "(residual_based | no_stabilization | edge_based | "
            "pressure_projection); the tau definition is "
            "DEFINITION_TAU; and the three flags are PSPG, SUPG "
            "and GRAD_DIV with an UNDERSCORE, not 'GRAD-DIV'. "
            "Turning stabilisation off, and changing "
            "DEFINITION_TAU, each change the computed velocity "
            "field on the same deck. Signal: a wrong section "
            "name gives \"Section 'FLUID DYNAMIC/STABILIZATION' "
            "is not a valid section name.\" from "
            "core/io/src/4C_io_input_file.cpp; a wrong key gives "
            "'Could not match this input' from "
            "core/io/src/4C_io_input_spec_builders.cpp with the "
            "offending line echoed under 'The following data "
            "remains unused:'. The old Signal is NOT usable — a "
            "4C fluid run reports no integrated kinetic energy, "
            "so there is nothing to watch grow. Pin the effect "
            "with a RESULT DESCRIPTION entry instead. (Corrected "
            "by execution 2026-08-06.)",
            "[Integration] ALE (arbitrary Lagrangian-Eulerian) "
            "mesh movement requires a SEPARATE ALE problem set "
            "up alongside the fluid problem in PROBLEM TYPE: "
            "'Fluid_Ale'. The mesh motion equation (typically "
            "elastic) is solved each step on the same "
            "discretization, and the ALE DYNAMIC section is "
            "required. Signal: 'Fluid_Ale' is the exact enum "
            "spelling — 'Fluid_ALE' is refused with \"Candidate "
            "deprecated_selection 'PROBLEMTYPE' has wrong "
            "value\" followed by the whole legal enum; deleting "
            "ALE DYNAMIC gives 'No linear solver defined for ALE "
            "problems. Please set LINEAR_SOLVER in ALE DYNAMIC "
            "to a valid number!' from "
            "src/adapter/4C_adapter_ale.cpp, which names the key "
            "rather than the missing section. Running an ALE "
            "deck under PROBLEMTYPE 'Fluid' is the dangerous "
            "one: it SEGFAULTS during setup with no 4C "
            "diagnostic at all. (Verified by execution "
            "2026-08-06.)",
            "[Numerical] X-wall functions: extended near-wall "
            "treatment for high-Re flows where direct DNS-"
            "resolved boundary layers are infeasible. It is "
            "activated by the bool X_WALL inside the section "
            "'FLUID DYNAMIC/WALL MODEL', together with "
            "Tauw_Type, Tauw_Calc_Type, Projection, Switch_Step "
            "and the GP_Wall_* quadrature counts. There is NO "
            "key WALL_NORMAL_NODE_DISTANCE anywhere in 4C and "
            "no XWALL_*-prefixed keys; an earlier version of "
            "this entry invented both. X_WALL is read while the "
            "mesh is being read, because it selects the ELEMENT "
            "TYPE, so it is not an accuracy dial you can toggle "
            "on a finished deck. Signal: an invented key gives "
            "'Could not match this input' from "
            "core/io/src/4C_io_input_spec_builders.cpp with the "
            "line echoed under 'The following data remains "
            "unused:'; flipping X_WALL to false on a deck built "
            "for x-wall does not degrade the profile, it aborts "
            "with Teuchos::Exceptions::InvalidParameterName, "
            "'Error!  The parameter \"xwalltoggle\" does not "
            "exist'. The old Signal (comparing a log-law slope "
            "with the model on and off) is not observable, "
            "because the off case never produces a profile. "
            "(Corrected by execution 2026-08-06.)",
            "[API] 4C time-integration enum naming is "
            "SECTION-DEPENDENT — the same conceptual scheme "
            "has different spellings in different YAML sections. "
            "FLUID DYNAMIC/TIMEINTEGR accepts {Af_Gen_Alpha, "
            "Np_Gen_Alpha, BDF2, One_Step_Theta, Stationary} "
            "(underscored, with Af_/Np_ prefixes on Gen-Alpha). "
            "SCALAR TRANSPORT DYNAMIC/TIMEINTEGR accepts "
            "{Gen_Alpha, BDF2, One_Step_Theta, Stationary} "
            "(underscored, no prefix). STRUCTURAL DYNAMIC/"
            "DYNAMICTYPE accepts {GenAlpha, GenAlphaLieGroup, "
            "OneStepTheta, Statics, CentrDiff, AdamsBashforth2, "
            "AdamsBashforth4, ExplicitEuler} (CamelCase, no "
            "underscores). THERMAL DYNAMIC/DYNAMICTYPE accepts "
            "{GenAlpha, OneStepTheta, Statics, Undefined} "
            "(CamelCase). The earlier catalog used bare "
            "'GenAlpha' / 'OneStepTheta' uniformly across "
            "physics — wrong for fluid + scatra. Signal: wrong "
            "enum value fails at YAML parse with "
            "input_spec_builders.cpp 'Could not match this "
            "input'. Verified empirically against 4C's schema "
            "2026-06-01; the same value sets in 4C 2026.2.0 "
            "and 2026.3.0 (2026-09-30).",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SCALAR TRANSPORT (scatra, scatra_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "scalar_transport": {
        # ---- READ THIS FIRST ----
        "description": (
            "Convection-diffusion-reaction of one or more scalars. Used for "
            "Poisson, heat-by-diffusion, electrochemistry and level sets.\n"
            "  PROBLEMTYPE:      Scalar_Transport\n"
            "  control section:  SCALAR TRANSPORT DYNAMIC  (NOT 'SCATRA DYNAMIC')\n"
            "  element section:  TRANSPORT ELEMENTS\n"
            "  element line:     <eid> TRANSP <CELLTYPE> <nodes...> MAT <m> TYPE Std\n"
            "                    (MAT and TYPE are both REQUIRED)\n"
            "  material:         MAT_scatra with DIFFUSIVITY   (NOT MAT_Fourier)\n"
            "  BC DOF count:     NUMDOF: 1, and ONOFF/VAL/FUNCT each ONE entry\n"
            "  output:           .vtu is written automatically; there is NO\n"
            "                    runtime-VTK section for scatra and naming one "
            "aborts the run\n"
            "The deck below is complete and runs as written."
        ),
        "problemtype": "Scalar_Transport",
        "yaml_section": "SCALAR TRANSPORT DYNAMIC",

        # ---- COMPLETE RUNNABLE DECK, with a TIME-DEPENDENT Dirichlet BC ----
        "minimal_working_input": """\
# Transient diffusion on a unit square with a time-ramped Dirichlet edge.
# Everything is inline: no mesh file, no include.
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "linear_full"
  TIMEINTEGR: "One_Step_Theta"   # underscores here; the structural section
  THETA: 1.0                     # spells its enum in CamelCase instead
  TIMESTEP: 0.05
  NUMSTEP: 20
  MAXTIME: 1.0
  VELOCITYFIELD: "zero"          # optional - "zero" is already the default
  INITIALFIELD: "zero_field"
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Scatra_Solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1                    # ONE scalar DOF - not 3
    ONOFF: [1]                   # exactly one entry
    VAL: [1.0]                   # amplitude, multiplied by FUNCT
    FUNCT: [1]                   # -> phi = 1.0 * t on this edge
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
TRANSPORT ELEMENTS:
  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 2
      QUANTITY: "phi"
      VALUE: 0.0
      TOLERANCE: 1.0e30          # record mode: abs(diff) prints the true value
""",

        "time_dependent_bc_recipe": (
            "A time-dependent boundary condition is TWO pieces that must "
            "agree in count:\n"
            "  FUNCT<n>:\n"
            "    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: \"<expr in x,y,z,t>\"\n"
            "  DESIGN LINE DIRICH CONDITIONS:\n"
            "    - E: 1\n"
            "      NUMDOF: 1\n"
            "      ONOFF: [1]\n"
            "      VAL: [1.0]\n"
            "      FUNCT: [1]        <- index of FUNCT1, NOT a value\n"
            "The imposed value is VAL[i] * FUNCT[i](x,t). FUNCT: [0] (or "
            "null) means 'no function', i.e. a constant VAL. For scalar "
            "transport NUMDOF is 1 and all three arrays have ONE entry — a "
            "three-entry array copied from a structural example is a "
            "different physics and will not match."
        ),

        "required_vs_optional": {
            "SCALAR TRANSPORT DYNAMIC / LINEAR_SOLVER": (
                "REQUIRED IN PRACTICE (default -1 is not a solver id). "
                "Verified by omission. Signal: 'no linear solver defined for "
                "SCALAR_TRANSPORT problem. Please set LINEAR_SOLVER in "
                "SCALAR TRANSPORT DYNAMIC to a valid number!' The same "
                "section name appears in several sibling messages naming a "
                "DIFFERENT problem (ELCH, LOMA, least-square NURBS), so read "
                "which problem the message names before assuming which key "
                "it means."
            ),
            "SCALAR TRANSPORT DYNAMIC / TIMEINTEGR": "Optional, default 'One_Step_Theta'. Choices: Stationary, One_Step_Theta, BDF2, Gen_Alpha. Note the underscores.",
            "SCALAR TRANSPORT DYNAMIC / SOLVERTYPE": "Optional, default 'linear_full'. Use 'nonlinear' for reaction terms.",
            "SCALAR TRANSPORT DYNAMIC / VELOCITYFIELD": "Optional, default 'zero'. Choices: zero, function, Navier_Stokes. Omitting it is safe for pure diffusion.",
            "SCALAR TRANSPORT DYNAMIC / INITIALFIELD": "Optional, default 'zero_field'. Use 'field_by_function' + INITFUNCNO for a non-zero start.",
            "SCALAR TRANSPORT DYNAMIC / THETA": "Optional, default 0.5. Only read under TIMEINTEGR One_Step_Theta.",
            "SCALAR TRANSPORT DYNAMIC / RESULTSEVERY": "Optional, default 1.",
            "TRANSPORT ELEMENTS line / MAT": "REQUIRED.",
            "TRANSPORT ELEMENTS line / TYPE": "REQUIRED. 'Std' for plain convection-diffusion; other values select the elch / level-set / cardiac variants.",
            "MATERIALS / MAT_scatra / DIFFUSIVITY": "REQUIRED.",
        },

        "time_integration": ["Gen_Alpha", "BDF2", "One_Step_Theta", "Stationary"],

        "physics_variants": {
            "standard": "Pure convection-diffusion-reaction (TYPE Std)",
            "electrochemistry": "Nernst-Planck ion transport (elch, elch_diffcond, elch_scl)",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher models)",
            "level_set": "Level-set advection + reinitialization (TYPE Ls)",
            "porous_media": "Scalar transport in porous media",
            "growth_remodel": "Growth and remodeling scalar transport",
        },

        "elements": (
            "TRANSP with cell types QUAD4/8/9, TRI3/6, HEX8/20/27, TET4/10, "
            "WEDGE6/15, PYRAMID5, LINE2/3 and the NURBS family. Required keys "
            "on every one of them: MAT and TYPE."
        ),

        "pitfalls": [
            (
                "[Input] SCALAR TRANSPORT DYNAMIC: The section name is the full "
                "'SCALAR TRANSPORT DYNAMIC'. The abbreviation 'SCATRA "
                "DYNAMIC' that the source-tree and application names suggest "
                "is not a section. The element section is likewise "
                "'TRANSPORT ELEMENTS', not STRUCTURE or FLUID or SCATRA. "
                "Signal: \"Section 'SCATRA DYNAMIC' is not a valid section "
                "name.\" from core/io/src/4C_io_input_file.cpp, exit 1, "
                "before anything is set up."
            ),
            (
                "[Input] MATERIALS: Scalar transport uses MAT_scatra (key: "
                "DIFFUSIVITY). MAT_Fourier (keys: CAPA, CONDUCT) belongs to "
                "PROBLEMTYPE: Thermo and its THERMO elements. The two are "
                "not interchangeable, and they fail in opposite ways. Signal: "
                "MAT_Fourier under Scalar_Transport aborts cleanly with "
                "'Material type m_thermo_fourier is not supported!' from "
                "scatra_ele/4C_scatra_ele_calc.cpp, but MAT_scatra under "
                "PROBLEMTYPE: Thermo SEGFAULTS — no 4C error block, no "
                "message at all, shell exit status 139. If a Thermo run dies "
                "with no output, check the material first. (Both directions "
                "were run; the segfault reproduced three times out of "
                "three.) The rule is decided by PROBLEMTYPE, not by whether "
                "the physical quantity is called a temperature: heat "
                "conduction modelled through the scatra framework uses "
                "MAT_scatra with the thermal diffusivity as DIFFUSIVITY."
            ),
            (
                "[Output] IO: There is NO runtime-VTK section for scalar transport, "
                "and naming one is fatal rather than merely ineffective. Signal: "
                "Both 'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' and "
                "'IO/RUNTIME VTK OUTPUT/SCATRA' abort at parse with "
                "\"Section '<name>' is not a valid section name.\" and exit "
                "1. You do not need one: a Scalar_Transport run writes "
                "<prefix>-vtk-files/scatra-*.vtu and <prefix>-scatra.pvd "
                "automatically, with no output section in the deck at all. "
                "(Both wrong section names were tried; the automatic .vtu "
                "output was confirmed on a deck containing no IO section "
                "whatsoever.) This corrects the older advice to 'omit the "
                "subsection and convert with post_vtu' — post-processing is "
                "not required."
            ),
            (
                "[Output] The scalar array in the VTU is named phi_1, phi_2, "
                "... one per transported scalar — not 'temperature', 'u' or "
                "'phi'. The RESULT DESCRIPTION spelling is different again: "
                "QUANTITY: \"phi\" with the field group SCATRA and DIS: "
                "\"scatra\". Signal: the written "
                "<prefix>-vtk-files/scatra-*.vtu carries "
                "Name=\"phi_1\" in its PointData; a post-processing script "
                "that asks for 'temperature' finds no such array. "
                "(Confirmed by reading the array names out of a .vtu the "
                "run actually wrote.)"
            ),
            (
                "[Input] DESIGN * DIRICH CONDITIONS: NUMDOF is 1 for scalar "
                "transport and ONOFF, VAL and FUNCT must each have exactly "
                "one entry. Copying a three-entry structural block is the "
                "single most common transcription error here. Signal: a "
                "mismatched array length is rejected at parse with 'Could "
                "not match this input' followed by an echo of the offending "
                "block and the candidate specification."
            ),
            (
                "[Input] SCALAR TRANSPORT DYNAMIC: VELOCITYFIELD already defaults "
                "to 'zero', so a pure-diffusion deck that omits it runs "
                "correctly. Earlier guidance that the key must be set "
                "explicitly was wrong: a deck identical except for the "
                "deleted VELOCITYFIELD line completed with exit 0. Set it "
                "explicitly only to document intent, or when you actually "
                "want 'function' or 'Navier_Stokes'. Signal: none - there is "
                "no error and no warning for the omitted key, which is "
                "why the older advice went unchallenged."
            ),
            (
                "[Input] SCALAR TRANSPORT DYNAMIC: For a stationary Poisson "
                "problem use TIMEINTEGR: \"Stationary\" and drive it with a "
                "DESIGN * NEUMANN condition as the source; for a prescribed "
                "boundary VALUE use DIRICH instead. Swapping the two gives a "
                "run that succeeds and answers a different question — "
                "NEUMANN imposes a flux, DIRICH imposes the scalar. "
                "Signal: none — the run exits 0 either way, so pin the "
                "expected boundary value with a RESULT DESCRIPTION entry, "
                "which turns the wrong choice into 'is WRONG' and exit 1."
            ),
        ],
    },


    # ═══════════════════════════════════════════════════════════════════════
    # THERMAL (thermo)
    # ═══════════════════════════════════════════════════════════════════════
    "thermal": {
        # ---- READ THIS FIRST ----
        "description": (
            "Heat conduction as its own problem type (standalone), or as the "
            "thermal field of TSI / STI / SSTI.\n"
            "  PROBLEMTYPE:      Thermo\n"
            "  control section:  THERMAL DYNAMIC   (the word THERMAL)\n"
            "  element section:  THERMO ELEMENTS   (the word THERMO)\n"
            "  element line:     <eid> THERMO <CELLTYPE> <nodes...> MAT <m>\n"
            "                    MAT is the ONLY key it accepts - adding "
            "KINEM or anything else aborts\n"
            "  material:         MAT_Fourier with CAPA and CONDUCT\n"
            "                    CONDUCT is tensor-typed: write "
            "CONDUCT: {constant: [k]}, a bare scalar is rejected\n"
            "  boundary cond.:   PLAIN 'DESIGN SURF DIRICH CONDITIONS' with "
            "NUMDOF: 1.\n"
            "                    DO NOT use 'DESIGN SURF THERMO DIRICH "
            "CONDITIONS' in a standalone\n"
            "                    Thermo run - it parses cleanly and is then "
            "SILENTLY IGNORED.\n"
            "  result check:     field group THERMAL, DIS: \"thermo\", "
            "QUANTITY: \"temp\"\n"
            "The deck below is complete and runs as written."
        ),
        "problemtype": "Thermo",
        "yaml_section": "THERMAL DYNAMIC",

        # ---- COMPLETE RUNNABLE DECK ----
        "minimal_working_input": """\
# Transient heat conduction in a bar, hot face at x-, everything inline.
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMO DOMAIN:                 # box generator - no NODE COORDS needed
  bottom_corner_point: [0.0, 0.0, 0.0]
  top_corner_point: [1.0, 0.2, 0.2]
  subdivisions: [10, 2, 2]
  elements:
    THERMO:
      HEX8:
        MAT: 1
THERMAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"  # Statics | OneStepTheta | GenAlpha
  TIMESTEP: 0.01
  NUMSTEP: 20
  MAXTIME: 0.2                 # MAXTIME truncates: the run stops at
  INITIALFIELD: "zero_field"   # min(NUMSTEP*TIMESTEP, MAXTIME)
  TOLTEMP: 1.0e-10             # temperature-update tolerance
  TOLRES: 1.0e-08              # residual tolerance
  MAXITER: 30
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 1.0                   # 1.0 = backward Euler, 0.5 = Crank-Nicolson
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermo_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]        # tensor-typed - NOT "CONDUCT: 1.0"
DESIGN SURF DIRICH CONDITIONS: # PLAIN. NOT "DESIGN SURF THERMO DIRICH".
  - E: 1
    NUMDOF: 1                  # one temperature DOF
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]                 # 0 = constant; use a FUNCT index to ramp
DSURF-NODE TOPOLOGY:
  - "SIDE thermo x- DSURFACE 1"
IO/RUNTIME VTK OUTPUT:         # OPTIONAL - drop both VTK sections and the
  INTERVAL_STEPS: 5            # deck still runs, it just writes no .vtu.
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:  # For .vtu you need BOTH sections AND at
  OUTPUT_THERMO: true                # least one field flag. INTERVAL_STEPS
  TEMPERATURE: true                  # lives ONLY in the parent section.
RESULT DESCRIPTION:
  - THERMAL:                   # THERMAL, not THERMO
      DIS: "thermo"
      NODE: 1
      QUANTITY: "temp"
      VALUE: 0.0
      TOLERANCE: 1.0e30        # record mode: abs(diff) prints the true value
""",

        "inline_mesh_variant": (
            "To hand-write the mesh instead of generating it, delete the "
            "THERMO DOMAIN section and supply three sections instead:\n"
            "  NODE COORDS:      - \"NODE 1 COORD 0.0 0.0 0.0\"  ...\n"
            "  THERMO ELEMENTS:  - \"1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1\"\n"
            "  DSURF-NODE TOPOLOGY: - \"NODE 1 DSURFACE 1\"  ...\n"
            "The box generator only makes HEX8/20/27 and WEDGE6/15, so any 2D "
            "thermal problem must use the inline route with "
            "THERMO QUAD4 / TRI3 elements and DLINE topology."
        ),

        "required_vs_optional": {
            "THERMAL DYNAMIC / LINEAR_SOLVER": (
                "REQUIRED IN PRACTICE (default -1 is not a solver id). "
                "Verified by omission. Signal: 'No linear solver defined for "
                "thermal solver. Please set LINEAR_SOLVER in THERMAL DYNAMIC "
                "to a valid number!' — note the capital 'No', where the "
                "structural, scatra and contact variants of this same "
                "message start lowercase, so a case-sensitive grep for one "
                "will miss the others."
            ),
            "THERMAL DYNAMIC / DYNAMICTYPE": "Optional, default 'OneStepTheta'. Choices: Statics, OneStepTheta, GenAlpha.",
            "THERMAL DYNAMIC / TIMESTEP": "Optional, default 0.05.",
            "THERMAL DYNAMIC / NUMSTEP": "Optional, default 200.",
            "THERMAL DYNAMIC / MAXTIME": "Optional, default 5. Truncates the run: it stops at min(NUMSTEP*TIMESTEP, MAXTIME).",
            "THERMAL DYNAMIC / TOLTEMP": "Optional, default 1e-10. Temperature-update convergence tolerance.",
            "THERMAL DYNAMIC / TOLRES": "Optional, default 1e-08. Residual convergence tolerance.",
            "THERMAL DYNAMIC / INITIALFIELD": "Optional, default 'zero_field'. Choices: zero_field, field_by_function (+ INITFUNCNO), field_by_condition.",
            "THERMAL DYNAMIC / MAXITER": "Optional, default 50.",
            "THERMAL DYNAMIC / PREDICT": "Optional, default 'ConstTemp'. Choices: ConstTemp, ConstTempRate, TangTemp.",
            "THERMAL DYNAMIC/ONESTEPTHETA / THETA": "Optional, default 0.5. Only read under DYNAMICTYPE OneStepTheta.",
            "THERMAL DYNAMIC/GENALPHA": "Sub-section with GAMMA, ALPHA_M, ALPHA_F, RHO_INF, GENAVG. Only read under DYNAMICTYPE GenAlpha.",
            "THERMO ELEMENTS line / MAT": "REQUIRED, and the ONLY key the THERMO element accepts.",
            "MATERIALS / MAT_Fourier / CAPA": "REQUIRED. Volumetric heat capacity.",
            "MATERIALS / MAT_Fourier / CONDUCT": "REQUIRED, tensor-typed. Isotropic form: CONDUCT: {constant: [k]}.",
        },

        "time_integration": ["Statics", "GenAlpha", "OneStepTheta"],

        "elements": (
            "USE ONLY THESE, all confirmed to run: 2D QUAD4, QUAD8, QUAD9, "
            "TRI3; 3D HEX8, HEX20, HEX27, TET4, TET10, WEDGE6, PYRAMID5; "
            "1D LINE2; plus NURBS27 with the full NURBS apparatus. Every "
            "one takes MAT and nothing else.\n"
            "DO NOT USE these five, even though `4C --parameters` lists "
            "them under THERMO: TRI6, WEDGE15, LINE3, NURBS4, NURBS9. They "
            "parse and then die at element evaluation with 'Element shape "
            "TRI6 (6 nodes) not activated. Just do it.' from "
            "thermo/src/element/4C_thermo_ele_impl.cpp. Appearing in "
            "`--parameters` means the PARSER accepts it, not that the "
            "physics module implements it."
        ),

        "boundary_conditions": {
            "_which_prefix": (
                "STANDALONE PROBLEMTYPE: Thermo -> use the PLAIN sections: "
                "DESIGN POINT/LINE/SURF/VOL DIRICH CONDITIONS and "
                "... NEUMANN CONDITIONS, with NUMDOF: 1. The THERMO-prefixed "
                "sections exist and parse but are dropped. COUPLED TSI -> the "
                "THERMO-prefixed sections are the ones that reach the thermal "
                "field, because the plain ones belong to the structure."
            ),
            "DESIGN SURF DIRICH CONDITIONS": "Prescribed temperature (standalone Thermo). NUMDOF 1.",
            "DESIGN SURF NEUMANN CONDITIONS": "Prescribed heat flux (standalone Thermo). NUMDOF 1.",
            "DESIGN SURF THERMO DIRICH CONDITIONS": "Prescribed temperature IN A TSI RUN. Silently ignored in a standalone Thermo run.",
            "DESIGN SURF THERMO NEUMANN CONDITIONS": "Prescribed heat flux IN A TSI RUN. Silently ignored in a standalone Thermo run.",
            "DESIGN THERMO CONVECTION SURF CONDITIONS": "Convective heat transfer, h*(T - T_inf).",
            "DESIGN THERMO ROBIN SURF CONDITIONS": "Robin boundary condition.",
            "DESIGN SURF THERMO INITIAL FIELD CONDITIONS": "Initial temperature by condition (with INITIALFIELD: field_by_condition).",
        },

        "pitfalls": [
            (
                "[Input] DESIGN * DIRICH CONDITIONS: THE DANGEROUS ONE, and it is "
                "the first thing to get right. In a STANDALONE "
                "PROBLEMTYPE: Thermo run, 'DESIGN SURF THERMO DIRICH "
                "CONDITIONS' is a perfectly valid section name, parses "
                "without a single warning, and is then SILENTLY DROPPED — "
                "the temperature stays at its initial value and the run "
                "exits 0 looking like a success. The sections that actually "
                "reach the thermo discretisation are the PLAIN ones: DESIGN "
                "POINT/LINE/SURF/VOL DIRICH CONDITIONS and ... NEUMANN "
                "CONDITIONS. The THERMO-prefixed variants belong to the "
                "coupled TSI path. Signal: there is NO diagnostic — that is "
                "the whole problem. Detect it by adding a RESULT DESCRIPTION "
                "THERMAL entry on a node whose temperature you prescribed; "
                "the silent wrong answer then becomes 'is WRONG --> "
                "actresult= 0.00000000000000000e+00' and exit 1. (Verified "
                "on both a static and a transient run, 3D: with the "
                "prefixed section a mid-bar node read exactly 0.0 while one "
                "face was held at 100; changing only that section name gave "
                "the expected diffusive profile. Verified by execution "
                "2026-08-03 and re-confirmed on a transient 2D and a "
                "transient 3D deck.)"
            ),
            (
                "[Syntax] section names: THERMAL vs THERMO is not interchangeable "
                "and there is no rule of thumb — you have to know which is "
                "which. The control section is 'THERMAL DYNAMIC' and the "
                "RESULT DESCRIPTION field group is 'THERMAL'; the element "
                "section is 'THERMO ELEMENTS', the element type is 'THERMO', "
                "the discretisation is named 'thermo', and the coupled "
                "condition sections are '... THERMO DIRICH ...'. Writing "
                "'DESIGN SURF THERMAL DIRICH CONDITIONS' is fatal — but "
                "note it fails LOUDLY, unlike the THERMO-prefixed trap "
                "above. Signal: \"Section 'DESIGN SURF THERMAL DIRICH "
                "CONDITIONS' is not a valid section name.\" from "
                "core/io/src/4C_io_input_file.cpp, exit 1. (An earlier "
                "version of this entry attributed the message to "
                "input_spec_builders.cpp and quoted it as 'unknown section'; "
                "that string does not occur anywhere in the binary. "
                "Corrected against the real output.)"
            ),
            (
                "[Input] THERMO ELEMENTS: The THERMO element accepts MAT and "
                "nothing else — no KINEM, no TYPE, no THICK. Any extra "
                "token on the line is fatal. Signal: \"After parsing, the "
                "line still contains 'KINEM linear'.\" followed by 'Parsed "
                "parameters: MAT : 1', from core/io/src/4C_io_input_spec.cpp."
            ),
            (
                "[Input] MATERIALS: MAT_Fourier.CONDUCT is tensor-typed. The "
                "isotropic case must be written as a 'constant:' list — "
                "'CONDUCT: {constant: [k]}' — and a bare scalar is rejected. "
                "Signal: 'CONDUCT: 1.0' produces \"Failed to match "
                "specification in section 'MATERIALS'\"; the wrapped form "
                "reaches fill_complete on discretisation 'thermo'."
            ),
            (
                "[Input] MATERIALS: Do not put MAT_scatra in a PROBLEMTYPE: Thermo "
                "deck. It does not produce a 4C error — the process "
                "SEGFAULTS and the shell exit status is 139. "
                "MAT_scatra (DIFFUSIVITY) belongs to Scalar_Transport with "
                "TRANSP elements; MAT_Fourier (CAPA, CONDUCT) belongs to "
                "Thermo with THERMO elements. Signal: no 'PROC 0 ERROR' "
                "block — this is not a FOUR_C_THROW — but the run is NOT "
                "silent, which tells you where it died: 4C parses the deck, "
                "prints 'fill_complete() on discretization thermo' and "
                "'Welcome to Thermal Time Integration', and only then hits "
                "SIGSEGV inside the Thermo::TimIntImpl constructor, which "
                "OpenMPI's handler names in its backtrace. So the marker is "
                "'Signal: Segmentation fault (11)' AFTER a normal-looking "
                "setup, plus shell status 139. (An earlier version of this "
                "entry said there is no 4C output at all; there is. "
                "Reproduced three times out of three.)"
            ),
            (
                "[Output] IO: Thermal runtime VTU is switched on by ONE section, "
                "'THERMAL DYNAMIC/RUNTIME VTK OUTPUT'. It carries the field "
                "flags (OUTPUT_THERMO, TEMPERATURE, TEMPERATURE_RATE, "
                "CONDUCTIVITY, ELEMENT_OWNER, ELEMENT_GID, NODE_GID) and "
                "nothing else — no frequency key. That subsection ALONE "
                "writes .vtu; the parent 'IO/RUNTIME VTK OUTPUT' is not "
                "required and on its own writes none. Putting INTERVAL_STEPS "
                "in the thermal subsection is fatal, not ignored. Signal: "
                "'Could not match this input' echoing the THERMAL "
                "DYNAMIC/RUNTIME VTK OUTPUT block. With neither section "
                "present the run still succeeds and writes only the binary "
                ".control / .mesh / .result files — no .vtu at all. The "
                "frequency knob for the thermo writer is THERMAL "
                "DYNAMIC/RESULTSEVERY: INTERVAL_STEPS parses in the parent "
                "section but does not throttle it. (An earlier version of "
                "this entry said both sections were needed and that "
                "INTERVAL_STEPS lived in the parent; neither survived "
                "execution.)"
            ),
            (
                "[Input] THERMAL DYNAMIC: The thermal integrator is NOT NOX and "
                "prints NO tolerance in its output — unlike STRUCTURAL "
                "DYNAMIC, where the status block echoes your TOLDISP and "
                "TOLRES back at you. A thermal run prints 'Predictor thermo "
                "absolute res-norm <r>' and then a table with columns "
                "numiter / abs-res-norm / abs-temp-norm / wct. So you "
                "cannot confirm TOLTEMP and TOLRES were read by looking for "
                "them in the log; grepping for '<' finds nothing. Confirm "
                "them by behaviour instead: an impossible TOLTEMP with a "
                "small MAXITER gives 'Newton unconverged in <n> "
                "iterations', and NORMCOMBI_RESFTEMP: \"Or\" makes the "
                "same deck converge in fewer iterations. Signal: 'Newton "
                "unconverged in <n> iterations' is the ONLY place a "
                "thermal tolerance surfaces in the output."
            ),
            (
                "[Input] THERMAL DYNAMIC: INITIALFIELD: 'field_by_function' needs "
                "INITFUNCNO pointing at a FUNCT block. Omitting INITIALFIELD "
                "defaults to T = 0, which for a structural material carrying "
                "INITTEMP > 0 gives spurious thermal strain at t = 0, since "
                "it is the difference T - INITTEMP that drives the "
                "expansion. Match INITIALFIELD to INITTEMP. Signal: none at "
                "parse or run time - the detector is a non-zero "
                "displacement at t = 0 in a problem that should start "
                "unstrained, so pin the initial state with a RESULT "
                "DESCRIPTION entry."
            ),
            (
                "[Physics] TSI: In a coupled TSI run the thermal field is SOLVED by "
                "4C, not prescribed. Pinning the temperature with a "
                "Dirichlet condition on every node defeats the coupling — "
                "there is then no feedback from the structure to the thermal "
                "field. Use thermal sources (Joule heating, mechanical "
                "dissipation) and put boundary conditions only on real "
                "heat-input/output boundaries. Signal: none - the run "
                "succeeds and answers a different question. The "
                "detector is a thermal field identical to the "
                "prescribed one at every step."
            ),
        ],
    },


    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-PHYSICS COUPLING
    # ═══════════════════════════════════════════════════════════════════════
    "tsi": {
        "description": "Thermo-Structure Interaction — the key multi-physics coupling in 4C",
        "problemtype": "Thermo_Structure_Interaction",
        "yaml_sections": ["STRUCTURAL DYNAMIC", "THERMAL DYNAMIC", "TSI DYNAMIC"],

        # COUPALGO enum values from 4C's input grammar
        # (TSI DYNAMIC/COUPALGO). Verified via 4C_schema.json
        # 2026-06-01, and the same seven in 4C 2026.2.0 and
        # 2026.3.0 (2026-09-30). Names that LOOK obvious are NOT — the
        # underscore between 'iterstagg' and the variant is
        # load-bearing, 'fixedrel' is actually 'fixedrelax',
        # and the monolithic variant is 'tsi_monolithic'
        # (NOT bare 'monolithic'). Wrong values fail at
        # YAML parse with input_spec_builders.cpp
        # 'Could not match this input'.
        "coupling_algorithms": {
            "tsi_oneway": "One-way: thermal → structural (no feedback)",
            "tsi_sequstagg": "Sequential staggered (solve thermal, then structural, once per step)",
            "tsi_iterstagg": "Iterative staggered (iterate until convergence)",
            "tsi_iterstagg_aitken": "Iterative staggered with Aitken acceleration",
            "tsi_iterstagg_aitkenirons": "Aitken-Irons variant",
            "tsi_iterstagg_fixedrelax": "Fixed relaxation iterative staggered",
            "tsi_monolithic": "Simultaneous solve of all fields (TSI DYNAMIC/MONOLITHIC section)",
        },

        "requirements": [
            "SOLIDSCATRA elements (NOT plain SOLID — the SCATRA coupling is needed). "
            "3D ONLY: HEX8, HEX27, TET4, TET10, NURBS27 (the module is "
            "solid_scatra_3D_ele — QUAD4/TRI3 are NOT accepted, and WALL QUAD4 "
            "rejects MAT_Struct_ThermoStVenantK with 'Invalid type of material "
            "law for wall element'). For 2D plane strain use the "
            "plane_strain_2d variant: a one-element-thick HEX8 slab with u_z "
            "fixed on all nodes.",
            "MAT_Struct_ThermoStVenantK (structural material with thermal expansion)",
            "MAT_Fourier (thermal material, reached through the CLONING MATERIAL MAP; 4C before 2026.3.0 could also name it with THERMOMAT, which 4C 2026.3.0 removed)",
            "CLONING MATERIAL MAP: SRC_FIELD structure → TAR_FIELD thermo",
            "Two LINEAR_SOLVERs: one for thermal, one for structural",
            "FUNCT for INITIALFIELD: SYMBOLIC_FUNCTION_OF_SPACE_TIME for initial temperature",
        ],

        "pitfalls": [
            (
                "[Input] 4C runs NO 2D TSI (measured on a 2026.2.0 "
                "development build and on the 2026.2.0 and 2026.3.0 "
                "releases), and every TSI corpus test is 3D. Signal: a 2D "
                "thermo-mechanical deck dead-ends EVERY way on all three. "
                "On the development build, WALL QUAD4 + MAT_Struct_ThermoStVenantK aborts "
                "with 'Unsupported solid element type!' from src/tsi/"
                "4C_tsi_utils.cpp — TSI's clone strategy rejects anything "
                "that is not a SolidScatra element before a material is ever "
                "evaluated, so you do NOT get the material-law message the "
                "wall element has ('Invalid type of material law for wall "
                "element' in 4C_w1_mat.cpp is real code but is never "
                "reached). SOLID QUAD4 and SOLIDSCATRA QUAD4 both abort with "
                "\"Element '<name>' does not seem to know cell type 'quad4'\" "
                "(4C_fem_general_element_definition.cpp): there SOLIDSCATRA "
                "declares only hex8, hex27, tet4, tet10 and nurbs27. On "
                "2026.2.0 and 2026.3.0 there is no WALL, and SOLID QUAD4 and SOLIDSCATRA "
                "QUAD4 (with THICKNESS and PLANE_ASSUMPTION) pass the input "
                "check, build the structure and then abort with the same "
                "'Unsupported solid element type!' from src/tsi/"
                "4C_tsi_utils.cpp. "
                "For 2D plane "
                "strain use generate_input('tsi', 'plane_strain_2d', ...): a "
                "one-element-thick 3D SOLIDSCATRA HEX8 slab with u_z fixed "
                "on all nodes (exact plane strain), the temperature field "
                "imposed volume-wide via DESIGN VOL THERMO DIRICH + a "
                "symbolic FUNCT (param temp_expr — pass the partner code's "
                "temperature solution). Verified live against a built 4C "
                "binary: the deck runs and the slab deforms under the "
                "imposed thermal field. Check the displacement you get "
                "against the analytic plane-strain thermal-expansion "
                "estimate for your own parameters."
            ),
            (
                "[Input] Without CLONING MATERIAL MAP, 4C "
                "crashes at initialization. Signal: the TSI "
                "setup phase aborts with the shared cloning "
                "utility's generic message, 'At least one "
                "material pairing required in --CLONING "
                "MATERIAL MAP.' from core/fem/src/general/"
                "utils/4C_fem_general_utils_createdis.hpp. It "
                "names neither TSI nor the thermo field and "
                "still spells the section in the retired "
                "--SECTION dat form, so only the backtrace "
                "(ThermoStructureCloneStrategy, setup_tsi) "
                "tells you which clone failed. The thermal "
                "discretisation has no way to inherit the "
                "structural cell topology + nodes. Standard "
                "form: SRC_FIELD: structure, SRC_MAT: <struct_"
                "mat_id>, TAR_FIELD: thermo, TAR_MAT: <thermo_"
                "mat_id>. (An earlier version quoted 'cannot "
                "clone material for thermo field' from "
                "4C_adapter_str_factory.cpp; neither the "
                "string nor that origin occurs in the output.)"
            ),
            (
                "[Input] THEXPANS in MAT_Struct_"
                "ThermoStVenantK is the thermal-expansion "
                "coefficient, and it is the one number here "
                "that does NOT need converting between "
                "Kelvin and Celsius: a linear expansion "
                "coefficient in 1/K and in 1/degC is the "
                "same value, and an all-Kelvin and an all-"
                "Celsius deck describing the same state give "
                "the same displacement with THEXPANS "
                "unchanged. What must be on ONE temperature "
                "scale is INITTEMP and the thermal field. "
                "Signal: mixing them — INITTEMP written as a "
                "Celsius number while the thermal solution "
                "is in Kelvin, or vice versa — shifts the "
                "displacement by exactly the T_reference "
                "offset (273.15) times alpha*length, which "
                "is easily mistaken for a boundary-condition "
                "error. Use one scale (all SI, all CGS, "
                "etc.) throughout. (An earlier version "
                "blamed the units of THEXPANS; the offset it "
                "described is real but comes from the "
                "temperature scale of INITTEMP.)"
            ),
            (
                "[Input] INITTEMP is the reference "
                "temperature for ZERO thermal strain, and it "
                "is a REQUIRED key of MAT_Struct_"
                "ThermoStVenantK — it does not default to "
                "anything. Signal: omit it and the deck does "
                "not parse, with \"Failed to match "
                "specification in section 'MATERIALS'\" "
                "(4C_global_data_read.cpp), 'Could not match "
                "this input' (4C_io_input_spec_builders.cpp) "
                "and the candidate dump line \"[X] Expected "
                "parameter 'INITTEMP'\", exit 1 before any "
                "element is built. The danger is therefore "
                "not omission but writing INITTEMP: 0 (or "
                "any value that is not the stress-free "
                "state): expansion is u = alpha * (T - "
                "INITTEMP) * L, so a zero reference makes a "
                "heated specimen behave as if it started "
                "from absolute zero. Set INITTEMP to the "
                "stress-free temperature (room temperature "
                "for typical experiments). (An earlier "
                "version said omitting INITTEMP defaults to "
                "0; it is rejected at parse time instead.)"
            ),
            (
                "[Input] TSI DYNAMIC controls the COUPLING "
                "(time step, ITEMAX, COUPALGO); the per-"
                "field STRUCTURAL DYNAMIC and THERMAL "
                "DYNAMIC sections control the individual "
                "field solvers. Signal: setting NUMSTEP in "
                "STRUCTURAL DYNAMIC but not in TSI DYNAMIC "
                "is silently ignored — TSI DYNAMIC's "
                "NUMSTEP wins and the structural section's "
                "value is unused. Always set time-loop "
                "controls in TSI DYNAMIC; use per-field "
                "DYNAMIC sections for tolerances and "
                "predictor type only. (Audit 2026-06-02.)"
            ),
            (
                "[Input] For one-way TSI (no feedback): "
                "ITEMAX = 1 (only one coupling iteration "
                "needed), but raising it costs nothing — "
                "tsi_oneway makes exactly one pass per step "
                "whatever ITEMAX says, and ITEMAX 1 and "
                "ITEMAX 10 issue the same number of linear "
                "solves. Where ITEMAX matters is the other "
                "direction: ITEMAX = 1 on a TWO-way "
                "(tsi_iterstagg*) problem stops before "
                "convergence and yields a partly-converged "
                "coupled response. Signal: it is not silent "
                "— 4C prints '>>>>>> not converged in itemax "
                "steps!' once per coupling step, which a "
                "converged run never prints, and the "
                "temperature and displacement drift away "
                "from the converged values. Match ITEMAX to "
                "coupling type. (An earlier version claimed "
                "extra iterations waste wall-clock on a "
                "one-way problem and that the truncated "
                "two-way answer looks right; neither held "
                "under execution.)"
            ),
            (
                "[Input] SOLIDSCATRA elements REQUIRE a TYPE "
                "token, and 'TYPE Undefined' is the right "
                "one for TSI — but it is a recommendation, "
                "not the only value that works: 'TYPE Std' "
                "and the other enum members run the same "
                "deck to the same answer, because the "
                "SCATRA half is cloned away into the thermal "
                "field. Two different failures. OMITTING "
                "TYPE is a PARSE error while the element "
                "line is read: \"Required value 'TYPE' not "
                "found in input line\" from "
                "4C_io_input_spec_builders.cpp — not a "
                "runtime throw. Writing a value OUTSIDE the "
                "enum gives 'The input type <value> is not "
                "valid for SOLIDSCATRA elements!' from "
                "src/solid_scatra_3D_ele/"
                "4C_solid_scatra_3D_ele_lib.cpp, still "
                "during element reading — no mesh is built "
                "and no time step starts. Signal: those two "
                "messages. Full format: <id> SOLIDSCATRA "
                "HEX8 <n1..n8> MAT <id> KINEM nonlinear TYPE "
                "Undefined. (An earlier version said 'TYPE "
                "Std' throws and that omission is a runtime "
                "error at problem setup; neither is true.)"
            ),
            (
                "[Input] For one-way thermal -> structural "
                "TSI: MUST add TSI DYNAMIC/PARTITIONED "
                "section with COUPVARIABLE: Temperature. "
                "Signal: without it, 4C defaults to "
                "displacement coupling (structural -> "
                "thermal), which is BACKWARDS for heating "
                "problems — the result is zero "
                "displacement everywhere because the "
                "structural field gets no thermal forcing "
                "input. Sanity check: a heated bar should "
                "expand; if it doesn't, COUPVARIABLE is "
                "likely missing or set to Displacement. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Monolithic TSI requires the "
                "Belos iterative solver with a block "
                "preconditioner (NOT UMFPACK). Signal: "
                "pointing TSI DYNAMIC/MONOLITHIC's "
                "LINEAR_SOLVER at a 'SOLVER: UMFPACK' block "
                "aborts inside "
                "TSI::Monolithic::create_linear_solver with "
                "the two-word message 'Iterative solver "
                "expected' from 4C_tsi_monolithic.cpp, "
                "before the first time step. Note the "
                "message names neither Belos nor the solver "
                "you wrote, so grepping the error for either "
                "finds nothing. Note also that omitting "
                "LINEAR_SOLVER from TSI DYNAMIC entirely is "
                "a DIFFERENT and much more explicit error: "
                "'no linear solver defined for monolithic "
                "TSI. Please set LINEAR_SOLVER in TSI "
                "DYNAMIC to a valid number!'. For simple "
                "one-way problems, use partitioned "
                "tsi_oneway with UMFPACK (much simpler "
                "setup). (An earlier version quoted "
                "'monolithic TSI requires Belos'; that "
                "string does not exist.)"
            ),
            (
                "[Input] Volume-level thermal Dirichlet: "
                "use DESIGN VOL THERMO DIRICH CONDITIONS + "
                "DVOL-NODE TOPOLOGY to prescribe "
                "temperature on all nodes in a volume "
                "region. Signal: applying a Dirichlet to "
                "the surface-only set when you want a "
                "constant-temperature volume leaves "
                "interior nodes free — temperature "
                "develops a non-uniform interior profile "
                "instead of staying clamped. For "
                "uniform-T initial condition over a "
                "region, prefer INITIALFIELD + FUNCT over "
                "DIRICH BCs (more efficient). (Audit "
                "2026-06-02.)"
            ),
            "[API] TSI DYNAMIC/COUPALGO is an enum of exactly 7 "
            "values: tsi_oneway, tsi_sequstagg, tsi_iterstagg, "
            "tsi_iterstagg_aitken, tsi_iterstagg_aitkenirons, "
            "tsi_iterstagg_fixedrelax, tsi_monolithic. The "
            "earlier catalog had FOUR wrong names: "
            "'tsi_iterstaggaitken' (missing underscore — real: "
            "'tsi_iterstagg_aitken'), 'tsi_iterstaggaitkenirons' "
            "(missing underscore — real: 'tsi_iterstagg_"
            "aitkenirons'), 'tsi_iterstaggfixedrel' (wrong stem "
            "— real: 'tsi_iterstagg_fixedrelax'), and "
            "'monolithic' (missing 'tsi_' prefix — real: "
            "'tsi_monolithic'). Signal: invalid COUPALGO value "
            "in YAML produces 'PROC 0 ERROR' from "
            "input_spec_builders.cpp with 'Could not match this "
            "input' and the offending YAML block echoed. "
            "Verified empirically against 4C's schema "
            "2026-06-01; the same seven values in 4C 2026.2.0 "
            "and 2026.3.0 (2026-09-30).",
            "[API] SOLIDSCATRA TYPE values depend on the release. "
            "The development build before 2026.2.0 and 4C 2026.2.0 "
            "accept 10: Undefined, AdvReac, CardMono, GR, "
            "Chemo, ChemoReac, ElchDiffCond, ElchElectrode, "
            "Loma, Std (src/solid_scatra_3D_ele/"
            "4C_solid_scatra_3D_ele_lib.cpp; in 4C 2026.2.0 "
            "src/solid_scatra_ele/4C_solid_scatra_ele_lib.cpp). "
            "4C 2026.3.0 adds 'NLS' as an eleventh value; the "
            "two older builds reject NLS, and the 4C 2026.1.0 "
            "source has neither GR nor NLS. For TSI specifically, "
            "use 'TYPE Undefined' — the SCATRA half is cloned "
            "into a thermal discretization, so the SCATRA impl "
            "in the structure is unused. The TYPE value is not "
            "checked by the YAML schema, so an out-of-enum "
            "value fails later, while the element line is "
            "read: 'The input type <value> is not valid for "
            "SOLIDSCATRA elements!'. That is still before any "
            "mesh or time step exists — not 'at the first time "
            "step'. Cell types: on 4C before 2026.2.0 SOLIDSCATRA "
            "declares HEX8, HEX27, TET4, TET10 and NURBS27 and "
            "NOTHING ELSE — QUAD4, QUAD9, TRI3, TRI6, PYRAMID5 "
            "and WEDGE6 are all refused with \"Element "
            "'SOLIDSCATRA' does not seem to know cell type "
            "'<ct>'.\" 4C 2026.2.0 and later add QUAD4, QUAD9, TRI3 and "
            "TRI6 (with THICKNESS and PLANE_ASSUMPTION), but a "
            "TSI deck with them still stops, at 'Unsupported "
            "solid element type!' (the 2D entry above). "
            "Signal: those messages.",
        ],
    },

    "fsi": {
        "description": "Fluid-Structure Interaction — partitioned and monolithic coupling",
        "problemtype": "Fluid_Structure_Interaction",

        "partitioned_algorithms": {
            "Dirichlet-Neumann": "Standard: displacement/velocity/force coupling at interface",
            "DirichletNeumannSlideALE": "Sliding interface variant",
            # FSI DYNAMIC/COUPALGO values
            "relaxation": ["Fixed (iter_stagg_fixed_rel_param)", "Aitken (iter_stagg_AITKEN_rel_param, the default)",
                           "Steepest descent (iter_stagg_steep_desc)", "NLCG (iter_stagg_NLCG)",
                           "MPE, minimal polynomial extrapolation (iter_stagg_MPE)"],
            "MFNK": "Matrix-free Newton-Krylov (advanced, robust; iter_stagg_MFNK_FD / iter_stagg_MFNK_FSI)",
        },

        "monolithic_algorithms": {
            "fluid_split": "Monolithic with fluid-based splitting",
            "structure_split": "Monolithic with structure-based splitting",
            "mortar": "Mortar-based monolithic (non-matching meshes)",
            "xfem": "XFEM-based monolithic (no mesh conformity needed)",
        },

        "required_sections": [
            "PROBLEM TYPE", "PROBLEM SIZE",
            "STRUCTURAL DYNAMIC", "STRUCTURAL DYNAMIC/GENALPHA",
            "FLUID DYNAMIC", "ALE DYNAMIC",
            "FSI DYNAMIC", "FSI DYNAMIC/MONOLITHIC SOLVER",
            "MATERIALS", "CLONING MATERIAL MAP",
            "STRUCTURE GEOMETRY", "FLUID GEOMETRY",
            "DESIGN FSI COUPLING LINE CONDITIONS (2D) or SURF CONDITIONS (3D)",
        ],

        "ale_boundary_conditions": {
            "rules": [
                "ALL walls with no-slip fluid BC: apply ALE Dirichlet (fix mesh)",
                "Inflow boundary: apply ALE Dirichlet (fix mesh)",
                "Outflow boundary: apply ALE Dirichlet (fix mesh)",
                "Cylinder/obstacle surfaces: apply ALE Dirichlet (fix mesh)",
                "FSI interface: do NOT apply ALE Dirichlet (mesh moves with structure)",
            ],
            "common_mistake": (
                "Forgetting ALE Dirichlet on some outer boundary causes the "
                "ALE mesh to distort freely, leading to inverted elements."
            ),
        },

        "valid_2d_elements": {
            "FLUID": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "SOLID (structure)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "notes": (
                "QUAD4 most validated. TRI3 less accurate for "
                "pressure. NOTE: the legacy 'WALL' eletype was "
                "replaced by 'SOLID' in the 4C 2026.2.0 release, with "
                "THICKNESS / PLANE_ASSUMPTION in place of WALL's THICK / "
                "STRESS_STRAIN / GP — see the [API] "
                "pitfall in SOL_MECH for the parobjectfactory.cpp "
                "error you get if you write 'WALL QUAD4'."
            ),
        },

        "pitfalls": [
            (
                "[Reference] FSI is the most complex problem "
                "type in 4C — three coupled fields (structure "
                "+ fluid + ALE), each needs its own DYNAMIC "
                "section + SOLVER, plus FSI DYNAMIC/MONOLITHIC "
                "or PARTITIONED SOLVER + CLONING MATERIAL MAP. "
                "The five sections fail in five different ways, "
                "so do not grep for one message. Signal: only "
                "PROBLEM TYPE is a required SECTION — omitting "
                "it gives \"Required section 'PROBLEM TYPE' not "
                "found in input file.\" from "
                "core/io/src/4C_io_input_file.cpp. Omitting "
                "STRUCTURAL / FLUID / ALE DYNAMIC instead "
                "aborts from that field's own adapter "
                "(4C_adapter_str_structure.cpp, "
                "4C_adapter_fld_base_algorithm.cpp, "
                "4C_adapter_ale.cpp) with 'no linear solver "
                "defined for ...', because LINEAR_SOLVER has no "
                "default. Omitting FSI DYNAMIC does not abort "
                "at all: 4C silently defaults COUPALGO to a "
                "PARTITIONED scheme and MAXTIME/NUMSTEP to "
                "their defaults, then fails much later on a "
                "negative fluid Jacobian. Only CLONING MATERIAL "
                "MAP aborts on its own content (see below). "
                "There is no 'missing required section' string "
                "in the binary — work from a tutorial instead "
                "of greenfield. (Audit 2026-06-02; corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Input] FSI fluid elements MUST set "
                "NA: ALE (not Euler) in the FLUID GEOMETRY "
                "ELEMENT_BLOCKS entry. Signal: leaving NA: "
                "Euler aborts inside "
                "FSI::Monolithic::setup_system, before the "
                "first time step, with 'got N master nodes but "
                "0 slave nodes for coupling' from "
                "coupling/src/adapter/4C_coupling_adapter.cpp "
                "— the master count is the structure side of "
                "the FSI interface and the zero is the fluid "
                "side, whose interface map is empty because a "
                "non-ALE fluid field has no FSI interface. The "
                "message names neither the element nor the NA "
                "keyword, so it reads like a mesh problem. "
                "There is no 'incompatible with ALE mesh "
                "motion' string and no runs-but-does-not-move "
                "mode. (Audit 2026-06-02; corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Input] ALE Dirichlet BCs MUST be applied on "
                "ALL outer fluid boundaries except the FSI "
                "interface (where the mesh follows the "
                "structure). Signal: this failure is SILENT, "
                "not an abort. Dropping the ALE Dirichlet "
                "blocks lets the mesh drift, but the run "
                "completes every time step with no det(J) "
                "complaint, no inverted-element message and no "
                "ALE warning of any kind; what changes is the "
                "answer, and the sharpest tell is that the FSI "
                "interface Lagrange multipliers (RESULT "
                "DESCRIPTION QUANTITY lambdax/lambday/lambdaz) "
                "collapse to round-off, i.e. the interface "
                "transmits no force. Result-test a lambda "
                "component, or you will not notice. The ALE "
                "Dirichlet pins the mesh at fluid-domain "
                "edges. (Audit 2026-06-02; corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Input] CLONING MATERIAL MAP is REQUIRED: it "
                "maps the fluid material ID to a derived ALE "
                "(St. Venant-Kirchhoff pseudo-) material. "
                "Signal: missing CLONING MATERIAL MAP aborts "
                "with 'At least one material pairing required "
                "in --CLONING MATERIAL MAP.' from "
                "core/fem/src/general/utils/"
                "4C_fem_general_utils_createdis.hpp, inside "
                "Core::FE::clone_discretization<"
                "ALE::Utils::AleCloneStrategy> — note the "
                "legacy '--SECTION' spelling in the message, "
                "which is not how you write it in YAML. "
                "(Neither 'cannot clone material for ALE "
                "field' nor 4C_adapter_fld_base_algorithm "
                "appears in that log.) Standard "
                "form: SRC_FIELD: fluid, SRC_MAT: <fluid_id>, "
                "TAR_FIELD: ale, TAR_MAT: <ale_id>. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] SHAPEDERIVATIVES in FSI DYNAMIC/"
                "MONOLITHIC SOLVER adds the derivative of the "
                "fluid residual w.r.t. ALE displacement to the "
                "Jacobian. It is NOT required: 4C declares it "
                "with default_value = false "
                "(src/inpar/4C_inpar_fsi.cpp) and the upstream "
                "monolithic driven-cavity benchmarks do not "
                "set it at all. Signal: leaving it false does "
                "not degrade the monolithic Newton — on the "
                "upstream 2D driven-cavity fluidsplit deck "
                "both settings converge every time step and "
                "match the same pinned results, and turning it "
                "ON costs MORE Newton iterations, not fewer. "
                "Treat it as a tuning knob to try, not a "
                "prerequisite; for partitioned algorithms it "
                "is irrelevant. (Audit 2026-06-02; corrected "
                "by execution 2026-08-06.)"
            ),
            (
                "[Input] Each FSI field (structure, fluid, "
                "ALE) needs its OWN SOLVER N entry, "
                "referenced by LINEAR_SOLVER: N in the "
                "respective DYNAMIC section. Signal: "
                "referencing a SOLVER block that is not "
                "defined produces NO 4C diagnostic — no "
                "'PROC 0 ERROR' block, no MPI_ABORT banner and "
                "no mention of the section at fault. Trilinos "
                "throws Teuchos::Exceptions::"
                "InvalidParameterName ('Error!  The parameter "
                "\"SOLVER\" does not exist in the parameter "
                "(sub)list \"ROOT->SOLVER N\".') uncaught, "
                "std::terminate runs and the process dies on "
                "SIGABRT with exit status 134; the only clue "
                "is the mangled setup_<field> frame in the "
                "signal backtrace. Reusing one SOLVER for all "
                "three fields is "
                "ALLOWED but typically suboptimal (e.g. "
                "structure benefits from CG+ML, fluid from "
                "GMRES+ILU, ALE from direct UMFPACK). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] FSI coupling conditions are "
                "CONVENTIONALLY written on lines in 2D "
                "(DESIGN FSI COUPLING LINE CONDITIONS) and on "
                "surfaces in 3D (DESIGN FSI COUPLING SURF "
                "CONDITIONS), but 4C does not enforce that: "
                "FSI::Monolithic::setup_system looks the "
                "interface up by condition NAME "
                "('FSICoupling'), not by geometry type, so a "
                "3D problem coupled through LINE conditions "
                "runs and gives the same answer provided the "
                "DLINE set exists and holds the interface "
                "nodes. What must match is the TOPOLOGY. "
                "Signal: naming an entity type the deck never "
                "defines is a loud abort, not a silent "
                "decoupling — 'DLine 0 not in range [0:0[' and "
                "'DLine condition on non existent DLine?Could "
                "not read set from entity type.' from "
                "core/fem/src/condition/4C_fem_condition.cpp. "
                "(Audit 2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] Field NUMDOF: structure uses NUMDOF "
                "matching dimension (2 or 3); fluid uses "
                "NUMDOF = dim + 1 (extra DOF is pressure); "
                "ALE uses NUMDOF = dim. Signal: the check is "
                "ONE-SIDED. Its source is "
                "`if (num_dbc_dofs < numdf)` in "
                "Core::FE::Dbc::read_dirichlet_condition "
                "(core/fem/src/discretization/"
                "4C_fem_discretization_utils_dbc.cpp), so too "
                "FEW entries abort with '<given> DOFs given "
                "but <expected> expected in <Point|Line|Surface"
                "|Volume> Dirichlet boundary condition' while "
                "too MANY are accepted in silence and the "
                "surplus is dropped. There is no 'invalid "
                "NUMDOF' string and the message never names "
                "the field. The dangerous direction — a 3D "
                "block pasted into a 2D deck, i.e. "
                "over-declared NUMDOF — is exactly the one 4C "
                "does not catch. (Audit 2026-06-02; corrected "
                "by execution 2026-08-06.)"
            ),
            (
                "[Input] DESIGN LINE DIRICH CONDITIONS in "
                "FSI applies to ALL discretisations "
                "containing a node — structure AND fluid AND "
                "ALE. Signal: a shared node between "
                "structure (NUMDOF=2) and fluid (NUMDOF=3) "
                "hit by a Dirichlet whose NUMDOF suits only one of them is "
                "rejected by the DOF-count check, whose template is "
                "'{} DOFs given but {} expected in {}' — e.g. '3 DOFs given "
                "but 6 expected in Point Dirichlet boundary condition'. "
                "(Verified by execution on a beam deck, which uses the same "
                "check; an earlier version quoted 'NUMDOF mismatch' from "
                "'4C_dofset.cpp', which is not a string the binary "
                "contains.) Only ONE direction of the clash is caught: a "
                "node needing MORE dofs than the condition declares aborts, "
                "the reverse is accepted silently and quietly imposes the "
                "condition on the other field's dofs (verified by execution "
                "on an FSI deck 2026-08-06). "
                "Workarounds: (a) offset structural mesh "
                "slightly to avoid shared nodes, "
                "(b) mortar coupling with non-conforming "
                "meshes, (c) remove structural Dirichlet "
                "and rely on FSI coupling. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] DESIGN FLUID LINE LIFT&DRAG does NOT "
                "exist in 4C for 2D. Signal: writing it in a "
                "2D FSI input raises \"Section 'DESIGN FLUID LINE "
                "LIFT&DRAG' is not a valid section name.\" from "
                "core/io/src/4C_io_input_file.cpp, exit 1 at parse. "
                "(Verified by execution; an earlier version quoted "
                "'unknown section ...' from input_spec_builders.cpp, which "
                "is not a string the binary contains.) DESIGN FLUID SURF "
                "LIFT&DRAG is the only definition that registers the "
                "'LIFTDRAG' condition (src/inpar/4C_inpar_fluid.cpp), and "
                "FLD::Utils::lift_drag() starts by fetching that condition, "
                "so LIFTDRAG: true in FLUID DYNAMIC on its own computes "
                "NOTHING: the run parses, converges and prints no "
                "\"lift'n'drag\" line and no warning. There is no 2D "
                "lift/drag route through a line condition — either build a "
                "surface condition or integrate the traction yourself. "
                "(Audit 2026-06-02; the LIFTDRAG-flag advice corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Syntax] IO section has NO EVERY_ITERATION "
                "parameter — that is not valid in 4C. Signal: "
                "writing 'EVERY_ITERATION: true' in IO "
                "aborts at parse time. Signal: 'Could not match this "
                "input' from core/io/src/4C_io_input_spec_builders.cpp, "
                "echoing the IO block and the candidate specification, "
                "exit 1. (Verified by execution; an earlier version quoted "
                "'unknown parameter EVERY_ITERATION', which is not a string "
                "the binary contains. Note EVERY_ITERATION IS a real key — "
                "it lives in IO/RUNTIME VTK OUTPUT, not in IO.) Use "
                "RESULTSEVERY in each field's DYNAMIC section "
                "(STRUCTURAL DYNAMIC, FLUID DYNAMIC, ALE "
                "DYNAMIC) to control output frequency per "
                "field. (Audit 2026-06-02.)"
            ),
            (
                "[Syntax] In a FUNCT that combines "
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME with a "
                "VARIABLE, the VARIABLE is its OWN list entry, "
                "a sibling of the function entry — never a key "
                "inside the same item, which is what actually "
                "fails to parse ('Could not match this input' "
                "from core/io/src/"
                "4C_io_input_spec_builders.cpp). COMPONENT: 0 "
                "on the function entry is OPTIONAL. Signal: "
                "omitting COMPONENT does NOT silently drop the "
                "VARIABLE — the deck evaluates identically "
                "with and without it, and changing the "
                "variable's VALUES changes the answer either "
                "way, so there is no 'ramp stuck at 0' mode to "
                "look for. SYMBOLIC_FUNCTION_OF_TIME (pure "
                "time) does not need COMPONENT either. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] Monolithic FSI requires SEPARATE "
                "nodes at the FSI interface — structure and "
                "fluid must NOT share nodes. Signal: 4C does "
                "NOT report 'no FSI interface nodes found' — "
                "that string does not exist — and it does not "
                "run uncoupled either. It accepts the mesh, "
                "clones the ale field, builds the coupling and "
                "integrates, then aborts part-way through the "
                "time window with 'Nonlinear solver did not "
                "converge in <ITEMAX> iterations in time step "
                "<n>.' from "
                "fsi/src/monolithic/4C_fsi_monolithic.cpp, in "
                "FSI::Monolithic::non_lin_error_check(). It "
                "reads like a physics or time-step problem, "
                "not a mesh one. (A shared interface node that "
                "also carries a Dirichlet trips the "
                "slave-side check first — see that entry.) "
                "Post-process Gmsh to duplicate "
                "interface nodes and remap fluid connectivity, "
                "OR use mortar coupling "
                "(iter_mortar_monolithicfluidsplit) which "
                "handles non-matching meshes natively. "
                "(Audit 2026-06-02.)"
            ),
            "[API] Which eletype owns 2D structural cells in an FSI deck is "
            "VERSION-DEPENDENT, and the two spellings share no keywords. "
            "4C before 2026.2.0: 'WALL QUAD4 <n..> MAT m KINEM k EAS none THICK t "
            "STRESS_STRAIN s GP 2 2'. 4C 2026.2.0 and later: 'SOLID QUAD4 <n..> MAT m "
            "KINEM k THICKNESS t PLANE_ASSUMPTION p'. Do not assume either; "
            "COPY THE STRUCTURE ELEMENTS LINE from a 2D deck in "
            "tests/input_files of the tree you are running against "
            "(volmortar2D_fsi, fsi_dc_mono_*, contact2D_*). Signal: the "
            "loser is refused at element reading, but by different text in "
            "each era — \"Element 'SOLID' does not seem to know cell type "
            "'quad4'.\" from core/fem/src/general/element/"
            "4C_fem_general_element_definition.cpp on a WALL-era binary, "
            "versus \"Unknown type 'WALL' of finite element\" from "
            "core/comm/src/4C_comm_parobjectfactory.cpp on a SOLID-era one. "
            "The winner reaches fill_complete on the structure "
            "discretization. (Verified empirically 2026-06-01 and re-run "
            "2026-08-06, where the deployed 4C 2026.2.0-dev was in the WALL "
            "era — Tier-2 fixtures structural_2d_solid_quad4_not_wall and "
            "fsi_2d_structural_eletype_is_version_dependent in "
            "scripts/tier2_fixtures/fourc/.)",
            (
                "[Output] IO/RUNTIME VTK OUTPUT/STRUCTURE "
                "CONFLICTS with FSI, and you cannot fix it "
                "from the input. Signal: an FSI input carrying "
                "that section aborts before the first time "
                "step with 'Runtime output is not available in "
                "the old structure time integration! You need "
                "to take the new one, i.e. set `INT_STRATEGY: "
                "Standard`!' from "
                "structure/4C_structure_timint.cpp (not "
                "'inconsistent integration strategy', which "
                "does not exist). Doing exactly what that "
                "message says — INT_STRATEGY: \"Standard\" in "
                "STRUCTURAL DYNAMIC — reproduces the identical "
                "abort, because the FSI adapter has already "
                "overridden the user's choice. Remove the "
                "section and post-process with post_vtu "
                "instead. (Audit 2026-06-02; corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Output] 2D fluid VTK output of 4C before 2026.3.0 "
                "(the 2026.2.0 release included) shows "
                "NaN pressure and a nonzero vz component — this "
                "is a VTK output artifact, NOT divergence. The "
                "vz component is the pressure (measured "
                "2026-09-28 on openPASO's fsi_2d deck: it equals "
                "the 'pressure' array 4C 2026.3.0 writes for the "
                "same run to 7e-15), and 4C 2026.3.0 writes a "
                "correct 'pressure' array and a two-component "
                "velocity. "
                "Signal: with PRESSURE: true in IO/RUNTIME VTK "
                "OUTPUT/FLUID, EVERY point of the written "
                ".vtu pressure array is NaN, and the third "
                "component of the velocity array is nonzero "
                "in a problem that has no third "
                "dimension — while the same run converges and "
                "passes a RESULT DESCRIPTION test on fluid "
                "pressure to 1e-10. Note displacement and "
                "grid-velocity keep component 2 identically "
                "zero, so the writer is not just padding every "
                "vector. Check vx/vy (correct in 2D), the "
                "convergence log, and a pressure result test — "
                "the issue is output, not solve. (Audit "
                "2026-06-02; confirmed by decoding the .vtu "
                "2026-08-06.)"
            ),
            (
                "[Mesh] For complex FSI geometries (e.g. flag "
                "attached to cylinder): offset the flag "
                "slightly (e.g. 0.1mm gap) to avoid Gmsh "
                "fragment operations that create non-quad-"
                "meshable surfaces. Signal: a flag glued to "
                "a cylinder produces a degenerate "
                "intersection edge that Gmsh can only mesh "
                "with TRI3 (not QUAD4) — typically 100x more "
                "elements than a clean offset geometry; or "
                "Gmsh aborts with 'cannot quad-mesh non-"
                "planar fragment'. A 0.1mm gap is a "
                "negligible geometric approximation that "
                "vastly simplifies meshing. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] FSI SLAVE interface CANNOT carry "
                "Dirichlet BCs. Signal: with "
                "iter_monolithicstructuresplit "
                "(structure=slave), a structural Dirichlet "
                "on a node that also belongs to the FSI "
                "coupling interface aborts with a boxed banner "
                "headed 'DIRICHLET BOUNDARY CONDITIONS ON "
                "SLAVE SIDE OF FSI INTERFACE', containing 'The "
                "slave side of the interface is not allowed to "
                "carry Dirichlet boundary conditions.' and "
                "spelling out the split ('MASTER  = FLUID', "
                "'SLAVE   = STRUCTURE'), from fsi/src/"
                "monolithic/model_evaluator/"
                "4C_fsi_monolithicstructuresplit.cpp — note the "
                "file name has no underscore before "
                "'structuresplit', and 'slave node carries "
                "Dirichlet' is not a string 4C contains. Fix: "
                "exclude the overlapping nodes from the FSI "
                "interface, or switch the split ONLY after "
                "checking the other field — swapping to "
                "iter_monolithicfluidsplit makes the FLUID the "
                "slave, and if its interface nodes sit inside "
                "any domain-wide Dirichlet (a DESIGN VOL/SURF "
                "DIRICH covering the whole fluid) you get the "
                "mirror-image banner from "
                "4C_fsi_monolithicfluidsplit.cpp instead. The "
                "rule is: whichever field is the SLAVE must "
                "have a Dirichlet-free interface. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Output] IO/RUNTIME VTK OUTPUT/ALE does NOT "
                "exist. Signal: writing /ALE as a subsection "
                "is rejected at parse by the ordinary "
                "invalid-section check, which quotes the WHOLE "
                "path as one name: \"Section 'IO/RUNTIME VTK "
                "OUTPUT/ALE' is not a valid section name.\" "
                "from core/io/src/4C_io_input_file.cpp. There "
                "is no 'unknown subsection' message, no "
                "candidate list and no 'did you mean' — "
                "nothing in the output tells you /FLUID or "
                "/STRUCTURE would have worked, so you have to "
                "know it. Only /STRUCTURE and /FLUID "
                "subsections are valid for FSI VTK output. For "
                "ALE fields, use post_processor --filter=vtu "
                "on native output instead. (Audit 2026-06-02; "
                "corrected by execution 2026-08-06.)"
            ),
            (
                "[Input] Valid COUPALGO values for monolithic "
                "FSI: iter_monolithicfluidsplit "
                "(structure=master, recommended), "
                "iter_monolithicstructuresplit "
                "(structure=slave), "
                "iter_mortar_monolithicfluidsplit (non-"
                "matching meshes), "
                "iter_sliding_monolithicfluidsplit (sliding "
                "interface). For partitioned the names end in "
                "_rel_param, NOT _rel_force: "
                "iter_stagg_AITKEN_rel_param, "
                "iter_stagg_fixed_rel_param (an earlier "
                "version of this entry gave both as "
                "..._rel_force, which 4C rejects). Signal: a "
                "mis-spelled COUPALGO aborts at parse with "
                "'Could not match this input' from "
                "core/io/src/4C_io_input_spec_builders.cpp — "
                "not 'unknown coupling algorithm', and not "
                "from 4C_fsi_adapter.cpp, which does not exist "
                "— and the abort PRINTS THE AUTHORITATIVE "
                "LIST: \"Candidate deprecated_selection "
                "'COUPALGO' has wrong value, possible values: "
                "...\". Read the enumeration out of that line "
                "rather than trusting any written list. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Numerical] Inflow ramp rate costs Newton "
                "iterations and changes the answer; it does "
                "not, on its own, break the solve. Signal: on "
                "the upstream monolithic 2D driven-cavity "
                "benchmark, replacing the shipped 5 s cosine "
                "ramp with an INSTANTANEOUS step to the same "
                "peak amplitude still completes every time "
                "step and prints no divergence or "
                "non-convergence message at all — it simply "
                "takes substantially more Newton iterations "
                "and lands on a different solution, so the "
                "pinned RESULT DESCRIPTION values all move. Do "
                "not expect 'Newton divergence within ~10 time "
                "steps' as the tell; watch the per-step "
                "iteration count and result values instead. A "
                "slow ramp (5-10 s period, e.g. cos(pi*t/5)) "
                "is still the cheaper starting point. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
        ],
    },

    "ssi": {
        "description": "Structure-Scalar Interaction (e.g., battery electrode mechanics)",
        "problemtype": "Structure_Scalar_Interaction",
        # SSI CONTROL/COUPALGO, spelled exactly so; the default is ssi_IterStagg.
        "coupling_types": ["ssi_OneWay_ScatraToSolid", "ssi_OneWay_SolidToScatra",
                          "ssi_IterStagg",
                          "ssi_IterStaggFixedRel_ScatraToSolid", "ssi_IterStaggFixedRel_SolidToScatra",
                          "ssi_IterStaggAitken_ScatraToSolid", "ssi_IterStaggAitken_SolidToScatra",
                          "ssi_Monolithic"],
    },

    "ssti": {
        "description": "Structure-Scalar-Thermo Interaction (three-field coupling)",
        "problemtype": "Structure_Scalar_Thermo_Interaction",
        "coupling": "Monolithic (all three fields simultaneously)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CONTACT MECHANICS
    # ═══════════════════════════════════════════════════════════════════════
    "contact": {
        # ---- READ THIS FIRST ----
        "description": (
            "Mortar contact between two deformable bodies. Contact is NOT a "
            "problem type: PROBLEMTYPE stays 'Structure'. Contact adds "
            "exactly THREE sections to an otherwise ordinary structural "
            "deck, and all three are mandatory:\n"
            "  1. CONTACT DYNAMIC  — must set LINEAR_SOLVER; must set "
            "PENALTYPARAM if STRATEGY is Penalty\n"
            "  2. MORTAR COUPLING  — REQUIRED whenever STRATEGY is anything "
            "other than the default Lagrange (Penalty, Nitsche, ...), "
            "and then one line is enough: LM_DUAL_CONSISTENT: "
            "\"none\". Under the DEFAULT STRATEGY (Lagrange) the "
            "section may be omitted entirely or left empty and the run "
            "still works.\n"
            "  3. DESIGN SURF MORTAR CONTACT CONDITIONS 3D (2D: DESIGN LINE "
            "MORTAR CONTACT CONDITIONS 2D) — EXACTLY ONE entry with "
            "Side: \"Master\" and EXACTLY ONE with Side: \"Slave\", both "
            "carrying the SAME InterfaceID\n"
            "CONTACT DYNAMIC/LINEAR_SOLVER may point at the SAME SOLVER n "
            "block the structure uses — a separate contact solver is "
            "conventional but NOT required.\n"
            "The deck below is complete and runs as written."
        ),
        "problemtype": "Structure",
        "yaml_sections": [
            "CONTACT DYNAMIC",
            "MORTAR COUPLING",
            "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
            "DESIGN LINE MORTAR CONTACT CONDITIONS 2D",
        ],

        # ---- COMPLETE RUNNABLE DECK. Copy this whole thing. ----
        "minimal_working_input_3d": """\
# Complete 3D contact deck: two unit cubes, the upper one pressed into the
# lower one across an initial 0.1 gap. Nothing external is needed - no mesh
# file, no include. Runs to completion; the contact active set becomes
# non-empty from the load step at which the prescribed displacement closes
# the gap.
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1            # load-step size; SHRINK THIS FIRST if Newton fails
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:           # (1) REQUIRED whenever a contact condition exists
  LINEAR_SOLVER: 2         #     REQUIRED - may reuse id 1
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4      #     REQUIRED for Penalty; 0 is rejected
MORTAR COUPLING:           # (2) REQUIRED for Penalty/Nitsche (this deck).
  LM_DUAL_CONSISTENT: "none"  #    Under the DEFAULT Lagrange strategy the
                              #    whole section may be omitted or left empty.
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:                  # optional: CONTACT DYNAMIC could reuse SOLVER 1
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1                   # bottom face of the lower block: clamped
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 4                   # top face of the upper block: pushed down
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, -0.3]
    FUNCT: [0, 0, 1]
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:  # (3) one Master, one Slave
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 9
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e30    # record mode: prints abs(diff) = the true value
""",

        # ---- COMPLETE RUNNABLE 2D DECK. Copy this whole thing. ----
        # Do NOT try to derive this from the 3D deck by renaming sections:
        # a 2D mesh is a different mesh (every z must be 0), so the node
        # list, the connectivity and the design-entity map all change too.
        # This deck was run as written.
        "minimal_working_input_2d": """\
# 4C before 2026.2.0 (WALL). For 4C 2026.2.0 and later use minimal_working_input_2d_solid.
# Complete 2D plane-strain contact deck: two unit squares, the upper one
# pressed into the lower one across an initial 0.1 gap. Self-contained.
PROBLEM SIZE:
  DIM: 2                   # REQUIRED in 2D; without it the mortar search
PROBLEM TYPE:              # fails with 'auxiliary_plane called for unknown
  PROBLEMTYPE: "Structure" # element type'
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:     # LINE, not SURF, in 2D
  - E: 1
    NUMDOF: 2                      # 2 in 2D
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, -0.3]
    FUNCT: [0, 1]
DESIGN LINE MORTAR CONTACT CONDITIONS 2D:   # '... 2D', not '... 3D'
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DLINE-NODE TOPOLOGY:               # DLINE, not DSURFACE
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
  - "NODE 3 DLINE 2"
  - "NODE 4 DLINE 2"
  - "NODE 5 DLINE 3"
  - "NODE 6 DLINE 3"
  - "NODE 7 DLINE 4"
  - "NODE 8 DLINE 4"
NODE COORDS:                       # every z MUST be 0.0 in a 2D problem
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 1.1 0.0"
  - "NODE 6 COORD 1.0 1.1 0.0"
  - "NODE 7 COORD 1.0 2.1 0.0"
  - "NODE 8 COORD 0.0 2.1 0.0"
STRUCTURE ELEMENTS:                # WALL, not SOLID, and all six keys
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
  - "2 WALL QUAD4 5 6 7 8 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 5
      QUANTITY: "dispy"
      VALUE: 0.0
      TOLERANCE: 1.0e30
""",

        # The same contact deck for 4C 2026.2.0 and later. Run on 2026.2.0 and
        # 2026.3.0: both record the same dispy at node 5 as the WALL deck on a
        # 2026.2.0 development build (0.2043361023, relative difference 5e-15).
        "minimal_working_input_2d_solid": """\
# 4C 2026.2.0 and later (no WALL). Complete 2D plane-strain contact deck: two unit squares, the upper one
# pressed into the lower one across an initial 0.1 gap. Self-contained.
PROBLEM SIZE:
  DIM: 2                   # REQUIRED in 2D; without it the mortar search
PROBLEM TYPE:              # fails with 'auxiliary_plane called for unknown
  PROBLEMTYPE: "Structure" # element type'
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:     # LINE, not SURF, in 2D
  - E: 1
    NUMDOF: 2                      # 2 in 2D
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, -0.3]
    FUNCT: [0, 1]
DESIGN LINE MORTAR CONTACT CONDITIONS 2D:   # '... 2D', not '... 3D'
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DLINE-NODE TOPOLOGY:               # DLINE, not DSURFACE
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
  - "NODE 3 DLINE 2"
  - "NODE 4 DLINE 2"
  - "NODE 5 DLINE 3"
  - "NODE 6 DLINE 3"
  - "NODE 7 DLINE 4"
  - "NODE 8 DLINE 4"
NODE COORDS:                       # every z MUST be 0.0 in a 2D problem
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 1.1 0.0"
  - "NODE 6 COORD 1.0 1.1 0.0"
  - "NODE 7 COORD 1.0 2.1 0.0"
  - "NODE 8 COORD 0.0 2.1 0.0"
STRUCTURE ELEMENTS:                # 4C 2026.2.0 and later: SOLID with THICKNESS and PLANE_ASSUMPTION
  - "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"
  - "2 SOLID QUAD4 5 6 7 8 MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 5
      QUANTITY: "dispy"
      VALUE: 0.0
      TOLERANCE: 1.0e30
""",

        "what_differs_between_the_2d_and_3d_decks": (
            "CONTACT DYNAMIC and MORTAR COUPLING are byte-identical. "
            "Everything else changes:\n"
            "  PROBLEM SIZE: DIM: 2                    (added)\n"
            "  DESIGN SURF MORTAR CONTACT CONDITIONS 3D -> "
            "DESIGN LINE MORTAR CONTACT CONDITIONS 2D\n"
            "  DESIGN SURF DIRICH CONDITIONS -> DESIGN LINE DIRICH CONDITIONS\n"
            "  DSURF-NODE TOPOLOGY -> DLINE-NODE TOPOLOGY, DSURFACE n -> DLINE n\n"
            "  SOLID HEX8 (2 required keys) -> WALL QUAD4 (6 required keys) on "
            "4C before 2026.2.0, SOLID QUAD4 with THICKNESS and PLANE_ASSUMPTION "
            "(4 required keys) on 4C 2026.2.0 and later\n"
            "  and THE MESH ITSELF: 8 nodes instead of 16, all with z = 0.0, "
            "different connectivity, different design-entity map. A 3D node "
            "list carried over unchanged aborts with 'Node <id> has a "
            "non-zero coordinate <value> in direction 2 but "
            "discretization is 2D!', naming the first offending node.\n"
            "NUMDOF/ONOFF/VAL/FUNCT of 3 entries are TOLERATED in a 2D DIRICH "
            "block, so that is not what breaks."
        ),

        # ---- REQUIRED vs OPTIONAL, stated, not implied ----
        "contact_dynamic_keys": {
            "_note": (
                "Section CONTACT DYNAMIC, 37 keys. `4C --parameters` marks "
                "EVERY key required:false, but two of them are required in "
                "practice because their defaults are rejected at setup: "
                "LINEAR_SOLVER (default -1) and, under STRATEGY Penalty, "
                "PENALTYPARAM (default 0)."
            ),
            "LINEAR_SOLVER": (
                "REQUIRED IN PRACTICE (schema default -1 is rejected); "
                "verified by omission. Note the message names CONTACT "
                "DYNAMIC specifically -- each field has its own variant "
                "naming its own section. "
                "Integer id of a SOLVER n block. MAY be the same id "
                "STRUCTURAL DYNAMIC uses; a separate contact solver is not "
                "required. Pointing at an id with no SOLVER block is NOT a "
                "clean error — see pitfalls."
            ),
            "STRATEGY": (
                "Optional, default 'Lagrange'. Working choices: Lagrange / "
                "LagrangianMultipliers, Penalty, Nitsche. 'Uzawa' contact "
                "runs only with STRUCTURAL DYNAMIC INT_STRATEGY Old; under "
                "the default Standard it aborts (mortar meshtying takes "
                "Uzawa under Standard too). Lowercase aliases "
                "'lagrange' and 'penalty' are accepted."
            ),
            "PENALTYPARAM": (
                "Optional in the schema (default 0) but REQUIRED under "
                "STRATEGY Penalty / Nitsche — 0 is explicitly rejected. For "
                "Penalty it is a physical interface stiffness with units and "
                "scales with the material's YOUNG. For Nitsche with the "
                "default NITSCHE_PENALTY_ADAPTIVE: true it is a much smaller "
                "dimensionless multiplier; a value that works for Penalty is "
                "typically far too large for Nitsche."
            ),
            "PENALTYPARAMTAN": (
                "Optional, default 0, but REQUIRED once FRICTION is set to "
                "anything but None — this is the tangential penalty, and "
                "there is no separate friction-coefficient key in this "
                "section (the coefficient lives on the condition, as "
                "FrCoeffOrBound)."
            ),
            "SYSTEM": (
                "Optional, default 'Condensed'. Choices: Condensed, "
                "Condensedlagmult, SaddlePoint (plus lowercase aliases). "
                "Condensed requires LM_SHAPEFCN: Dual."
            ),
            "FRICTION": "Optional, default 'None'. Choices: None, Coulomb, Stick, Tresca.",
            "SEMI_SMOOTH_NEWTON": (
                "Optional, default true. LEAVE IT TRUE — setting it false "
                "aborts, the non-semi-smooth path is not supported."
            ),
            "TOLCONTCONSTR": "Optional, default 1e-06. Contact-constraint convergence tolerance.",
            "TOLLAGR": "Optional, default 1e-06. Lagrange-multiplier convergence tolerance.",
            "ADHESION": "Optional, default 'none'. Choices: none, bounded.",
            "INITCONTACTBYGAP": "Optional, default false. Initialise the active set from an initial gap.",
            "INITCONTACTGAPVALUE": "Optional, default 0. Gap threshold when INITCONTACTBYGAP is true.",
            "NITSCHE_THETA": "Optional, default 0. Nitsche symmetry parameter (0 = non-symmetric).",
            "NITSCHE_WEIGHTING": "Optional, default 'harmonic'. Choices: slave, master, harmonic, physical.",
            "NITSCHE_PENALTY_ADAPTIVE": "Optional, default true. Scales PENALTYPARAM by element stiffness/size.",
            "CONSTRAINT_DIRECTIONS": "Optional, default 'ntt'. Choices: vague, ntt, xyz.",
            "NONSMOOTH_GEOMETRIES": "Optional, default false. Enable edge/corner (LTL, NTS) treatment.",
            "REGULARIZED_NORMAL_CONTACT": "Optional, default false. Regularised normal law.",
            "RESTART_WITH_CONTACT": "Optional, default false. Start contact from a contact-free restart.",
            "_full_key_count": 37,
        },

        "mortar_coupling_keys": {
            "_note": (
                "Section MORTAR COUPLING, 15 keys, all optional in the "
                "schema. Under the DEFAULT STRATEGY (Lagrange) the section "
                "really is optional — omitting it or writing it empty runs "
                "fine. Under any OTHER strategy it becomes mandatory, "
                "because the DEFAULT "
                "combination (LM_SHAPEFCN: Dual + LM_DUAL_CONSISTENT: "
                "boundary) is invalid for every STRATEGY except Lagrange. "
                "Minimum for Penalty/Nitsche: LM_DUAL_CONSISTENT: \"none\"."
            ),
            "LM_SHAPEFCN": (
                "Default 'Dual'. Choices: Dual, Standard, PetrovGalerkin "
                "(plus lowercase aliases). Dual is what SYSTEM: Condensed "
                "needs; Standard is one of the two escapes from the "
                "dual-consistency check."
            ),
            "LM_DUAL_CONSISTENT": (
                "Default 'boundary'. Choices: all, boundary, none. Must be "
                "'none' for any non-Lagrange strategy while LM_SHAPEFCN is "
                "Dual."
            ),
            "ALGORITHM": (
                "Default 'Mortar'. Choices: Mortar, GPTS, NTS, LTS, LTL, STL "
                "(plus lowercase). STRATEGY Nitsche accepts ONLY GPTS. GPTS "
                "is not Nitsche-exclusive: Penalty + GPTS also runs."
            ),
            "SEARCH_ALGORITHM": "Default 'BinaryTree'. Choices: BinaryTree, BruteForce, BruteForceEleBased.",
            "SEARCH_PARAM": "Default 0.3. Search-box inflation factor.",
            "INTTYPE": "Default 'Segments'. Choices: Segments, Elements, Elements_BS.",
            "NUMGP_PER_DIM": "Default 0 (= automatic). Gauss points per direction on the interface.",
            "TRIANGULATION": "Default 'Delaunay'. Choices: Delaunay, Center.",
            "LM_QUAD": (
                "Default 'undefined'. Multiplier order on the interface: "
                "quad, lin, piecewiselinear, const. The lin/const settings "
                "need quadratic mortar elements (line3/tri6/quad8/quad9) and "
                "abort on linear ones."
            ),
            "MESH_RELOCATION": "Default 'Initial'. Choices: Initial, None.",
            "CROSSPOINTS": "Default false.",
            "OUTPUT_INTERFACES": "Default false. Write the mortar interfaces as separate output.",
            "BINARYTREE_UPDATETYPE": "Default 'BottomUp'. Choices: BottomUp, TopDown.",
            "SEARCH_USE_AUX_POS": "Default true.",
            "RESTART_WITH_MESHTYING": "Default false.",
        },

        "contact_condition_keys": {
            "_note": (
                "One list entry per interface SIDE, in section "
                "DESIGN SURF MORTAR CONTACT CONDITIONS 3D (3D) or "
                "DESIGN LINE MORTAR CONTACT CONDITIONS 2D (2D). Minimum two "
                "entries: one Master, one Slave, same InterfaceID."
            ),
            "InterfaceID": "REQUIRED. Integer. Master and Slave of one interface must match.",
            "Side": (
                "REQUIRED. Choices: Master, Slave, Selfcontact. "
                "CASE-SENSITIVE — 'slave' is rejected even though "
                "STRATEGY: 'penalty' lowercase is accepted."
            ),
            "E": (
                "Design-entity id, for inline meshes. Give EITHER E (+ "
                "optional ENTITY_TYPE) OR NODE_SET_NAME, never both."
            ),
            "NODE_SET_NAME": (
                "Node-set name in the external Exodus file, for the "
                "STRUCTURE GEOMETRY mesh path. Mutually exclusive with E."
            ),
            "ENTITY_TYPE": "Optional. Choices: legacy_id, node_set_id, element_block_id.",
            "Initialization": "Optional, default 'Inactive'. Choices: Active, Inactive.",
            "FrCoeffOrBound": "Optional, default 0. Friction coefficient / Tresca bound.",
            "AdhesionBound": "Optional, default 0.",
            "Application": "Optional, default 'Solidcontact'. Choices: Solidcontact, Beamtosolidcontact, Beamtosolidmeshtying.",
            "TwoHalfPass": "Optional, default 0.",
            "ConstitutiveLawID": "Optional. Id into CONTACT CONSTITUTIVE LAWS.",
        },

        "strategy_recipes": {
            "Penalty (simplest)": (
                "CONTACT DYNAMIC: STRATEGY: \"Penalty\", PENALTYPARAM: <k>, "
                "LINEAR_SOLVER: <id>  +  MORTAR COUPLING: "
                "LM_DUAL_CONSISTENT: \"none\". Nothing else. Choose <k> on "
                "the order of the material's YOUNG."
            ),
            "Lagrange, condensed (the default STRATEGY)": (
                "CONTACT DYNAMIC: STRATEGY: \"Lagrange\", LINEAR_SOLVER: "
                "<id>  +  MORTAR COUPLING: LM_SHAPEFCN: \"Dual\". No penalty "
                "parameter, no parameter to tune, and the constraint is "
                "satisfied exactly rather than approximately."
            ),
            "Lagrange, saddle point": (
                "CONTACT DYNAMIC: STRATEGY: \"Lagrange\", SYSTEM: "
                "\"SaddlePoint\", LINEAR_SOLVER: <id>. Then LM_SHAPEFCN may "
                "stay Standard."
            ),
            "Nitsche": (
                "CONTACT DYNAMIC: STRATEGY: \"Nitsche\", PENALTYPARAM: <k>, "
                "LINEAR_SOLVER: <id>  +  MORTAR COUPLING: ALGORITHM: "
                "\"GPTS\", LM_DUAL_CONSISTENT: \"none\", NUMGP_PER_DIM: 1, "
                "TRIANGULATION: \"Center\"  +  a material that supplies a "
                "Cauchy-stress derivative (MAT_ElastHyper + "
                "ELAST_CoupNeoHooke). MAT_Struct_StVenantKirchhoff does NOT "
                "work with Nitsche. <k> here is orders of magnitude SMALLER "
                "than the Penalty-strategy value for the same material, "
                "because NITSCHE_PENALTY_ADAPTIVE scales it."
            ),
        },

        "methods": {
            "penalty": "Penalty method (simple, one parameter, approximate constraint)",
            "lagrange": "Lagrange multiplier (exact constraint, saddle-point or condensed)",
            "nitsche": "Nitsche method (consistent, no extra DOFs, needs ALGORITHM GPTS)",
            "mortar": "Mortar surface integration; the default ALGORITHM under all of the above",
        },
        "variants": ["Standard contact", "Self-contact (Side: Selfcontact)",
                     "Wear contact", "Friction (Coulomb / Tresca / Stick)",
                     "TSI contact", "Poro contact", "FSI contact", "SSI contact"],
        "constitutive_laws": ["Linear", "Cubic", "Power law", "Broken rational",
                              "MIRCO (microscale)", "Python surrogate"],

        # ---- PITFALLS, each attached to the section it concerns ----
        "pitfalls": [
            (
                "[Input] DESIGN * MORTAR CONTACT CONDITIONS: THE DANGEROUS ONE: a "
                "deck with CONTACT DYNAMIC and MORTAR COUPLING but NO "
                "contact-condition section runs to completion, exit 0, with "
                "no warning and no contact. The word 'contact' does not "
                "appear in the output at all — no 'Building contact "
                "interface(s)', no strategy banner — and the two bodies pass "
                "through each other. This looks exactly like success. Signal: "
                "ABSENCE of the lines 'Building contact interface(s)' and "
                "'fill_complete() on discretization mortar_interface_1', "
                "which a correct contact run always prints. Check for those "
                "before believing a contact result. (Verified by execution: "
                "the same deck with the condition section deleted ran 10/10 "
                "steps, rc=0, zero occurrences of 'contact' or 'mortar' in "
                "the log.)"
            ),
            (
                "[Input] CONTACT DYNAMIC: LINEAR_SOLVER is REQUIRED even though "
                "`4C --parameters` reports required:false with default -1. "
                "Omitting it — or omitting the whole CONTACT DYNAMIC section "
                "while a contact condition is present — aborts identically. "
                "It may reuse the structural solver id. Signal: 'no linear "
                "solver defined for meshtying/contact problem. Please set "
                "LINEAR_SOLVER in CONTACT DYNAMIC to a valid number!' "
                "(Verified on HEX8, TET4 and 2D QUAD4 meshes.)"
            ),
            (
                "[Input] CONTACT DYNAMIC: Pointing LINEAR_SOLVER at an id that has "
                "no SOLVER n block does NOT produce a 4C diagnostic. The "
                "process dies with a raw C++ abort and SIGABRT, so there is "
                "no 'PROC 0 ERROR' block and no stack trace in 4C's own "
                "format. Signal: 'terminate called after throwing an "
                "instance of Teuchos::Exceptions::InvalidParameterName' and "
                "'Error!  The parameter \"SOLVER\" does not exist', with a "
                "shell exit status of 134. Cross-check that every "
                "LINEAR_SOLVER id in the deck has a matching SOLVER block."
            ),
            (
                "[Input] CONTACT DYNAMIC: PENALTYPARAM defaults to 0 and 0 is "
                "rejected, so under STRATEGY: \"Penalty\" the key is "
                "effectively required. The same holds for PENALTYPARAMTAN as "
                "soon as FRICTION is anything but None. Signal: both "
                "confirmed by running the wrong variant: 'Penalty parameter "
                "eps = 0, must be greater than 0' and 'Tangential penalty "
                "parameter eps = 0, must be greater than 0'."
            ),
            (
                "[Input] MORTAR COUPLING: The DEFAULT MORTAR COUPLING settings are "
                "INVALID for every STRATEGY except Lagrange. Omitting the "
                "section, writing it empty as 'MORTAR COUPLING: {}', or "
                "writing it with defaults (LM_SHAPEFCN: Dual, "
                "LM_DUAL_CONSISTENT: boundary) all abort at setup under "
                "Penalty, Nitsche, Ehl or MultiScale — but ALL THREE ARE "
                "FINE under the default Lagrange strategy, which was "
                "verified by running a Lagrange deck with no MORTAR "
                "COUPLING section at all and again with an empty one. Two "
                "independent "
                "one-line fixes, both confirmed: set 'LM_DUAL_CONSISTENT: "
                "\"none\"' OR set 'LM_SHAPEFCN: \"Standard\"'. Signal: "
                "'Consistent dual shape functions in boundary elements only "
                "for Lagrange multiplier strategy.' (The four combinations "
                "of LM_SHAPEFCN x LM_DUAL_CONSISTENT were run under "
                "STRATEGY Penalty on HEX8, TET4, QUAD4 and TRI3 meshes; only "
                "Dual+boundary aborts, and Lagrange+boundary runs fine.)"
            ),
            (
                "[Input] CONTACT DYNAMIC: STRATEGY: \"Lagrange\" is the DEFAULT, and "
                "its default SYSTEM: \"Condensed\" refuses to run with "
                "standard shape functions. A deck that sets no STRATEGY and "
                "no LM_SHAPEFCN is fine; a deck that sets 'LM_SHAPEFCN: "
                "\"Standard\"' and leaves STRATEGY at its default aborts. Two "
                "fixes, both verified: 'LM_SHAPEFCN: \"Dual\"' in MORTAR "
                "COUPLING, or 'SYSTEM: \"SaddlePoint\"' in CONTACT DYNAMIC. "
                "Signal: 'Condensation of linear system only possible for "
                "dual Lagrange multipliers', raised by "
                "CONTACT::STRATEGY::Factory::read_and_check_input in "
                "4C_contact_strategy_factory.cpp. Setting LM_QUAD to escape the "
                "check is not a third fix: on linear elements it aborts with "
                "'Lin/Lin interpolation of LM only for line3/tri6/quad8/"
                "quad9 mortar elements'."
            ),
            (
                "[Input] DESIGN * MORTAR CONTACT CONDITIONS: The list needs EXACTLY "
                "ONE Side: \"Master\" and EXACTLY ONE Side: \"Slave\" per "
                "InterfaceID, both carrying the SAME InterfaceID. Each way "
                "of getting this wrong has its own diagnostic. Signal: all four "
                "confirmed by triggering them: two Masters and no Slave -> "
                "'Slave side missing in contact condition group!'; two "
                "Slaves and no Master -> 'Master side missing in contact "
                "condition group!'; a single entry of either kind -> 'Not "
                "enough contact conditions in discretization'; Master with "
                "InterfaceID 1 and Slave with InterfaceID 2 -> 'Cannot find "
                "matching contact condition for id'. Note what is NOT "
                "checked: THREE entries in one group, InterfaceID 0, and "
                "contact surfaces that do not face each other all run to "
                "completion with exit 0."
            ),
            (
                "[Input] DESIGN * MORTAR CONTACT CONDITIONS: 'Side' is "
                "case-sensitive: Master / Slave / Selfcontact. Writing "
                "'slave' is rejected — even though 'STRATEGY: \"penalty\"' "
                "IS accepted in lowercase, so the casing rule is not uniform "
                "across the deck. Signal: 'Failed to match condition "
                "specification in section 'DESIGN SURF MORTAR CONTACT "
                "CONDITIONS 3D'.'"
            ),
            (
                "[Numerical] STRUCTURAL DYNAMIC: When contact Newton fails, SHRINK THE "
                "LOAD STEP — do not lower PENALTYPARAM. The failure mode is "
                "active-set chatter, not a stiff-system stall: the trace "
                "alternates forever between two states, one with a non-zero "
                "Contact-Normal-Active-Set-Size and a large residual and one "
                "with an empty active set and a small residual, with the "
                "update norm pinned at a constant. Raising MAXITER does not "
                "help, and it is not monotone in the load (a LARGER "
                "prescribed displacement can converge where a smaller one "
                "does not). Verified: a deck that exhausts MAXITER at "
                "TIMESTEP 0.1 completes every step, unchanged in all other "
                "respects, at TIMESTEP 0.01. Lowering PENALTYPARAM also "
                "makes it converge, but by allowing more penetration — it "
                "buys convergence with accuracy, so reach for it second, not "
                "first. Signal: 'The nonlinear solver did not converge!' "
                "from 4C_solver_nonlin_nox_problem.cpp, preceded by "
                "'Failed.......Number of Iterations = <MAXITER> < <MAXITER>'."
            ),
            (
                "[Numerical] STRUCT NOX: A line search does NOT rescue a failing "
                "contact Newton. Backtrack, Polynomial and Full Step all "
                "behave identically on a chattering active set, with the "
                "step length staying at 1.0 throughout. The STRUCT NOX "
                "section IS being read — a bogus 'Method' value is rejected "
                "with 'Could not match this input' — so a silent no-op is "
                "not the explanation. Signal: the per-iteration trace prints "
                "'step = 1.00000e+00' unchanged for every line-search "
                "method and the run still ends in 'The nonlinear solver "
                "did not converge!'. Shrink the time step instead."
            ),
            (
                "[Input] CONTACT DYNAMIC: STRATEGY: \"Nitsche\" needs THREE things "
                "Penalty does not, and each has its own message, so the "
                "Signal: below tells you which one is missing. (a) "
                "Without 'ALGORITHM: \"GPTS\"' in MORTAR COUPLING it aborts "
                "with 'Unrecognized strategy: "
                "\"CONTACT::SolvingStrategy::nitsche\"' — the same message "
                "appears for ALGORITHM NTS, LTS and Mortar, so GPTS is the "
                "only accepted value. (b) With GPTS but a "
                "St.-Venant-Kirchhoff material it aborts with "
                "'evaluate_cauchy_n_dir_and_derivatives not implemented for "
                "material of type' — use MAT_ElastHyper with "
                "ELAST_CoupNeoHooke. (c) The PENALTYPARAM that works for the "
                "mortar Penalty strategy is far too large for Nitsche, "
                "because NITSCHE_PENALTY_ADAPTIVE rescales it; a Nitsche "
                "deck that only fails to converge is usually asking for a "
                "much smaller value."
            ),
            (
                "[Input] CONTACT DYNAMIC: 'Uzawa' contact runs only with the "
                "old structural time integration (STRUCTURAL DYNAMIC "
                "INT_STRATEGY: Old, as upstream's contact3D_lin_uzawa deck "
                "sets it); under INT_STRATEGY Standard, the default, a "
                "contact deck with STRATEGY Uzawa aborts before the first "
                "step, while mortar meshtying takes Uzawa under Standard. "
                "Signal: 'This contact strategy is not yet considered!'. "
                "(Measured on a development build before 4C 2026.2.0, on "
                "4C 2026.2.0 and on 4C 2026.3.0.) Likewise "
                "'SEMI_SMOOTH_NEWTON: false' is a valid boolean the code "
                "refuses: 'Currently we support only the semi-smooth Newton "
                "case!'. Being present in `--parameters` is necessary, not "
                "sufficient."
            ),
            (
                "[Input] PROBLEM SIZE: A 2D contact deck that is otherwise correct "
                "but omits 'PROBLEM SIZE:\\n  DIM: 2' gets all the way into "
                "the mortar search before failing, so the message points at "
                "geometry rather than at the missing key. Signal: "
                "'auxiliary_plane called for unknown element type'."
            ),
            (
                "[Input] DESIGN * MORTAR CONTACT CONDITIONS: The dimension suffix "
                "is part of the section name and is not optional: "
                "'DESIGN SURF MORTAR CONTACT CONDITIONS 3D' and "
                "'DESIGN LINE MORTAR CONTACT CONDITIONS 2D'. There is no "
                "spelling without the suffix. Signal: any wrong spelling is "
                "caught before anything runs, with \"Section '<what you "
                "wrote>' is not a valid section name.\" and exit 1."
            ),
            (
                "[Output] IO: To get contact tractions into the output, "
                "'IO/RUNTIME VTK OUTPUT/STRUCTURE' has a dedicated flag "
                "'OUTPUT_CONTACT: true' alongside DISPLACEMENT. As with all "
                "runtime VTK, the parent section 'IO/RUNTIME VTK OUTPUT' "
                "with INTERVAL_STEPS must ALSO be present, or nothing is "
                "written at all. Signal: there is no error either "
                "way - the detector is the ABSENCE of a "
                "<prefix>-vtk-files/ directory next to the output prefix."
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PARTICLE METHODS
    # ═══════════════════════════════════════════════════════════════════════
    "particles": {
        "description": "Particle methods: SPH, DEM, Peridynamics",
        "problemtype": "Particle",

        "sph": {
            "kernels": ["CubicSpline (default)", "QuinticSpline"],
            "eos": ["GenTait (generalized Tait)", "IdealGas"],
            "momentum": ["Adami formulation", "Monaghan formulation"],
            "density": ["Summation", "Integration", "Predict-Correct"],
            "boundary": ["Adami boundary particles", "Virtual wall particles"],
            "extra_physics": ["Surface tension (CSF)", "Phase change", "Temperature"],
        },

        "dem": {
            "contact_normal": ["LinearSpring", "LinearSpringDamp", "Hertz",
                               "LeeHerrmann", "KuwabaraKono", "Tsuji"],
            "contact_tangential": ["None", "LinearSpringDamp"],
            "rolling": ["None", "Viscous", "Coulomb"],
            "adhesion": ["None", "VdWDMT", "RegDMT"],
        },

        "peridynamics": {
            "dimensions": ["3D (Peridynamic_3D)", "2D Plane Stress (Peridynamic_2DPlaneStress)",
                          "2D Plane Strain (Peridynamic_2DPlaneStrain)"],
            "features": ["Bond-based PD", "Damage via critical stretch criterion",
                        "Volume correction factor", "Pre-crack definition via line segments"],
            "material": "MAT_ParticlePD: INITRADIUS, INITDENSITY, YOUNG, CRITICAL_STRETCH",
            "input_section": "PARTICLE DYNAMIC/PD",
            "key_params": {
                "INTERACTION_HORIZON": "delta = m * dx (typically m=3, so horizon = 3*particle_spacing)",
                "PERIDYNAMIC_GRID_SPACING": "dx (particle spacing, must match actual particle grid)",
                "PD_DIMENSION": "Peridynamic_2DPlaneStrain / Peridynamic_2DPlaneStress / Peridynamic_3D",
                "PRE_CRACKS": "Line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4' — bonds crossing these are pre-broken (pre-merge branch form; not in upstream 4C)",
                "PRE_CRACK_LINES": "4C 2026.2.0 and later: a list of '- START: [x1, y1, z1]' / 'END: [x2, y2, z2]' segments; bonds crossing them are broken at initialization. PRE_CRACK_PLANES is the same with parallelogram patches given by corner points P0, P1, P2 (both lists take 3-D points; both are accepted in 2-D peridynamic decks)",
                "NORMALCONTACTLAW": "NormalLinearSpring (for impactor-body contact)",
                "NORMAL_STIFF": "Contact stiffness (e.g., 1.0e4)",
            },
            "particle_grid_generation": {
                "description": "PD requires a REGULAR GRID of particles with sufficient resolution",
                "pattern": "Loop over nx*ny (2D) or nx*ny*nz (3D) with uniform spacing dx",
                "spacing": "dx should be chosen based on the problem scale; horizon = m*dx (m=3 typical)",
                "notches_cracks": "Skip particles inside notch gaps OR use pre-crack segments (PRE_CRACK_LINES in 4C 2026.2.0 and later; PRE_CRACKS on the pre-merge branch)",
                "example": "for iy in range(ny): for ix in range(nx): particles.append((ix*dx, iy*dx, 0.0))",
                "convergence": "PD converges as dx→0 AND m→∞ (delta-convergence AND m-convergence)",
            },
        },

        "time_integration": ["Semi-implicit Euler (SemiImplicitEuler)", "Velocity Verlet (VelocityVerlet)"],

        "vtk_output": {
            "description": (
                "Particle ParaView output needs no configuration at all "
                "and cannot be configured: a Particle run writes it "
                "unconditionally, at the RESULTSEVERY interval of "
                "PARTICLE DYNAMIC."
            ),
            "yaml_section": """
# Nothing to add. The knob is RESULTSEVERY:
PARTICLE DYNAMIC:
  RESULTSEVERY: 10""",
            "output_format": (
                "One .pvd per particle phase plus .vtu / .pvtu pieces "
                "under <prefix>-vtk-files. Not .vtp."
            ),
            "pitfall": (
                "There is NO IO/RUNTIME VTK OUTPUT/PARTICLES section and "
                "no PARTICLE_OUTPUT key. Adding either turns a working "
                "deck into a hard parse error, \"Section 'IO/RUNTIME VTK "
                "OUTPUT/PARTICLES' is not a valid section name.\" 4C's "
                "runtime-VTK vocabulary is IO/RUNTIME VTK OUTPUT plus "
                "/BEAMS, /FLUID and /STRUCTURE only."
            ),
        },

        "mandatory_sph_section": {
            "description": "Even for PURE peridynamics, the SPH section is MANDATORY in 4C",
            "reason": "The PD implementation lives inside the SPH interaction framework. Without SPH section, pd_neighbor_pairs=0 → no PD forces computed",
            "yaml": """
PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity""",
        },

        "impactor_setup": {
            "description": "Rigid impactor as boundary phase particles",
            "material": "MAT_ParticleSPHBoundary: INITRADIUS, INITDENSITY",
            "phase_mapping": "PHASE_TO_MATERIAL_ID: 'boundaryphase 1 pdphase 2'",
            "velocity": "Applied via FUNCT + DIRICHLET_BOUNDARY_CONDITION on boundaryphase",
        },

        "pitfalls": [
            (
                "[Input] PARTICLE DYNAMIC/SPH section is "
                "MANDATORY even for PURE peridynamics — the "
                "PD implementation lives inside the SPH "
                "interaction framework. Signal: omitting the "
                "SPH sub-section is a hard abort before the "
                "first time step, 'negative initial particle "
                "spacing!' from "
                "particle/src/interaction/"
                "4C_particle_interaction_sph.cpp, because "
                "INITIALPARTICLESPACING lives there and sets "
                "every particle's mass; there is no silent "
                "no-force run. Do NOT read the line 'Number "
                "of pd_neighbor_pairs in peridynamic "
                "evaluation on this proc: 0' as the symptom — "
                "that counter is the body-to-body contact "
                "pair list and is zero for most steps of a "
                "correct run. Add the SPH block with KERNEL: "
                "QuinticSpline, KERNEL_SPACE_DIM: Kernel2D "
                "(or 3D), INITIALPARTICLESPACING matching dx; "
                "those three keys alone are sufficient for a "
                "pure PD body. (Audit 2026-06-02; corrected "
                "by execution 2026-08-06.)"
            ),
            (
                "[Output] Particle ParaView output needs no "
                "section at all and is never .vtp: a Particle "
                "run writes one .pvd per phase plus .vtu / "
                ".pvtu pieces under <prefix>-vtk-files "
                "unconditionally, at the RESULTSEVERY "
                "interval of PARTICLE DYNAMIC. Signal: adding "
                "the section some sources recommend, "
                "IO/RUNTIME VTK OUTPUT/PARTICLES with "
                "PARTICLE_OUTPUT: true, turns a working deck "
                "into a hard parse error — \"Section "
                "'IO/RUNTIME VTK OUTPUT/PARTICLES' is not a "
                "valid section name.\" from "
                "core/io/src/4C_io_input_file.cpp. 4C's "
                "runtime-VTK vocabulary is IO/RUNTIME VTK "
                "OUTPUT plus /BEAMS, /FLUID and /STRUCTURE "
                "only; there is no /PARTICLES section and no "
                "PARTICLE_OUTPUT key. (Audit 2026-06-02; "
                "falsified by execution 2026-08-06.)"
            ),
            (
                "[Input] PD requires a REGULAR particle grid "
                "(uniform spacing in all directions). "
                "Signal: 4C accepts a graded or anisotropic "
                "lattice in total silence — no spacing, "
                "packing or uniformity check exists — and the "
                "only visible trace is the line it already "
                "prints, 'Number of initialized peridynamic "
                "bonds on this proc: N', which grows sharply "
                "for the same particle count because each "
                "particle gains partners along the dense "
                "axis while every bond keeps the single "
                "scalar dx volume correction. Compare that "
                "count against a uniform lattice; do not "
                "expect a wave-speed or fracture-pattern "
                "diagnostic, 4C emits none. Generate "
                "particles on a regular grid (e.g. nx*ny loop "
                "with uniform dx). (Audit 2026-06-02; "
                "corrected by execution 2026-08-06.)"
            ),
            (
                "[Input] INTERACTION_HORIZON must equal m * "
                "dx where m is the horizon ratio (typically "
                "3). Signal: do NOT diagnose a too-small "
                "horizon by counting neighbours — on a "
                "regular lattice even m below 2 leaves each "
                "particle several partners (the 3x3 stencil "
                "minus its centre, then its axial members), "
                "never the one or two sometimes claimed. What "
                "you actually get is an unstable run killed "
                "after a few dozen steps by \"a particle of "
                "phase 'pdphase' traveled more than one bin "
                "on this processor!\" from "
                "particle/src/algorithm/"
                "4C_particle_algorithm.cpp, with no line "
                "mentioning the horizon. Set m deliberately; "
                "m=3 is the minimum for delta-convergence to "
                "classical elasticity. (Audit 2026-06-02; "
                "corrected by execution 2026-08-06.)"
            ),
            (
                "[Input] PERIDYNAMIC_GRID_SPACING in the "
                "input must EXACTLY match the actual "
                "particle spacing in the mesh. Signal: there "
                "is none — 4C never compares the key against "
                "the particle coordinates, the bond list is "
                "unchanged because the neighbour search uses "
                "the horizon and not this key, every time "
                "step runs, and no line matching spacing or "
                "mismatch is printed. Only the answer moves, "
                "and it moves hard: in 2-D the bond force "
                "carries dx to the fourth power, so even a "
                "few per cent of slip is already outside any "
                "result tolerance. Verify dx yourself by "
                "computing the min pairwise distance between "
                "the first 10 particles. (Audit 2026-06-02; "
                "corrected by execution 2026-08-06.)"
            ),
            (
                "[Input] PRE_CRACKS (the pre-merge PD branch's key, "
                "see the end of this entry) uses semicolon-separated "
                "line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4'. "
                "Signal: mis-formatted PRE_CRACKS in PARTICLE "
                "DYNAMIC / PD (comma separator, or no "
                "separator at all) does NOT over-break: the "
                "parser splits on ';' and reads exactly four "
                "numbers per piece, so the list collapses to "
                "the FIRST crack and every later segment is "
                "silently discarded. You get fewer broken "
                "bonds than you asked for, nothing spurious, "
                "and a run that can still report PASS. The "
                "one line that gives it away is 4C's own "
                "'Number of pre-crack segments: N' — check N "
                "against the number of cracks you wrote. "
                "(Audit 2026-06-02; corrected by execution "
                "2026-08-06. PRE_CRACKS is the pre-merge branch "
                "form; upstream 4C 2026.2.0 and later read PRE_CRACK_LINES "
                "/ PRE_CRACK_PLANES lists of START and END "
                "points instead.)"
            ),
            (
                "[Input] PDBODYID must be specified for PD "
                "phase particles (e.g. PDBODYID 0). Signal: "
                "omitting PDBODYID gives every PD particle "
                "the default body id 0 (not -1 — the state is "
                "zero-filled), and bonds form between any two "
                "particles whose ids agree, so two bodies "
                "that both take the default are welded "
                "together across the gap. 4C prints no "
                "warning about body ids; the only trace is "
                "'Number of initialized peridynamic bonds on "
                "this proc: N' coming out higher than the sum "
                "of the two bodies' own bonds. What matters "
                "is that the ids DIFFER, not what they are. "
                "(Audit 2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] Boundary phase particles (impactor) "
                "need TYPE boundaryphase; PD particles need "
                "TYPE pdphase. Signal: swapping the two TYPE "
                "labels bonds the wrong body — the bond count "
                "jumps because what should be rigid becomes "
                "the peridynamic body — while the former PD "
                "body, now a boundary phase, stops moving "
                "entirely (velocity exactly zero, position at "
                "its reference). The only message 4C gives is "
                "\"state 'pd_damage_phi' not found in "
                "container!\" from particle/src/algorithm/"
                "4C_particle_algorithm_result_test.cpp, which "
                "names a missing STATE and never a phase, a "
                "TYPE or PHASE_TO_MATERIAL_ID. Verify TYPE "
                "per phase against the PHASE_TO_MATERIAL_ID "
                "mapping. (Audit 2026-06-02; corrected by "
                "execution 2026-08-06.)"
            ),
            (
                "[Numerical] The rule of thumb dt < 0.5 * dx "
                "/ c_wave with c_wave = sqrt(E/rho) is a "
                "conservative starting point for the explicit "
                "PD step, NOT the stability limit: 4C's own "
                "PD regression decks run at several times "
                "that dt and reproduce their reference values "
                "exactly, so applying it literally shrinks a "
                "working time step for nothing. Signal: when "
                "dt really is too large you get no NaN and no "
                "energy message — the string 'energy not "
                "conserved' does not exist in 4C — the run is "
                "killed after a handful of steps by \"a "
                "particle of phase 'pdphase' traveled more "
                "than one bin on this processor!\" from "
                "particle/src/algorithm/"
                "4C_particle_algorithm.cpp, which never "
                "mentions CFL or the time step. Calibrate dt "
                "by halving until that abort stops. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] BINNING STRATEGY's "
                "BIN_SIZE_LOWER_BOUND must be >= "
                "INTERACTION_HORIZON, and the constraint is "
                "enforced, so no degraded neighbour search is "
                "possible. Signal: a smaller bin aborts in "
                "the SPHPeridynamic constructor with "
                "'Peridynamic INTERACTION_HORIZON must be "
                "smaller than BIN_SIZE_LOWER_BOUND!' from "
                "particle/src/interaction/"
                "4C_particle_interaction_sph_peridynamic.cpp "
                "— before the bond list is built, before the "
                "first step, with no bond count printed. The "
                "guard is <=, so a bin EQUAL to the horizon "
                "is legal; and every admissible bin size "
                "gives the identical bond list and verdict, "
                "so there is nothing to tune here. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] DOMAINBOUNDINGBOX must enclose ALL "
                "particles INCLUDING the impactor motion "
                "range. Signal: nothing aborts. There is no "
                "'particle out of domain' message; 4C simply "
                "DELETES the particles that leave and keeps "
                "integrating, logging only 'on processor 0 "
                "removed N particle(s) being outside the "
                "computational domain!' — no id, no position, "
                "no step, no phase. It fires at setup for "
                "particles already outside and again "
                "mid-run for particles that walk out, the run "
                "completes all its steps, and the damage "
                "surfaces only at the end as failed result "
                "tests. Grep the log for 'outside the "
                "computational domain' on every run, and set "
                "the bbox larger than the initial particle "
                "extent by at least the maximum expected "
                "displacement over the simulation. (Audit "
                "2026-06-02; corrected by execution "
                "2026-08-06.)"
            ),
            (
                "[Input] The two phase-mapping strings in "
                "PARTICLE DYNAMIC do different jobs and fail "
                "in wildly different ways, and the dangerous "
                "one is the one that looks redundant. "
                "PHASE_TO_DYNLOADBALFAC is what DECLARES the "
                "phases; PHASE_TO_MATERIAL_ID gives each "
                "declared phase a material id. Signal: get "
                "the first wrong — omit it, or name a phase "
                "your PARTICLES lines do not use — and you "
                "get a clean, specific abort naming the "
                "offending phase, \"particle type 'phase1' of "
                "initial particle not defined!\" from "
                "particle/src/algorithm/"
                "4C_particle_algorithm.cpp. Get the SECOND "
                "wrong the same way and there is no message "
                "at all: the run dies on a raw 'Signal: "
                "Segmentation fault (11)' with exit status "
                "139, inside MaterialHandler::"
                "initialize_parameters, before the first time "
                "step and before any 4C error block. The "
                "mechanism is that the handler sizes its "
                "table from the largest key in an empty map. "
                "This is engine-level, so it bites SPH, DEM "
                "and PD identically. Always write both "
                "strings, with the same phase names in the "
                "same order. (Verified by execution "
                "2026-08-07.)"
            ),
            (
                "[Syntax] The PARTICLES section is a YAML list "
                "of whitespace-delimited STRINGS, not a list "
                "of mappings, and nothing else is accepted. "
                "The grammar is 'TYPE <phase> POS <x> <y> <z>' "
                "followed by optional tokens — RAD <r>, "
                "RIGIDCOLOR <c>, and on branches carrying "
                "peridynamics PDBODYID / PDFIXED / PDWEIGHT. "
                "There is no MAT, VELOCITY or RADIUS key on a "
                "particle line: the material comes from "
                "PHASE_TO_MATERIAL_ID and the initial velocity "
                "from INITIAL_VELOCITY_FIELD in PARTICLE "
                "DYNAMIC/INITIAL AND BOUNDARY CONDITIONS. "
                "Legal phase names are exactly phase1, phase2, "
                "boundaryphase, rigidphase, dirichletphase, "
                "neumannphase and pdphase — the set is closed. "
                "Signal: a mapping-shaped entry aborts with "
                "\"Yaml node does not contain a string. This "
                "legacy function is only meant for strings.\" "
                "from core/io/src/4C_io_input_file.cpp — a "
                "message that names neither the PARTICLES "
                "section nor particles at all, shows none of "
                "the grammar, and reads like a defect in 4C "
                "rather than a wrong schema in your deck. The "
                "only thing tying it to particles is the stack "
                "frame Particle::read_particles beneath it, so "
                "read the frames. An invented "
                "phase name is worse and gives NO message at "
                "all: declare it consistently in both phase "
                "maps and use it on a particle line, and 4C "
                "dies on 'Signal: Segmentation fault (11)', "
                "exit status 139, with zero error blocks — the "
                "name lookup is guarded by an assertion that a "
                "release build compiles out. Do not expect an "
                "'unknown particle type' abort; check the "
                "spelling against the closed list above "
                "yourself. Note that global particle ids are "
                "assigned in "
                "FILE ORDER starting at 0, and that is the id "
                "a RESULT DESCRIPTION PARTICLE entry means, so "
                "inserting a particle renumbers every test "
                "after it. (Verified by execution 2026-08-07.)"
            ),
            (
                "[Input] BIN_SIZE_LOWER_BOUND is a LOWER "
                "BOUND, not the bin size, and the safety check "
                "compares the method's interaction distance "
                "against the bin size the engine actually "
                "chose. The distance is whatever the active "
                "method computes — twice the largest particle "
                "radius for DEM, plus ADHESION_DISTANCE when "
                "an adhesion law is on; the largest INITRADIUS "
                "for SPH; INTERACTION_HORIZON for PD — so the "
                "number you must clear moves when you change "
                "the physics, and it is never something you "
                "wrote. Signal: when it does fire the abort "
                "prints BOTH numbers, 'the particle "
                "interaction distance is larger than the "
                "minimal bin size (<distance> > <binsize>)!' "
                "from particle/src/interaction/"
                "4C_particle_interaction_base.cpp, at setup "
                "and before any bond or neighbour count — one "
                "of the few 4C particle diagnostics you can "
                "act on directly. But do NOT read a passing "
                "run as evidence that your bound is adequate: "
                "because the engine sizes bins to divide the "
                "domain and only takes the key as a floor, "
                "lowering BIN_SIZE_LOWER_BOUND far below the "
                "support radius can leave the chosen bin, and "
                "the answer, bit-identical. Whether the check "
                "fires depends on the domain box as much as on "
                "the key. Beware also that a radius "
                "DISTRIBUTION sets the largest radius at run "
                "time, so widening MAX_RADIUS can trip the "
                "check on a deck whose bin size was previously "
                "fine and whose BINNING STRATEGY you did not "
                "touch. (Verified by execution 2026-08-07.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # POROUS MEDIA
    # ═══════════════════════════════════════════════════════════════════════
    "porous_media": {
        "description": "Biot poroelasticity and porous flow",
        "problem_types": {
            "Poroelasticity": "Biot consolidation (structure + fluid in pores)",
            "Poroelastic_scalar_transport": "Poro + scalar transport",
            "porofluid_pressure_based": "Pressure-based porous flow (standalone)",
        },
        "coupling": ["Monolithic", "Partitioned", "1-way", "2-way"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR / BIOMEDICAL
    # ═══════════════════════════════════════════════════════════════════════
    "cardiovascular": {
        "description": "Cardiovascular and biomedical simulation capabilities",
        "models": {
            "0D_windkessel": "4-element Windkessel for arterial pressure",
            "arterial_network": "1D arterial blood flow network (artery elements)",
            "reduced_airways": "Reduced lung airways with acinus elements",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher, etc.)",
        },
        "applications": "Arterial hemodynamics, cardiac mechanics, lung ventilation",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # BEAM INTERACTION
    # ═══════════════════════════════════════════════════════════════════════
    "beam_interaction": {
        "description": "Beam-to-beam, beam-to-solid, beam-to-sphere contact and meshtying",
        "contact_pairs": [
            "Beam-to-beam (point coupling, tangent smoothing)",
            "Beam-to-solid volume meshtying (Gauss point, mortar)",
            "Beam-to-solid surface meshtying",
            "Beam-to-solid surface contact",
            "Beam-to-sphere contact",
        ],
        "cross_linking": "Pin-jointed, rigid-jointed, truss links (biopolymer networks)",
        "brownian_dynamics": "Stochastic dynamics of beam networks",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LINEAR SOLVERS
    # ═══════════════════════════════════════════════════════════════════════
    "solvers": {
        "direct": {
            "UMFPACK": "Serial direct solver (recommended for small problems)",
            "Superlu": "Parallel direct solver (SuperLU_Dist); the SOLVER value is spelled exactly 'Superlu' ('SuperLU' stops with 'Could not match this input')",
            "MUMPS": "Parallel direct solver (MPI, recommended for large problems); not in 4C 2026.1.0",
            "KLU2": "Serial direct solver (alternative to UMFPACK); not in 4C 2026.1.0",
        },
        "iterative": {
            "CG": "Conjugate gradient (symmetric positive definite systems only)",
            "GMRES": "Generalized minimal residual (non-symmetric systems)",
            "BiCGSTAB": "Bi-conjugate gradient stabilized (non-symmetric, lower memory)",
        },
        "preconditioners": {
            "ILU": "Incomplete LU factorization (Ifpack package)",
            "MueLu": "Algebraic multigrid (MueLu, recommended for large problems)",
            "Block_Teko": "Block preconditioning for multi-field problems (Teko package)",
        },
        "nonlinear": {
            "NOX": "Trilinos NOX framework (Newton + line search + PTC + convergence tests)",
        },
        # The preconditioner file key differs between the releases: 4C 2026.2.0
        # and later accept only PRECONDITIONER_XML_FILE, earlier builds only the
        # key named after the preconditioner (MUELU_XML_FILE, TEKO_XML_FILE,
        # IFPACK_XML_FILE, AMGNXN_XML_FILE).
        "yaml_example_4c_2026_2_and_later": """
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
SOLVER 2:
  SOLVER: "Belos"
  SOLVER_XML_FILE: "iterative_gmres_template.xml"
  AZPREC: "MueLu"
  PRECONDITIONER_XML_FILE: "elasticity_template.xml"
  NAME: "iterative_solver"
""",
        "yaml_example_before_4c_2026_2": """
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
SOLVER 2:
  SOLVER: "Belos"
  SOLVER_XML_FILE: "iterative_gmres_template.xml"
  AZPREC: "MueLu"
  MUELU_XML_FILE: "elasticity_template.xml"
  NAME: "iterative_solver"
""",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT FILE FORMAT
    # ═══════════════════════════════════════════════════════════════════════
    "input_format": {
        "description": "YAML-based input files (.4C.yaml) — can use inline mesh or Exodus file",

        "mandatory_sections": [
            "PROBLEM TYPE (PROBLEMTYPE: Structure/Scalar_Transport/Fluid/...)",
            "Dynamics section matching problem type (STRUCTURAL DYNAMIC, etc.)",
            "At least one SOLVER",
            "MATERIALS",
            "Mesh (NODE COORDS + ELEMENTS, or STRUCTURE GEOMETRY with FILE)",
        ],

        # 2026-08-03 correction, verified by execution: PROBLEM SIZE
        # used to be listed above as mandatory. It is declared
        # {.required = false} in
        # src/global_legacy_module/4C_global_legacy_module_validparameters.cpp
        # and a deck with no PROBLEM SIZE section runs to completion.
        # Its only load-bearing field is DIM (default 3); ELEMENTS /
        # NODES / MATERIALS / NPATCHES / NUMDF are read and ignored.
        "optional_sections": {
            "PROBLEM SIZE": (
                "Optional. DIM defaults to 3. ELEMENTS / NODES / "
                "MATERIALS / NPATCHES / NUMDF are parsed into a "
                "parameter list and never consumed — the 4C source "
                "comments them as unused. Verified by execution "
                "2026-08-03 (omitted entirely, and with deliberately "
                "wrong counts: identical results, exit 0)."),
            "DISCRETISATION": (
                "Optional. NUMFLUIDDIS / NUMALEDIS / NUMTHERMDIS all "
                "default to 1 in "
                "global_legacy_module_validparameters. Present in "
                "558 of the 1974 tests/input_files decks, absent "
                "from the rest."),
            "TITLE": "Optional free-text description.",
            "IO": "Optional; every key has a default.",
            "RESULT DESCRIPTION": (
                "Optional, but it is the only built-in numerical "
                "self-check — see the result_description block."),
        },

        "boundary_conditions": {
            "structural": {
                "DESIGN POINT/LINE/SURF/VOL DIRICH CONDITIONS": "Prescribed displacement",
                "DESIGN POINT/LINE/SURF/VOL NEUMANN CONDITIONS": "Applied force/traction/body force",
            },
            "thermal": {
                "DESIGN SURF/VOL THERMO DIRICH CONDITIONS": "Prescribed temperature",
                "DESIGN SURF/VOL THERMO NEUMANN CONDITIONS": "Applied heat flux",
            },
            "bc_format": """
DESIGN SURF DIRICH CONDITIONS:
  - E: 1            # Design entity ID
    NUMDOF: 3       # Number of DOFs per node
    ONOFF: [1, 1, 0] # Which DOFs are constrained (1=yes, 0=no)
    VAL: [0.0, 0.0, 0.0]  # Prescribed values
    FUNCT: [0, 0, 0]       # Time function IDs (0=constant)
""",
        },

        "topology_sections": {
            "DNODE-NODE TOPOLOGY": "Map single nodes to design nodes (for point BCs)",
            "DLINE-NODE TOPOLOGY": "Map nodes to design lines (for line BCs in 2D)",
            "DSURF-NODE TOPOLOGY": "Map nodes to design surfaces (for surface BCs in 3D)",
            "DVOL-NODE TOPOLOGY": "Map nodes to design volumes (for volume BCs)",
        },

        "functions": """
# --- Simple space-time function (no time-varying sub-variables) ---
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "sin(2*pi*x)*cos(pi*t)"
  # Supports: x, y, z, t as variables

# --- Function with VARIABLE (e.g. ramp-up) ---
# COMPONENT is optional (it defaults to 0): a VARIABLE/multifunction
# function gives the same field with or without the COMPONENT: 0 line.
# Only a COMPONENT index other than the one being defined is an error
# ('expected COMPONENT 0 but got COMPONENT 1').
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*U_bar*y*(H-y)/(H*H)*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 2, 10000]
    DESCRIPTION: ["0.5*(1-cos(pi*t/2))", "1.0"]

# --- Pure time function (no COMPONENT needed) ---
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_TIME: "a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 1, 10000]
    DESCRIPTION: ["0.5*(1.0-cos((t*pi)/1.0))", "1.0"]

# --- Linear interpolation (piecewise linear in time) ---
FUNCT4:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "linearinterpolation"
    NUMPOINTS: 3
    TIMES: [0, 1, 101]
    VALUES: [0, 1, 100]
""",

        "inline_mesh_example": """
NODE COORDS:
  - "NODE 1 COORD 0.000000 0.000000 0.0"
  - "NODE 2 COORD 1.000000 0.000000 0.0"
TRANSPORT ELEMENTS:
  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"
""",

        # 2026-06-01 (critic-audit #5): renamed from
        # 'general_pitfalls' so the verify_signal_clauses /
        # orphan / parse-discipline harnesses (which key off
        # the literal 'pitfalls' field) can see these entries.
        # Combined with the fourc backend exposing
        # 'input_format' as a [Reference] PhysicsCapability,
        # users now reach them via discover + knowledge +
        # prepare_simulation.
        "pitfalls": [
            # ExodusII block IDs
            "[API] CRITICAL: meshio (Python) writes ExodusII "
            "element block IDs starting at 0 (0-indexed), while "
            "decks are usually written with 1-indexed "
            "ELEMENT_BLOCKS IDs. 4C matches ELEMENT_BLOCKS "
            "against whatever id the file carries and does NOT "
            "diagnose a mismatch. Signal: the only clue is 4C's "
            "own mesh summary, which echoes the numbering it "
            "read — 'cell-block 0 (): 1 cells of type hex8' "
            "against a deck that asked for ID: 1. There is no "
            "warning, no PROC 0 ERROR block and no mention of "
            "ELEMENT_BLOCKS anywhere; the run builds an empty "
            "discretisation and dies much later inside the "
            "linear solver with \"terminate called after "
            "throwing an instance of 'std::runtime_error'\" and "
            "'umfpack_solve has error code: -3', shell status "
            "134. Fix: after writing with "
            "meshio, patch with netCDF4 — "
            "import netCDF4; ds = netCDF4.Dataset("
            "'mesh.e', 'r+'); ds.variables['eb_prop1'][:] += "
            "1; ds.close(). Verify with: python3 -c \"import "
            "netCDF4; print(netCDF4.Dataset('mesh.e')"
            ".variables['eb_prop1'][:])\", or just read the "
            "cell-block line 4C prints. (Verified by execution "
            "2026-08-06; the previously recorded 'Pressure map "
            "empty' signal does not occur.)",

            # FUNCT COMPONENT requirement
            "[Syntax] COMPONENT is OPTIONAL in a FUNCT list "
            "item and defaults to 0; a "
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME driven by a "
            "VARIABLE works identically with and without it, "
            "and the VARIABLE is NOT ignored when it is "
            "omitted. What is fatal is naming a component "
            "index other than the one being defined. Signal: "
            "'expected COMPONENT 0 but got COMPONENT 1' from "
            "core/utils/src/functions/4C_utils_function.cpp, "
            "exit 1. SYMBOLIC_FUNCTION_OF_TIME (pure time "
            "functions) likewise do not need COMPONENT. "
            "(Verified by execution 2026-08-06: a Dirichlet "
            "driven by a linearinterpolation VARIABLE ramping "
            "to 0.5 gave the same displacement with "
            "'COMPONENT: 0' and without it, and 4C never "
            "mentioned COMPONENT in the run that omitted it; "
            "the earlier claim that the VARIABLE is silently "
            "dropped and the BC stays stuck at 0 is false.)",

            # Shared-node NUMDOF conflict
            "[API] In multi-physics problems (FSI, TSI, SSI), "
            "DESIGN ... DIRICH CONDITIONS are attached to "
            "NODES, so EVERY discretisation containing a node "
            "reads the block — including the one it was not "
            "written for. Signal: a node shared between "
            "structure (3 DOFs) and fluid (4 DOFs in 3D) makes "
            "the structure's NUMDOF-3 block fail on the fluid "
            "discretisation with '3 DOFs given but 4 expected "
            "in Volume Dirichlet boundary condition' from "
            "core/fem/src/discretization/"
            "4C_fem_discretization_utils_dbc.cpp, exit 1 — a "
            "RUNTIME DOF check, not a parse check. Raising "
            "NUMDOF to 4 does not help; the complaint simply "
            "moves to the next condition on the same nodes. "
            "Solutions: (a) use separate node sets per "
            "discretisation, (b) give each field its own nodes "
            "at the interface, (c) use mortar coupling with "
            "non-matching meshes. (Verified by execution "
            "2026-08-06 on a monolithic FSI deck whose "
            "structure element was rewired onto the fluid's "
            "interface nodes with no condition edited; the "
            "previously recorded 'inconsistent NUMDOF on "
            "shared node' text and the attribution to "
            "4C_io_input_spec.cpp are both wrong.)",

            # Invalid section names
            "[Syntax] 4C is STRICT about section names. Common "
            "invalid sections: EVERY_ITERATION (not a valid IO "
            "parameter), DESIGN FLUID LINE LIFT&DRAG (does not "
            "exist for 2D), DESIGN THERMO LINE DIRICH "
            "CONDITIONS (wrong — must be DESIGN LINE THERMO "
            "DIRICH CONDITIONS). Signal: 4C aborts with "
            "\"Section '<name>' is not a valid section name.\" from "
            "core/io/src/4C_io_input_file.cpp for a bad SECTION name, or "
            "'Could not match this input' "
            "from 4C_io_input_spec_builders.cpp at parse time, "
            "echoing the offending YAML block. The two are "
            "different diagnoses: the first means no such "
            "section, the second means right section, wrong "
            "contents. Check valid names against the binary's "
            "own schema, where sections appear as "
            "'    - name: <NAME>' under the top-level "
            "'sections:' key: "
            "4C --parameters 2>/dev/null | grep 'name: DESIGN'. "
            "(Verified by execution 2026-08-06: all three "
            "quoted bad names abort with the section-name "
            "message, and the schema confirms DESIGN LINE "
            "THERMO DIRICH CONDITIONS exists while the "
            "transposed spelling and DESIGN FLUID LINE "
            "LIFT&DRAG do not — LIFT&DRAG is SURF only.)",

            # Output
            "[API] 4C writes native .control/.mesh/.result "
            "files. To get VTU output for ParaView, either: "
            "(a) add IO/RUNTIME VTK OUTPUT sections "
            "(recommended), or (b) run post_vtu --file="
            "output_prefix AFTER the simulation. Signal: "
            "after PROBLEMTYPE / STRUCTURAL DYNAMIC / FLUID "
            "DYNAMIC / SCATRA DYNAMIC run completes, looking "
            "for a .vtu / .pvd output file in the results "
            "directory finds nothing — 4C only produced "
            ".control / .mesh / .result; either the IO/"
            "RUNTIME VTK OUTPUT section is missing or "
            "post_vtu was not invoked. The native files are "
            "HDF5-readable but not directly ParaView-"
            "loadable. post_vtu prints a deprecation notice "
            "and WAITS FOR ENTER unless you pass "
            "--postprocessor_deprecation_warning_off, so in a "
            "script give it --postprocessor_deprecation_"
            "warning_off or close its stdin. (Verified by "
            "execution 2026-08-06: a 2-step HEX8 run left "
            "res.control plus HDF5 res.mesh.structure.s0 and "
            "res.result.structure.s0 and zero .vtu; adding "
            "IO/RUNTIME VTK OUTPUT + IO/RUNTIME VTK OUTPUT/"
            "STRUCTURE wrote 3 .vtu and 1 .pvd, and post_vtu "
            "--file=res on the native output produced the "
            "same 3 .vtu / 1 .pvd.)",

            # 2026-06-01: .dat extension rejected
            "[Syntax] 4C 2026.3.0-dev accepts ONLY .yaml / .yml / .json input "
            "files. Passing a legacy .dat-format file (or any other extension) "
            "is rejected at file-open time with 'Cannot infer format of input "
            "file ... Only .yaml, .yml, and .json are supported.' from "
            "core/io/src/4C_io_input_file.cpp. Note: the section-name "
            "vocabulary (PROBLEM TYPE, STRUCTURAL DYNAMIC, DESIGN SURF NEUMANN, "
            "MAT_scatra, etc.) is unchanged — those are still valid as YAML "
            "keys; only the overall file format moved from dat-style "
            "section-header text to YAML mapping syntax. Signal: 4C ERROR "
            "from 4C_io_input_file.cpp with 'Cannot infer format' and "
            "'Only .yaml, .yml, and .json' substrings when an unsupported "
            "extension is passed on the CLI. (Verified empirically 2026-06-01.)",

            "[Syntax] 4C validates enum-like keys against an allowed set at "
            "input-parse time. Mis-spelling a PROBLEMTYPE value (e.g. "
            "'Hyperelasticity' instead of 'Structure', or a typo like "
            "'Scalar_Tranzport') triggers 'PROC 0 ERROR ... Could not match "
            "this input' from core/io/src/4C_io_input_spec_builders.cpp, "
            "with the offending YAML block echoed in the message. Signal: "
            "the substrings 'Could not match this input', 'PROBLEMTYPE', and "
            "'input_spec_builders' all appear in 4C stderr when the value is "
            "not in the allowed enum set. (Verified empirically 2026-06-01.)",

            # THICKNESS parameter for 2D plane strain
            "[Input] For 2D plane-strain elements the depth "
            "keyword is the OUT-OF-PLANE thickness, not the "
            "element width, and it is almost always 1.0. It "
            "multiplies the element stiffness while a Live "
            "line load is not scaled with it, so putting the "
            "element edge length there divides the computed "
            "displacement by exactly that factor. Signal: "
            "there is none — every value runs to exit 0 and "
            "nothing in the log mentions thickness; the only "
            "way to catch it is that the answer scales as "
            "1/thickness. CHECK THE KEYWORD AGAINST YOUR "
            "BUILD: on 4C before 2026.2.0 (measured on a 2026.2.0-dev build) the key is THICK and it "
            "belongs to WALL; THICKNESS does not exist as an "
            "element key at all and neither does "
            "PLANE_ASSUMPTION, so a deck written with the "
            "newer spelling does not merely behave differently "
            "— it does not parse. Read the legacy_element_specs "
            "block of `4C --parameters` before writing either. "
            "(Verified by execution 2026-08-06 on a WALL QUAD4 "
            "cantilever: dispy at THICK 0.5 / 1.0 / 2.0 / 4.0 "
            "scaled exactly as 1/THICK, with dispy*THICK "
            "constant to 1e-12 relative.)",

            # 2D VTK output artifacts — applies to fluid AND porofluid
            "[Output] In 2D, the runtime VTK output of the "
            "fluid field writes NaN for pressure at every "
            "point while the simulation itself is correct. "
            "The mechanism: a 2D node carries three DOFs "
            "(vx, vy, p), the writer copies all three into "
            "the 3-vector named 'velocity' and then reads a "
            "fourth DOF for 'pressure' that does not exist. "
            "So the pressure IS in the file — as the THIRD "
            "COMPONENT of 'velocity' — and the array named "
            "'pressure' is entirely NaN. Signal: opening "
            "the IO/RUNTIME VTK OUTPUT/FLUID .vtu in "
            "ParaView for a 2D run shows pressure = NaN "
            "everywhere (white/uncolored field) while the run "
            "exited 0 and its own RESULT DESCRIPTION tests "
            "all passed — an output artifact, NOT divergence. "
            "Native HDF5 .result files contain the correct "
            "pressure. 3D output is unaffected. (Verified by "
            "execution 2026-08-06 on upstream "
            "f2_stokes_residualbased.4C.yaml: all 64 pressure "
            "entries NaN, none in the velocity array, and the "
            "third velocity component matched the deck's own "
            "pinned pressures at nodes 1, 19 and 25; the same "
            "sections on a 3D Stokes deck gave 1000 pressures "
            "with zero NaN.)",

            # Poro-specific
            "[Numerical] 4C poro can be fully quasi-static, "
            "and it INSISTS on the pieces being consistent. "
            "A quasi-static run is STRUCTURAL DYNAMIC "
            "DYNAMICTYPE Statics + FLUID DYNAMIC TIMEINTEGR "
            "Stationary + POROELASTICITY DYNAMIC "
            "TRANSIENT_TERMS none, and in that configuration "
            "the solid density is completely inert — there is "
            "no rho*a term, so there is no elastic-wave "
            "ringing to ramp away. Signal: asking for "
            "transient terms next to a stationary fluid is "
            "refused with the fix named — \"Invalid option for "
            "stationary fluid! Set 'TRANSIENT_TERMS' in "
            "section POROELASTICITY DYNAMIC to 'none'!\" from "
            "poroelast/4C_poroelast_base.cpp — and the "
            "structure and porofluid integrators must be "
            "PAIRED (Statics+Stationary, OneStepTheta+"
            "one_step_theta, GenAlpha+afgenalpha), a mismatch "
            "giving 'porous media problem is limited in "
            "functionality (only one-step-theta scheme, "
            "stationary and (af)genalpha case possible)'. "
            "TRANSIENT_TERMS defaults to 'all', so a "
            "quasi-static deck has to set it explicitly. "
            "(Verified by execution 2026-08-06: multiplying "
            "the solid DENS by 1000 on upstream "
            "poro_2D_quad4_const_material_nodal_orthotropic_"
            "permeability.4C.yaml left all three result tests "
            "CORRECT with an unchanged abs(diff); the earlier "
            "claim that poro keeps inertia even in "
            "quasi-static problems is false.)",


            # 2D structural element type: VERSION-DEPENDENT.
            "[API] WHICH element type owns 2D structural cells is "
            "VERSION-DEPENDENT, and the two spellings share no "
            "keywords, so you must pick one:\n"
            "  WALL  QUAD4 <n..> MAT m KINEM k EAS e THICK t "
            "STRESS_STRAIN s GP a b\n"
            "  SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t "
            "PLANE_ASSUMPTION p\n"
            "Read it off `4C --parameters`, which lists the cell types "
            "each element type owns, before writing either. If SOLID's "
            "cell list is 3D-only (HEX/TET/WEDGE/PYRAMID), 2D belongs to "
            "WALL. 3D is always SOLID and never takes THICKNESS or "
            "PLANE_ASSUMPTION. Signal: an element TYPE this build does "
            "not register gives \"Unknown type 'WALL' of finite "
            "element\" from core/comm/src/4C_comm_parobjectfactory.cpp — "
            "NOTE THE QUOTES, the binary's template is \"Unknown type "
            "'{}' of finite element\", so an unquoted grep for it finds "
            "nothing. A registered element type asked for a cell type it "
            "does NOT own gives \"Element 'SOLID' does not seem to know "
            "cell type 'quad4'.\" instead, with the cell type echoed in "
            "lowercase. Measured: a 2026.2.0 development build registers WALL, "
            "the 2026.2.0 and 2026.3.0 releases the 2D SOLID and no WALL. "
            "(Verified by execution 2026-08-03; the earlier "
            "one-sided 'WALL was renamed to SOLID' claim is inverted on "
            "builds where SOLID registers 3D cell types only.)",

            # FSI mesh requirements
            "[API] For monolithic FSI: the structure and "
            "fluid meshes MUST have SEPARATE nodes at the "
            "FSI interface (NOT shared conforming nodes). "
            "The coupling operator pairs the two sides BY "
            "COORDINATE, so upstream FSI decks deliberately "
            "carry two node ids on each interface coordinate. "
            "Signal: a degenerate interface is never tolerated "
            "quietly — one side declared gives 'got 0 master "
            "nodes but 4 slave nodes for coupling' from "
            "coupling/src/adapter/4C_coupling_adapter.cpp, "
            "neither side gives 'No nodes in matching FSI "
            "interface. Empty FSI coupling condition?' from "
            "fsi/src/monolithic/4C_fsi_monolithic.cpp, both "
            "exit 1. Merging the interface nodes does not even "
            "get that far: the two fields' Dirichlet blocks "
            "then land on each other's nodes and the DOF check "
            "fires first (see the shared-node entry). "
            "Post-process Gmsh to duplicate interface nodes, "
            "remap connectivity. Alternative: mortar coupling "
            "(iter_mortar_monolithicfluidsplit) handles "
            "non-matching meshes natively. (Verified by "
            "execution 2026-08-06 on upstream "
            "fsi_fp_mono_fs_ga_ga.4C.yaml; the previously "
            "recorded 'no FSI interface nodes found' text does "
            "not exist, and the simulation never runs "
            "uncoupled.)",

            # Large inline YAML performance
            "[Performance] For meshes beyond a few hundred "
            "nodes, prefer an ExodusII mesh file (.e) over "
            "inline NODE COORDS + ELEMENTS sections: the two "
            "routes are numerically interchangeable and the "
            "deck shrinks by more than a factor of ten, which "
            "is the real reason to switch. Use meshio to write "
            "the mesh to .e format (remembering that meshio "
            "numbers element blocks from 0), then reference it "
            "with STRUCTURE GEOMETRY: FILE: mesh.e and "
            "ELEMENT_BLOCKS. Signal: none at parse time — "
            "large inline decks are read quickly, so a slow or "
            "hanging run is NOT evidence that the mesh should "
            "have been a file. (Verified by execution "
            "2026-08-06: the same 12x12x12 cantilever is 4306 "
            "lines inline against 388 lines plus a .e file, "
            "both exit 0 on the same pinned displacement at "
            "TOLERANCE 1e-12, and 4C's INPUT phase for the "
            "inline deck was a fraction of a second — the "
            "earlier '30+ seconds to parse' figure did not "
            "reproduce.)",

            # FSI + runtime VTK
            "[Output] IO/RUNTIME VTK OUTPUT/STRUCTURE is "
            "incompatible with MONOLITHIC FSI (COUPALGO "
            "iter_monolithic*), because "
            "FSI::MonolithicBase::create_structure_time_"
            "integrator builds the deprecated "
            "Adapter::StructureBaseAlgorithm unconditionally. "
            "PARTITIONED FSI (COUPALGO iter_stagg_*) is NOT "
            "affected and writes .vtu normally. Signal: "
            "'Runtime output is not available in the old "
            "structure time integration! You need to take the "
            "new one, i.e. set `INT_STRATEGY: Standard`!' from "
            "src/structure/4C_structure_timint.cpp, exit 1 — "
            "and IGNORE THAT ADVICE: INT_STRATEGY already "
            "defaults to Standard and setting it explicitly "
            "changes nothing. Remove the IO/RUNTIME VTK "
            "OUTPUT/STRUCTURE section and use post_vtu after "
            "the simulation instead. (Verified by execution "
            "2026-08-06 on upstream fsi_fp_mono_fs_ga_ga.4C.yaml "
            "and volmortar2D_fsi.4C.yaml; the previously "
            "recorded 'inconsistent integration strategy' text "
            "does not exist.)",

            # GPU / hardware acceleration
            "[Hardware] 4C linear algebra is CPU-ONLY "
            "(Epetra-based, Trilinos 16). Epetra does NOT "
            "support GPU execution. Signal: do not time it — "
            "look at the binary. `ldd lib4C.so` lists the "
            "Epetra libraries and NO CUDA / HIP / SYCL / ROCm "
            "runtime at all, and 4C's own timer table names "
            "'Epetra_CrsMatrix::Multiply(TransA,X,Y)' with no "
            "Tpetra entry beside it. Setting "
            "CUDA_VISIBLE_DEVICES, KOKKOS_NUM_DEVICES, "
            "HIP_VISIBLE_DEVICES or KOKKOS_DEVICES leaves the "
            "computed answer bit-identical and produces no "
            "device message whatsoever. Tpetra "
            "(GPU-capable via Kokkos CUDA/HIP/SYCL backends) "
            "is not yet integrated. Plan compute on CPU only. "
            "(Verified by execution 2026-08-06.)",

            # ArborX optional GPU component
            "[Hardware] The ONLY GPU-accelerated component in "
            "4C is ArborX (optional, OFF by default), used "
            "for geometric search (bounding-volume-hierarchy "
            "queries in contact / particle problems). Enable "
            "with cmake flag -DFOUR_C_WITH_ARBORX=ON and a "
            "Kokkos GPU backend in Trilinos; check your build "
            "with grep FOUR_C_WITH_ARBORX in the CMakeCache. "
            "What ArborX gates is the SEARCH, never the linear "
            "solver, so an ArborX-off build loses a search path "
            "and nothing else. Signal: asking for an "
            "ArborX-only code path on a build without it is a "
            "hard abort, not a slow fallback — MESH "
            "PARTITIONING METHOD: monolithic is the cheapest "
            "input-level route in and gives \"The struct "
            "'Core::GeometricSearch::BoundingVolume' can only "
            "be used with ArborX.To use it, enable ArborX "
            "during the configure process.\" from "
            "core/geometric_search/src/"
            "4C_geometric_search_bounding_volume.hpp (note the "
            "missing space where two literals are "
            "concatenated), exit 1, before any linear solve "
            "happens; METHOD hypergraph and multijagged run "
            "the same deck to the same answer. (Verified by "
            "execution 2026-08-06 on a build with "
            "FOUR_C_WITH_ARBORX:BOOL=OFF.)",

            # MPI parallelism
            "[Hardware] 4C uses MPI for domain decomposition. "
            "Standard invocation: mpirun -np N 4C input.4C."
            "yaml. Signal: you do not have to infer the rank "
            "count from a stopwatch — 4C PRINTS it in its "
            "startup banner, 'Total number of MPI ranks: N', "
            "and again as 'Number of procs used for "
            "redistribution: N'. Forgetting mpirun shows up "
            "there as 1 on a many-core box. The decomposition "
            "is a placement decision, not a physics one: the "
            "same deck on 1 and on 2 ranks gives the identical "
            "answer. MPI is the "
            "primary parallelism mechanism; thread-level "
            "parallelism uses OpenMP (set OMP_NUM_THREADS), "
            "which is orthogonal — it never changes the rank "
            "count or the answer, so threads are no substitute "
            "for ranks. "
            "Mixing both (mpirun + OMP_NUM_THREADS) is "
            "supported but oversubscribes if "
            "N_mpi * N_omp > N_cores. (Verified by execution "
            "2026-08-06 on a 64-element HEX8 cantilever.)",

            # ───────────────────────────────────────────────────
            # 2026-08-03 EXECUTION SWEEP.
            # Every entry below was produced by writing a minimal
            # input, running the deployed binary
            # the deployed 4C binary (2026.2.0-dev, git
            # 89519cf) and recording what actually happened. The
            # provenance note on each entry names the probe.
            # ───────────────────────────────────────────────────
            "[Input] RESULT DESCRIPTION is 4C's own numerical "
            "self-check and is the cheapest way to make a run "
            "assert its own correctness — 1925 of the 1974 files "
            "in tests/input_files carry one. A failing entry "
            "aborts the run with a non-zero exit code, so an "
            "agent can gate on the process status instead of "
            "parsing output. Required keys per entry: the field "
            "group name, DIS, one of NODE / LINE / SURFACE / "
            "VOLUME (or SPECIAL), QUANTITY, VALUE, TOLERANCE. "
            "See the result_description block for the full "
            "group and QUANTITY vocabulary. Signal: a passing "
            "check prints 'is CORRECT, abs(diff)= ... < ...' and "
            "exits 0; a failing one prints 'is WRONG --> "
            "actresult= ..., givenresult= ..., abs(diff)= ... > "
            "...' followed by 'Result check failed with N errors "
            "out of M tests' from utils_result_test and exits 1. "
            "(Verified by execution 2026-08-03 on 4C 2026.2.0-dev "
            "git 89519cf, single-HEX8 unit-cube cantilever, "
            "MAT_Struct_StVenantKirchhoff YOUNG 1000 NUE 0.3 "
            "DENS 1, KINEM nonlinear, surface Neumann VAL 1 in y, "
            "dispy at node 3 — the same deck the Tier-2 fixture "
            "result_description_gates_the_exit_code runs: VALUE set "
            "to the true 4.47909266337460053e-03 with TOLERANCE "
            "1e-12 → exit 0 'is CORRECT'; VALUE set to 999.0 → "
            "exit 1 'is WRONG' plus 'Result check failed with 1 "
            "errors out of 1 tests'. An earlier draft of this entry "
            "quoted 5.50797296442576741e-03 here; that value could "
            "not be reproduced on this deck at any of nodes 2/3/6/7 "
            "for dispx/dispy/dispz and disagreed with the fixture, "
            "so it has been replaced by the fixture's own executed "
            "value.)",

            "[Output] A failing RESULT DESCRIPTION check writes "
            "its diagnostic to raw std::cout while the run is "
            "torn down by MPI_Abort, so the 'is WRONG' line and "
            "the 'Result check failed' line are LOST from a "
            "block-buffered pipe — all a caller sees is "
            "'Checking results of N tests:' and then nothing. "
            "A line-buffered stdout via stdbuf -oL -eL is the ONLY "
            "capture that works: redirecting to a FILE does NOT "
            "help, because stdout to a regular file is fully "
            "buffered too and the buffer dies with the process. "
            "Always run "
            "`stdbuf -oL -eL 4C in.4C.yaml out`, whether you then "
            "pipe or redirect, otherwise a failed verification "
            "looks like an unexplained crash. Signal: stdout that "
            "ends exactly at 'Checking results of N tests:' with "
            "exit code 1 and no further text is a swallowed "
            "utils_result_test failure, not a solver crash (the "
            "MPI_ABORT banner still arrives, on stderr). "
            "(Verified by execution 2026-08-03 on 4C 2026.2.0-dev "
            "git 89519cf: same failing deck run four ways — plain "
            "pipe `4C in out 2>/dev/null | tail` ended at "
            "'Checking results of 1 tests:'; plain redirect "
            "`4C in out > f.log 2>&1` gave grep -c 'is WRONG' = 0 "
            "on three consecutive repeats; `stdbuf -oL -eL 4C in "
            "out > f.log 2>&1` gave 'is WRONG' = 1 and 'Result "
            "check failed' = 1.)",

            "[Input] A loose TOLERANCE in RESULT DESCRIPTION "
            "turns the self-check into a rubber stamp — 4C only "
            "compares abs(actresult - VALUE) > TOLERANCE, so a "
            "wide tolerance reports 'is CORRECT' for an answer "
            "that is wrong by orders of magnitude. Size "
            "TOLERANCE against the quantity, not against the "
            "solver tolerance. TOLERANCE <= 0 is rejected "
            "outright, and omitting TOLERANCE is a parse error, "
            "so the only failure mode left is a tolerance that "
            "is merely too big. Signal: 'is CORRECT, abs(diff)= "
            "<large> < <even larger>' where abs(diff) is the "
            "same order as the quantity itself means the check "
            "is not constraining anything; TOLERANCE: 0 raises "
            "'Tolerance for result test must be strictly "
            "positive!' from utils_result_test. (Verified by "
            "execution 2026-08-03: VALUE 0.0 with TOLERANCE 1.0 "
            "on a true answer of 5.5e-03 → exit 0 'is CORRECT'; "
            "TOLERANCE: 0.0 → exit 1 with the strictly-positive "
            "throw; TOLERANCE omitted → 'Could not match this "
            "input' + \"Expected parameter 'TOLERANCE'\" from "
            "input_spec_builders.)",

            "[Input] A RESULT DESCRIPTION entry that names a "
            "discretisation which does not exist is NOT silently "
            "skipped — the run aborts. utils_result_test counts "
            "how many entries were actually evaluated and throws "
            "'expected N tests but performed M' when the count "
            "falls short, so a typo in DIS cannot fake a passing "
            "verification. A bad NODE or a bad QUANTITY throws "
            "from the field-specific tester instead. Signal: "
            "'expected 1 tests but performed 0' = wrong DIS "
            "name; 'Node 99999 does not belong to "
            "discretization structure' = wrong NODE; \"Quantity "
            "'banana' not supported in structure testing\" = "
            "wrong QUANTITY, all with exit code 1. (Verified by "
            "execution 2026-08-03: DIS 'structur' → 'expected 1 "
            "tests but performed 0' from utils_result_test; "
            "NODE 99999 → structure_new_resulttest node throw; "
            "QUANTITY 'banana' → structure_new_resulttest "
            "quantity throw.)",

            "[Input] PROBLEM SIZE is OPTIONAL and its "
            "ELEMENTS / NODES / MATERIALS / NPATCHES counts are "
            "read into a parameter list and then never used — "
            "the 4C source marks them 'unused parameters ... "
            "Misuse is possible'. Do not spend effort making "
            "them consistent, and never treat a mismatch as the "
            "cause of a failure. The one field that DOES matter "
            "is DIM, which defaults to 3. Signal: a run that "
            "completes normally with PROBLEM SIZE claiming 999 "
            "ELEMENTS for a 1-element mesh proves the counts are "
            "inert; conversely, a 2D deck that behaves as if it "
            "were 3D means DIM was left at its default in "
            "global_legacy_module_validparameters. (Verified by "
            "execution 2026-08-03: the same HEX8 deck run with "
            "no PROBLEM SIZE section at all, and with "
            "ELEMENTS: 999 / NODES: 7 against a real 1-element "
            "8-node mesh, both completed with exit 0 and "
            "identical results.)",

            "[Input] A FUNCT index referenced from a condition "
            "must resolve to a function 4C actually READ — it "
            "does not fall back to a constant. The reference is "
            "1-based, 0 means 'no function, use VAL directly', "
            "and the index is POSITIONAL rather than the number "
            "in the section name: 4C reads FUNCT1, FUNCT2, ... "
            "consecutively and stops at the first gap, so a deck "
            "that defines FUNCT1 and FUNCT7 and points at 7 "
            "fails exactly like a deck with no functions at all "
            "(the guard is 'functions_.size() < num'). Signal: "
            "'Function with index 7 (i.e. input FUNCT7) not "
            "available.' from "
            "core/utils/src/functions/4C_utils_function_manager.hpp, "
            "exit 1, raised "
            "at the first evaluation rather than at parse time — "
            "the input is read and the discretisation built "
            "first. A function of the wrong KIND fails "
            "differently: 'You tried to query function 7 as a "
            "function of type ...', which is what you get for a "
            "SYMBOLIC_FUNCTION_OF_TIME where a space-time "
            "function is needed. "
            "(Verified by execution 2026-08-03: Neumann "
            "condition with FUNCT: [0, 7, 0] and no FUNCT7 "
            "section. Extended by execution 2026-08-06: the "
            "same condition against FUNCT1+FUNCT7 only fails "
            "identically, while a complete FUNCT1..FUNCT7 run "
            "succeeds and reproduces the FUNCT-0 answer.)",

            "[Input] Condition arrays must match NUMDOF exactly, "
            "and NUMDOF itself must match the nodal DOF count "
            "for Dirichlet — but NOT for Neumann. A Dirichlet "
            "block with NUMDOF smaller than the element's DOFs "
            "passes the YAML spec and only fails later inside "
            "the DBC extractor; a Neumann block may declare MORE "
            "entries than the element has DOFs (the classic "
            "NUMDOF: 6 structural convention) and the surplus is "
            "ignored, but declaring FEWER is fatal. Signal: "
            "\"Candidate parameter 'ONOFF' has incorrect size\" "
            "(plus the same for VAL and FUNCT) from "
            "input_spec_builders when the arrays disagree with "
            "NUMDOF; '2 DOFs given but 3 expected in Surface "
            "Dirichlet boundary condition' from "
            "fem_discretization_utils_dbc when NUMDOF disagrees "
            "with the element; 'Fewer functions or curves "
            "defined than the element has dofs.' from "
            "solid_3D_ele_surface_evaluate for an under-sized "
            "Neumann block. (Verified by execution 2026-08-03 on "
            "a HEX8 deck: ONOFF/VAL/FUNCT of length 2 under "
            "NUMDOF: 3 → parse error; NUMDOF: 2 with matching "
            "length-2 arrays → DBC runtime throw; Neumann "
            "NUMDOF: 6 → exit 0 and bit-identical displacement "
            "to NUMDOF: 3; Neumann NUMDOF: 2 → surface-evaluate "
            "throw.)",

            "[Input] DESIGN ... CONDITIONS entity IDs are "
            "1-based in the input and must exist in the matching "
            "D<X>-NODE TOPOLOGY section; 4C does not silently "
            "drop a condition on an empty entity. Note the "
            "diagnostic reports the 0-based internal id, so "
            "'E: 5' is reported as 'DSurface 4'. Signal: "
            "'DSurface 4 not in range [0:2[' followed by "
            "'DSurface condition on non existent DSurface?Could "
            "not read set from entity type.' from fem_condition, "
            "exit 1. (Verified by execution 2026-08-03: "
            "DESIGN SURF DIRICH CONDITIONS on E: 5 with only "
            "DSURFACE 1 and 2 declared in DSURF-NODE TOPOLOGY.)",

            "[Input] The TYPE of a DESIGN ... NEUMANN CONDITION "
            "is validated in two separate places and the two "
            "vocabularies differ. The parser accepts "
            "Dead | Live | PressureGrad | orthopressure | "
            "pseudo_orthopressure, but the 3D SOLID element only "
            "implements Live, orthopressure and "
            "pseudo_orthopressure — 'Dead' and 'PressureGrad' "
            "parse cleanly and then abort at the first element "
            "evaluation. Use Live for a follower-free traction. "
            "Signal: \"Candidate deprecated_selection 'TYPE' has "
            "wrong value, possible values: Dead|Live|"
            "PressureGrad|orthopressure|pseudo_orthopressure\" "
            "for a value outside the parser enum; 'Unknown type "
            "of SurfaceNeumann condition' from "
            "solid_3D_ele_surface_evaluate for Dead or "
            "PressureGrad on a SOLID element. (Verified by "
            "execution 2026-08-03: TYPE 'Follower' → parser "
            "enumeration; TYPE 'Dead' → surface-evaluate throw; "
            "TYPE 'Live' → exit 0.)",

            "[Output] Runtime VTK output needs THREE things, not "
            "two: the parent IO/RUNTIME VTK OUTPUT section with a "
            "positive INTERVAL_STEPS; the per-field sub-section "
            "IO/RUNTIME VTK OUTPUT/STRUCTURE with "
            "OUTPUT_STRUCTURE: true; AND at least one field flag "
            "inside that sub-section, e.g. DISPLACEMENT: true. "
            "OUTPUT_STRUCTURE is only the master switch — every "
            "one of the 12 field flags next to it "
            "(DISPLACEMENT, VELOCITY, ACCELERATION, ELEMENT_OWNER, "
            "ELEMENT_GID, ELEMENT_GHOSTING, NODE_GID, "
            "ELEMENT_MAT_ID, STRESS_STRAIN, ...) defaults to "
            "false, and with none of them set the writer opens a "
            "geometry file, writes no point data and then dies. "
            "Missing INTERVAL_STEPS or the sub-section is the "
            "SILENT failure (0 files, exit 0, no warning — "
            "INTERVAL_STEPS defaults to -1 in inpar_io, meaning "
            "'never'); a sub-section with no field flag is the "
            "LOUD one. Signal: a run that finishes normally while "
            "the output directory holds only <prefix>.control and "
            "<prefix>.mesh.* and no .vtu/.pvd means INTERVAL_STEPS "
            "or the sub-section is missing; 'No data was written "
            "or writer was already in final phase.' from "
            "io_vtk_writer_base with exit 1 and a stack through "
            "VtkWriterBase::write_vtk_footers / "
            "VisualizationWriterVtuPerRank::finalize_time_step "
            "means the sub-section is present but every field flag "
            "is still false. (Verified by execution 2026-08-03, "
            "2-step HEX8 deck, 4C 2026.2.0-dev git 89519cf: "
            "OUTPUT_STRUCTURE + DISPLACEMENT → 3 .vtu + 3 .pvtu + "
            "1 .pvd, exit 0; OUTPUT_STRUCTURE alone → exit 1 with "
            "the write_vtk_footers throw after 1 orphan .vtu; "
            "DISPLACEMENT alone without OUTPUT_STRUCTURE → 0 "
            "files, exit 0; parent only → 0 files, exit 0; "
            "sub-section only, no INTERVAL_STEPS → 0 files, exit "
            "0; neither → 0 files, exit 0.)",

            "[Output] EVERY_ITERATION does exist, but under "
            "IO/RUNTIME VTK OUTPUT — not under plain IO. Putting "
            "it in IO is a hard parse error, which is the "
            "friendly case; the dangerous neighbour is "
            "RESTARTEVERY, which exists in BOTH sections. The "
            "structural time integrator reads RESTARTEVERY from "
            "STRUCTURAL DYNAMIC (structure_new_timint_basedataio), "
            "so an IO-level RESTARTEVERY parses, runs, exits 0 "
            "and writes no restart records at all; the mistake "
            "only surfaces on the restart attempt, possibly "
            "hours later. The same per-field placement applies to "
            "FLUID DYNAMIC, SCALAR TRANSPORT DYNAMIC and "
            "PARTICLE DYNAMIC. Signal: EVERY_ITERATION under IO "
            "gives 'Could not match this input' from "
            "input_spec_builders; a misplaced RESTARTEVERY gives "
            "a clean run whose output contains no "
            "<prefix>.result.<field>.s<N> file, and the later "
            "restart fails with \"No restart entry for "
            "discretization 'structure' step 2 in control file. "
            "Control file corrupt?\" from io_control. (Verified "
            "by execution 2026-08-03: RESTARTEVERY: 2 under IO → "
            "exit 0, only .control and .mesh.structure.s0 "
            "written, --restart=2 then failed with the "
            "io_control message; the same value under STRUCTURAL "
            "DYNAMIC → .result.structure.s2 written and "
            "--restart=2 resumed at step 3 of 4 and finished "
            "with exit 0.)",

            "[API] `4C --parameters` dumps the complete, "
            "version-exact input schema of the installed binary "
            "as YAML on stdout — 2.9 MB on 4C 2026.2.0-dev — "
            "with seven top-level keys: metadata (commit_hash, "
            "version), sections, legacy_element_specs, "
            "legacy_particle_specs, cell_types, $references, "
            "legacy_string_sections. Every parameter carries its "
            "name, type, required flag and default. Prefer "
            "querying this over guessing a keyword or trusting a "
            "catalogue written against a different 4C release. "
            "Signal: the dump begins 'metadata:' / "
            "'commit_hash:' / 'version:'; if a section or key an "
            "agent wants to write is absent from this dump, it "
            "does not exist in the installed build regardless of "
            "what any documentation says. Redirect stdout ALONE "
            "(2>/dev/null) when you size or hash the dump — 4C's "
            "stderr carries unrelated host noise. On this build the "
            "dump declares 478 input sections and 60 legacy element "
            "names, so `--parameters` is also the right way to ask "
            "'does section X exist' before writing it. (Verified by "
            "execution 2026-08-03 on 4C 2026.2.0-dev git 89519cf: "
            "`4C --parameters 2>/dev/null` exited 0 and emitted "
            "2 926 432 bytes — a previously recorded 2 926 462 was "
            "measured with 2>&1 and included 30 bytes of local X11 "
            "warning — whose legacy_element_specs listed WALL with "
            "required MAT / KINEM / EAS / THICK / STRESS_STRAIN / GP "
            "entries and SOLID with the nine 3D cell types HEX8, "
            "HEX18, HEX20, HEX27, TET4, TET10, WEDGE6, PYRAMID5, "
            "NURBS27.)",

            "[Input] The NUE validator is in_range[-1,0.5) — the "
            "bracket at the LOW end is CLOSED, so NUE: -1.0 passes "
            "validation and is then fatal at solve time. It is the "
            "one admissible Poisson value that gets you a SIGFPE "
            "instead of either a number or a parse error, because "
            "the Lame constants blow up as NUE -> -1. Keep NUE "
            "strictly inside (-1, 0.5); the useful engineering range "
            "is [0, 0.5). Signal: NUE 0.5 and above are rejected at "
            "parse with \"Candidate parameter 'NUE' does not pass "
            "validation: in_range[-1,0.5)\" and exit 1, but NUE -1.0 "
            "produces NO parse complaint and the shell reports "
            "signal 8 / exit status 136. (Verified by execution "
            "2026-08-03 on 4C 2026.2.0-dev git 89519cf, HEX8 "
            "Statics cantilever with MAT_Struct_StVenantKirchhoff: "
            "NUE 0.49 → exit 0, dispy 3.48962414247091460e-03; "
            "NUE 0.5 → exit 1 at parse; NUE 0.6 → exit 1 at parse; "
            "NUE -1.0 → killed by SIGFPE with no material "
            "diagnostic.)",
        ],

        "element_type_per_physics": {
            "FLUID (2D)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "FLUID (3D)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10", "NURBS27"],
            # NOTE: 4C 2026.2.0 unified the legacy WALL 2D eletype
            # into the SOLID eletype factory. Writing 'WALL QUAD4'
            # raises "Unknown type 'WALL' of finite element" from
            # parobjectfactory.cpp:153 — see SOL_MECH [API] pitfall.
            #
            # 2026-08-03: the boundary runs the OTHER way on older
            # builds. On the March 2026 development build (2026.2.0-dev) the SOLID factory registers
            # 3D cell types only and 2D is the WALL eletype below —
            # verified by execution, and by 109 of the 1974 decks in
            # that tree's tests/input_files using WALL against 3 using
            # SOLID QUAD4 (those 3 do not run). Probe the installed
            # build rather than assuming; see structural_mechanics
            # pitfall #7 and the Tier-2 fixture
            # scripts/tier2_fixtures/fourc/
            # structural_2d_solid_quad4_not_wall, which prints a
            # VERDICT: 2D_ELEMENT=<WALL|SOLID> line.
            "WALL (2D structure, 4C before 2026.2.0)": [
                "QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6",
                "NURBS4", "NURBS9",
                "-- required keys: MAT, KINEM, EAS, THICK, "
                "STRESS_STRAIN, GP (2 ints)"],
            "SOLID (2D structure, 4C 2026.2.0 and later)": [
                "QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6",
                "-- required keys: MAT, KINEM, THICKNESS, PLANE_ASSUMPTION"],
            "SOLID (3D structure)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10",
                                     "WEDGE6", "PYRAMID5"],
            "TRANSP (scalar transport)": ["QUAD4", "QUAD9", "HEX8", "HEX27",
                                          "TRI3", "TRI6", "TET4", "TET10"],
            "SOLIDSCATRA (TSI/SSI)": ["HEX8", "TET4", "TET10", "HEX27"],
            "SOLIDSCATRA 2D (4C 2026.2.0 and later)": [
                "QUAD4", "QUAD9", "TRI3", "TRI6",
                "-- listed by 4C 2026.2.0 and 2026.3.0; a TSI deck with QUAD4 stops at "
                "'Unsupported solid element type!'"],
            "ALE (2D)": ["QUAD4", "TRI3"],
            "ALE (3D)": ["HEX8", "TET4"],
            "PORO (2D, 4C before 2026.2.0)": ["WALLQ4PORO QUAD4", "WALLQ9PORO QUAD9"],
            "PORO (2D, 4C 2026.2.0 and later)": [
                "SOLIDPORO_PRESSURE_VELOCITY_BASED QUAD4",
                "SOLIDPORO_PRESSURE_VELOCITY_BASED QUAD9",
                "-- with THICKNESS and PLANE_ASSUMPTION"],
            "PORO (3D)": ["SOLIDPORO_PRESSURE_VELOCITY_BASED HEX8",
                          "SOLIDPORO_PRESSURE_VELOCITY_BASED TET4",
                          "SOLIDPORO_PRESSURE_VELOCITY_BASED HEX27"],
            "BEAM": ["BEAM3R LINE2", "BEAM3EB LINE2", "BEAM3R LINE3"],
            "ARTERY": ["ART LINE2"],  # the element is ART, written in section ARTERY ELEMENTS
            "notes": (
                "QUAD4 is the workhorse element for most 2D problems.  "
                "HEX8 for 3D.  Higher-order elements (QUAD9, HEX27) give "
                "better accuracy but are slower.  TRI3/TET4 are available "
                "but less accurate for pressure in fluid problems."
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # RESULT DESCRIPTION — 4C's built-in numerical self-check
    #
    # Added 2026-08-03. Sources read:
    #   src/global_legacy_module/4C_global_legacy_module.cpp
    #       valid_result_lines()          -> the 18 field groups
    #   src/core/utils/src/result_test/4C_utils_result_test.cpp
    #       ResultTest::test_*/test_all() -> compare + abort logic
    #   src/structure_new/src/utils/4C_structure_new_resulttest.cpp
    #       get_nodal_result()            -> STRUCTURE QUANTITY names
    # Every behavioural claim below was then reproduced by running
    # the deployed 4C binary (2026.2.0-dev, git 89519cf) on a
    # minimal HEX8 cantilever and a minimal HEX8 Thermo cube.
    # NOTE: deliberately has no "pitfalls" key — the actionable
    # pitfalls live in input_format.pitfalls so the Signal-verification
    # harness harvests them under a registered physics name.
    # ═══════════════════════════════════════════════════════════════════════
    "result_description": {
        "description": (
            "RESULT DESCRIPTION is a list of point-wise assertions "
            "that 4C evaluates after the time loop. It is the "
            "cheapest way for an agent to make a run verify itself: "
            "a violated assertion aborts the process with a non-zero "
            "exit code, so correctness can be gated on the process "
            "status rather than on parsing solver output. 1925 of "
            "the 1974 decks in tests/input_files carry one."
        ),
        "section_name": "RESULT DESCRIPTION",
        "yaml_example": """
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"        # discretisation name, lowercase
      NODE: 3                 # 1-based node id (or LINE/SURFACE/VOLUME)
      QUANTITY: "dispy"
      VALUE: -0.007071067811865476
      TOLERANCE: 1e-11        # must be > 0
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
""",
        "field_groups": [
            "STRUCTURE", "FLUID", "XFLUID", "ALE", "THERMAL",
            "LUBRICATION", "POROFLUIDMULTIPHASE", "SCATRA", "SSI",
            "SSTI", "STI", "RED_AIRWAY", "ARTNET", "FSI", "PARTICLE",
            "PARTICLEWALL", "RIGIDBODY", "CARDIOVASCULAR0D",
        ],
        "field_group_notes": (
            "The thermal group is spelled THERMAL, not THERMO — the "
            "same asymmetry as the THERMAL DYNAMIC section name "
            "versus the THERMO element type. A wrong group name is "
            "rejected at parse time and 4C echoes the full list of "
            "18 valid groups, which makes the error self-correcting. "
            "Verified by execution 2026-08-03: a THERMO group on a "
            "Thermo problem printed \"Expected group 'STRUCTURE' ... "
            "'THERMAL' ... 'CARDIOVASCULAR0D'\"; THERMAL matched."
        ),
        "required_keys": {
            "DIS": (
                "Discretisation name, lowercase: 'structure', "
                "'fluid', 'thermo', 'ale', 'scatra', ... Required "
                "for every group except when SPECIAL: true is used."),
            "NODE | LINE | SURFACE | VOLUME": (
                "Exactly one geometric selector, 1-based. NODE is by "
                "far the most common. Exactly four groups also accept "
                "an ELEMENT selector: ARTNET, FLUID, "
                "POROFLUIDMULTIPHASE and RED_AIRWAY. (Corrected "
                "2026-08-03 from the `4C --parameters` dump of 4C "
                "2026.2.0-dev git 89519cf, which had previously been "
                "recorded as FLUID and POROFLUIDMULTIPHASE only.)"),
            "QUANTITY": "Field-component name, see quantities below.",
            "VALUE": "The expected number (double).",
            "TOLERANCE": (
                "Absolute tolerance, strictly positive. The check is "
                "abs(actresult - VALUE) > TOLERANCE."),
        },
        "optional_keys": {
            "NAME": (
                "Free-text label echoed in the report line. Accepted "
                "by all 18 groups."),
            "OP": (
                "STRUCTURE only. Enum with exactly four choices — "
                "sum | max | min | unknown — default 'unknown'."),
            "SPECIAL": (
                "bool; selects a whole-field special test instead of a "
                "point test. Accepted by NINE groups: CARDIOVASCULAR0D, "
                "FSI, PARTICLEWALL, POROFLUIDMULTIPHASE, SCATRA, SSI, "
                "SSTI, STI and STRUCTURE. (Corrected 2026-08-03 from "
                "the `4C --parameters` dump of 4C 2026.2.0-dev git "
                "89519cf; the list had previously named only "
                "STRUCTURE, SCATRA and POROFLUIDMULTIPHASE.)"),
        },
        "structure_quantities": [
            "dispx", "dispy", "dispz",
            "velx", "vely", "velz",
            "accx", "accy", "accz",
            "reactx", "reacty", "reactz",
            "press", "stress", "strain",
        ],
        "structure_quantities_source": (
            "src/structure_new/src/utils/4C_structure_new_resulttest.cpp "
            "get_nodal_result(); an unlisted name raises \"Quantity "
            "'<name>' not supported in structure testing\"."),
        "exit_semantics": {
            "all_pass": (
                "One 'is CORRECT, abs(diff)= ... < ...' line per "
                "entry, then normal shutdown, exit code 0."),
            "any_fail": (
                "'is WRONG --> actresult= ..., givenresult= ..., "
                "abs(diff)= ... > ...' per failing entry, then "
                "FOUR_C_THROW 'Result check failed with N errors out "
                "of M tests', MPI_Abort, exit code 1."),
            "not_evaluated": (
                "If fewer entries were evaluated than were listed "
                "(typically a wrong DIS name), 4C throws 'expected N "
                "tests but performed M' — an unmatched entry can "
                "never masquerade as a pass."),
            "no_section": (
                "'Checking results of 0 tests:' / 'OK (0)', exit 0. "
                "Absence of the section means absence of checking, "
                "not a passing check."),
        },
        "capture_note": (
            "ALWAYS launch the binary under `stdbuf -oL -eL`. The "
            "failure diagnostic is written to raw std::cout and is "
            "discarded by MPI_Abort whenever stdout is block-buffered, "
            "leaving only 'Checking results of N tests:' and exit "
            "code 1. Redirecting to a file does NOT rescue it — a "
            "regular file is fully buffered as well; only line "
            "buffering does. Corrected 2026-08-03 after execution: "
            "`4C in out > f.log 2>&1` produced zero 'is WRONG' lines "
            "on three consecutive runs of a deliberately failing "
            "deck, while the same command under stdbuf -oL -eL "
            "produced both the 'is WRONG' and the 'Result check "
            "failed' line."
        ),
        "agent_recipe": (
            "1. Run once with a deliberately huge TOLERANCE (e.g. "
            "1e30) and a placeholder VALUE — the 'is CORRECT, "
            "abs(diff)= X' line then reports X = the true magnitude "
            "of the quantity, at full 17-digit precision, for free. "
            "2. Copy X into VALUE and set TOLERANCE to a physically "
            "meaningful band. 3. From then on the deck is "
            "self-verifying and any regression flips the exit code. "
            "This is exactly how the 2026-08-03 execution sweep "
            "extracted reference values without post-processing a "
            "single VTU file."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CELL TYPES
    # ═══════════════════════════════════════════════════════════════════════
    "cell_types": {
        "1D": ["line2", "line3", "line4", "line5", "line6", "point1"],
        "2D": ["quad4", "quad6", "quad8", "quad9", "tri3", "tri6"],
        "3D": ["hex8", "hex16", "hex18", "hex20", "hex27", "tet4", "tet10",
               "wedge6", "wedge15", "pyramid5"],
        "NURBS": ["nurbs2", "nurbs3 (1D)", "nurbs4", "nurbs9 (2D)",
                  "nurbs8", "nurbs27 (3D)"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # XFEM
    # ═══════════════════════════════════════════════════════════════════════
    "xfem": {
        "description": "Extended Finite Element Method for interface problems",
        "capabilities": [
            "Level-set based interfaces (weak Dirichlet, Neumann, Navier slip, two-phase)",
            "Surface-based interfaces (displacement, FSI, FPI)",
            "Robin conditions (Dirichlet/Neumann)",
            "Edge stabilization",
            "Semi-Lagrangean time integration",
        ],
        "applications": "Fluid-XFEM, FSI-XFEM (no mesh conformity at interface)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ALL 41 PROBLEM TYPES (4C 2026.1.0 has 40: no Reduced_Lung_1D_Pipe_Flow)
    # ═══════════════════════════════════════════════════════════════════════
    "all_problem_types": {
        "Structure": "Structural mechanics",
        "Scalar_Transport": "Convection-diffusion / scalar transport",
        "Thermo": "Pure thermal analysis",
        "Fluid": "Incompressible Navier-Stokes",
        "Fluid_Ale": "Fluid on ALE mesh",
        "Ale": "Pure ALE mesh movement",
        "Fluid_Structure_Interaction": "FSI (standard)",
        "Fluid_Structure_Interaction_XFEM": "FSI with XFEM",
        "Thermo_Structure_Interaction": "TSI",
        "Structure_Scalar_Interaction": "SSI (electrode mechanics, etc.)",
        "Structure_Scalar_Thermo_Interaction": "SSTI (three-field)",
        "Scalar_Thermo_Interaction": "STI",
        "Fluid_Beam_Interaction": "3D fluid + 1D beam",
        "Fluid_Porous_Structure_Interaction": "FPSI",
        "Particle": "SPH / DEM / Peridynamics",
        "Particle_Structure_Interaction": "PASI",
        "Poroelasticity": "Biot poroelasticity",
        "Poroelastic_scalar_transport": "Poro + scalar",
        "Level_Set": "Level-set interface tracking",
        "Low_Mach_Number_Flow": "Variable-density flow",
        "Lubrication": "Thin film lubrication",
        "Elastohydrodynamic_Lubrication": "EHL coupling",
        "Electrochemistry": "Nernst-Planck electrochemistry",
        "ArterialNetwork": "1D arterial blood flow",
        "ReducedDimensionalAirWays": "Lung airways",
        "Cardiac_Monodomain": "Cardiac electrophysiology",
        "Biofilm_Fluid_Structure_Interaction": "Biofilm FSI",
        "Gas_Fluid_Structure_Interaction": "Gas + FSI",
        "Polymer_Network": "Polymer network",
        "Fluid_XFEM": "Fluid with XFEM (enum fluid_xfem, fluid_xfem_drt)",
        "Fluid_RedModels": "Fluid with reduced models (enum fluid_redmodels, dyn_fluid_drt)",
        "Fluid_Structure_Interaction_RedModels": "FSI with reduced models (enum fsi_redmodels, fsi_ale_drt)",
        "Fluid_Poro_Structure_Interaction_XFEM": "FPSI with XFEM (enum fpsi_xfem, xfpsi_drt)",
        "Fluid_Porous_Structure_Scalar_Scalar_Interaction": "FPS3I (enum fps3i, fs3i_dyn)",
        "Thermo_Fluid_Structure_Interaction": "Thermo-FSI (enum thermo_fsi, fs3i_dyn)",
        "Reduced_Lung": "Reduced lung (enum reduced_lung, ReducedLung::reduced_lung_main)",
        "Reduced_Lung_1D_Pipe_Flow": "1D pipe flow (enum one_d_pipe_flow, ReducedLung1dPipeFlow::main); not in 4C 2026.1.0",
        "porofluid_pressure_based": "Pressure-based porous-medium flow (porofluid_pressure_based_dyn)",
        "porofluid_pressure_based_elasticity": "porofluid_pressure_based + elasticity (enum porofluid_pressure_based_elast, porofluid_elast_dyn)",
        "porofluid_pressure_based_elasticity_scatra": "porofluid_pressure_based + elasticity + scalar transport (enum porofluid_pressure_based_elast_scatra, porofluid_pressure_based_elast_scatra_dyn)",
        "NP_Supporting_Procs": "Supporting processes for a multiscale run's micro problems (enum np_support, MultiScale::np_support_drt)",
    },
}
