cmake_minimum_required(VERSION 3.13)

find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
             $ENV{HOME}/dealii/build $ENV{HOME}/dealii)
deal_ii_initialize_cached_variables()

project(heat_iface_dealii CXX)

add_executable(heat_iface_dealii heat_iface_dealii.cc)
deal_ii_setup_target(heat_iface_dealii)

# ── the served program scaffold, its holes filled: what participant_dealii.py runs ──
# Served with the holes elided (knowledge(topic='coupling', solver='dealii') and
# write_participant_contract); this unserved copy is what the suite executes.
add_executable(dealii_side dealii_side.cc)
deal_ii_setup_target(dealii_side)

# ── the served TRANSIENT program scaffold, its holes filled: what participant_dealii_transient.py runs ──
add_executable(dealii_side_transient dealii_side_transient.cc)
deal_ii_setup_target(dealii_side_transient)

# ── the served ELASTIC program scaffold, its holes filled: what participant_dealii_elastic.py runs ──
add_executable(dealii_side_elastic dealii_side_elastic.cc)
deal_ii_setup_target(dealii_side_elastic)

# ── the TRANSIENT scalar participant: theta-scheme heat, waveform exchange ──
# The deal.II side of a time-dependent coupling. Needs nothing beyond deal.II
# itself (the three problem functions arrive as muparser expressions, and
# muparser is part of every standard deal.II build).
add_executable(heat_iface_dealii_transient heat_iface_dealii_transient.cc)
deal_ii_setup_target(heat_iface_dealii_transient)

# ── the VECTOR participant: plane-strain elasticity, displacement/traction ──
# The deal.II side of vector-valued interface exchange. Built unconditionally
# alongside the scalar one; it needs nothing beyond deal.II itself.
add_executable(elast_iface_dealii elast_iface_dealii.cc)
deal_ii_setup_target(elast_iface_dealii)

# ── the preCICE participant, built ONLY if libprecice is actually here ──────
#
# deal.II has no Python API, so a preCICE participant has to be a compiled
# executable that links libprecice.  This target is OPTIONAL on purpose: the
# `couple` participant above must keep building on an install with no preCICE,
# and a hard `find_package(precice)` would break it there.
#
# WHERE preCICE IS, WITHOUT NAMING A MACHINE. This file is served verbatim to
# agents on installs that are not the one it was written on, so it must not
# carry a path from the build host. In order of precedence: -DPRECICE_ROOT=...,
# then $PRECICE_ROOT from the environment, then whatever CMake's own search
# finds on the default prefixes. An empty PRECICE_ROOT is fine — find_library
# then searches the system paths, which is where a package-manager install of
# preCICE already is.
#
# A hand-rolled `g++ -I<dealii>/include` does NOT work — deal.II's bundled
# headers need the flags deal_ii_setup_target() supplies — so the preCICE
# include/lib are added ON TOP of deal.II's own target setup, not instead.
set(PRECICE_ROOT "$ENV{PRECICE_ROOT}" CACHE PATH
    "preCICE install prefix (optional; leave empty to search system paths)")
find_library(PRECICE_LIBRARY NAMES precice
             HINTS ${PRECICE_ROOT}/lib ${PRECICE_ROOT}/lib64)
find_path(PRECICE_INCLUDE_DIR precice/precice.hpp
          HINTS ${PRECICE_ROOT}/include)

if(PRECICE_LIBRARY AND PRECICE_INCLUDE_DIR)
  message(STATUS "preCICE found: ${PRECICE_LIBRARY} — building "
                 "precice_heat_dealii")
  add_executable(precice_heat_dealii precice_heat_dealii.cc)
  deal_ii_setup_target(precice_heat_dealii)
  target_include_directories(precice_heat_dealii PRIVATE ${PRECICE_INCLUDE_DIR})
  target_link_libraries(precice_heat_dealii ${PRECICE_LIBRARY})
else()
  message(STATUS "preCICE not found (PRECICE_ROOT='${PRECICE_ROOT}') — "
                 "skipping precice_heat_dealii; pass -DPRECICE_ROOT=<prefix> "
                 "if it is installed somewhere CMake does not search. The "
                 "`couple` participant still builds.")
endif()
