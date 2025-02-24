if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    add_compile_definitions(LINK_WITH_LIBPGMATH)
    message(STATUS "Found libpgmath: ${LIBPGMATH}")
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()

add_flang_library(FortranEvaluate
  call.cpp
  characteristics.cpp
  check-expression.cpp
  common.cpp
  complex.cpp
  constant.cpp
  expression.cpp
  fold.cpp
  fold-character.cpp
  fold-complex.cpp
  fold-designator.cpp
  fold-integer.cpp
  fold-logical.cpp
  fold-real.cpp
  fold-reduction.cpp
  formatting.cpp
  host.cpp
  initial-image.cpp
  integer.cpp
  intrinsics.cpp
  intrinsics-library.cpp
  logical.cpp
  real.cpp
  shape.cpp
  static-data.cpp
  tools.cpp
  type.cpp
  variable.cpp

  LINK_LIBS
  FortranCommon
  FortranDecimal
  FortranParser
  ${LIBPGMATH}

  LINK_COMPONENTS
  Support

  DEPENDS
  acc_gen
  omp_gen
)

