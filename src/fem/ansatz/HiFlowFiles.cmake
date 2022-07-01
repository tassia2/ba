set(ansatz_SOURCES
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND ansatz_SOURCES   
    ansatz.cc
)
endif()

set(ansatz_PUBLIC_HEADERS
  ansatz_space.h
  ansatz_sum.h
  ansatz_transformed.h
  ansatz_p_line_lagrange.h
  ansatz_p_tri_lagrange.h
  ansatz_p_tet_lagrange.h
  ansatz_pyr_lagrange.h
  ansatz_q_quad_lagrange.h
  ansatz_q_hex_lagrange.h
  ansatz_aug_p_tri_mono.h
  ansatz_aug_p_tet_mono.h
  ansatz_skew_aug_p_tri_mono.h
  ansatz_skew_aug_p_tet_mono.h
  ansatz_skew_aug_p_hex_mono.h
  ansatz_skew_aug_p_quad_mono.h
 )
