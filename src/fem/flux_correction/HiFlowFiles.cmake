set(flux_correction_SOURCES
)

# the following source files are only included for developping purpose
if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND flux_correction_SOURCES   
    flux_correction.cc
)
endif()

set(flux_correction_PUBLIC_HEADERS
 afc_tools.h
 flux_limiter.h
 )
