set(nonlinear_SOURCES
   )

if(EXPLICIT_TEMPLATE_INSTANTS)
  list(APPEND nonlinear_SOURCES   
    nonlinear.cc
  )
endif()

set(nonlinear_PUBLIC_HEADERS
  nonlinear_problem.h
  nonlinear_solver.h
  nonlinear_solver_creator.h
  nonlinear_solver_factory.h
  newton.h
  damping_strategy.h
  damping_armijo.h
  forcing_strategy.h
  forcing_eisenstat_walker.h
  )
