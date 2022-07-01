#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2011-2021 Vincent Heuveline
#
# HiFlow3 is free software: you can redistribute it and/or modify it under the
# terms of the European Union Public Licence (EUPL) v1.2 as published by the
#/ European Union or (at your option) any later version.
#
# HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the European Union Public Licence (EUPL) v1.2 for more
# details.
#
# You should have received a copy of the European Union Public Licence (EUPL) v1.2
# along with HiFlow3.  If not, see <https://joinup.ec.europa.eu/page/eupl-text-11-12>.

## @author Hartwig Anzt

import os
import sys
import shutil
from subprocess import *
import re

srcdir = os.path.abspath(sys.argv[1])

#-------------------------------------------------------------------------------
# generate the pdfs from the latex source files

#TEMP Tutorial
#os.chdir(srcdir+"/doc/tutorials/gpu-latoolbox/")
#os.system("pdflatex tut_gpu.tex")
#os.system("makeindex tut_gpu.tex")
#os.system("bibtex tut_gpu.aux")
#os.system("pdflatex tut_gpu.tex")
#os.system("pdflatex tut_gpu.tex")
#os.system("cp tut_gpu.pdf " + srcdir + "/examples/convection_diffusion/.")

#Direct Inverse Tutorial
os.chdir(srcdir+"/doc/tutorials/direct_inverse/")
os.system("pdflatex tut_direct_inverse.tex")
os.system("makeindex tut_direct_inverse.tex")
os.system("bibtex tut_direct_inverse.aux")
os.system("pdflatex tut_direct_inverse.tex")
os.system("pdflatex tut_direct_inverse.tex")
os.system("cp tut_direct_inverse.pdf " + srcdir + "/examples/direct_inverse/.")

#Porous Media Tutorial
os.chdir(srcdir+"/doc/tutorials/porous_media/")
os.system("pdflatex tut_porous_media.tex")
os.system("makeindex tut_porous_media.tex")
os.system("bibtex tut_porous_media.aux")
os.system("pdflatex tut_porous_media.tex")
os.system("pdflatex tut_porous_media.tex")
os.system("cp tut_porous_media.pdf " + srcdir + "/examples/porous_media/.")

#Poisson Tutorial
os.chdir(srcdir + "/doc/tutorials/poisson/")
os.system("pdflatex tut_Poisson.tex")
os.system("makeindex tut_Poisson.tex")
os.system("bibtex tut_Poisson.aux")
os.system("pdflatex tut_Poisson.tex")
os.system("pdflatex tut_Poisson.tex")
os.system("cp tut_Poisson.pdf " + srcdir + "/examples/poisson/.")

#Flow Tutorial
os.chdir(srcdir+"/doc/tutorials/flow/")
os.system("pdflatex tut_Navier_Stokes.tex")
os.system("makeindex tut_Navier_Stokes.tex")
os.system("bibtex tut_Navier_Stokes.aux")
os.system("pdflatex tut_Navier_Stokes.tex")
os.system("pdflatex tut_Navier_Stokes.tex")
os.system("cp tut_Navier_Stokes.pdf " + srcdir + "/examples/flow/.")

#GPU Tutorial
os.chdir(srcdir+"/doc/tutorials/gpu-latoolbox/")
os.system("pdflatex tut_gpu.tex")
os.system("makeindex tut_gpu.tex")
os.system("bibtex tut_gpu.aux")
os.system("pdflatex tut_gpu.tex")
os.system("pdflatex tut_gpu.tex")
os.system("cp tut_gpu.pdf " + srcdir + "/examples/convection_diffusion/.")

#convection_diffusion_instationary Tutorial
os.chdir(srcdir+"/doc/tutorials/convection_diffusion_instationary/")
os.system("pdflatex tut_convdiff_instat.tex")
os.system("makeindex tut_convdiff_instat.tex")
os.system("bibtex tut_convdiff_instat.aux")
os.system("pdflatex tut_convdiff_instat.tex")
os.system("pdflatex tut_convdiff_instat.tex")
os.system("cp tut_convdiff_instat.pdf " + srcdir + "/examples/convection_diffusion/.")

#convection_diffusion_stabilization Tutorial
os.chdir(srcdir+"/doc/tutorials/convection_diffusion_stabilization/")
os.system("pdflatex tut_convdiff_stab.tex")
os.system("makeindex tut_convdiff_stab.tex")
os.system("bibtex tut_convdiff_stab.aux")
os.system("pdflatex tut_convdiff_stab.tex")
os.system("pdflatex tut_convdiff_stab.tex")
os.system("cp tut_convdiff_stab.pdf " + srcdir + "/examples/convection_diffusion/.")

#distributed_control_poisson Tutorial
os.chdir(srcdir+"/doc/tutorials/distributed_control_poisson/")
os.system("pdflatex tut_distributed_control_Poisson.tex")
os.system("makeindex tut_distributed_control_Poisson.tex")
os.system("bibtex tut_distributed_control_Poisson.aux")
os.system("pdflatex tut_distributed_control_Poisson.tex")
os.system("pdflatex tut_distributed_control_Poisson.tex")
os.system("cp tut_distributed_control_Poisson.pdf " + srcdir + "/examples/distributed_control_poisson/.")

#newton Tutorial
os.chdir(srcdir+"/doc/tutorials/newton/")
os.system("pdflatex tut_inexact_newton_method.tex")
os.system("makeindex tut_inexact_newton_method.tex")
os.system("bibtex tut_inexact_newton_method.aux")
os.system("pdflatex tut_inexact_newton_method.tex")
os.system("pdflatex tut_inexact_newton_method.tex")
os.system("cp tut_inexact_newton_method.pdf " + srcdir + "/examples/newton/.")

#Elasticity Tutorial
os.chdir(srcdir+"/doc/tutorials/elasticity/")
os.system("pdflatex tut_Elasticity.tex")
os.system("makeindex tut_Elasticity.tex")
os.system("bibtex tut_Elasticity.aux")
os.system("pdflatex tut_Elasticity.tex")
os.system("pdflatex tut_Elasticity.tex")
os.system("cp tut_Elasticity.pdf " + srcdir + "/examples/elasticity/.")
os.system("rm -rf Code/ liver.pdf liver.pdf_tex")

#Blood Flow Tutorial
os.chdir(srcdir + "/doc/tutorials/blood_flow/")
os.system("pdflatex tut_blood_flow.tex")
os.system("bibtex tut_blood_flow.aux")
os.system("pdflatex tut_blood_flow.tex")
os.system("pdflatex tut_blood_flow.tex")
os.system("cp tut_blood_flow.pdf " + srcdir + "/examples/blood_flow/.")

#Poisson Adaptive Tutorial
os.chdir(srcdir + "/doc/tutorials/poisson_adaptive/")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("bibtex tut_Poisson_adaptive.aux")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("cp tut_Poisson_adaptive.pdf " + srcdir + "/examples/poisson_adaptive/.")

#Poisson Uncertainty Tutorial
os.chdir(srcdir + "/doc/tutorials/poisson_uncertainty/")
os.system("pdflatex tut_stochastic_poisson.tex")
os.system("makeindex tut_stochastic_poisson.tex")
os.system("bibtex tut_stochastic_poisson.aux")
os.system("pdflatex tut_stochastic_poisson.tex")
os.system("pdflatex tut_stochastic_poisson.tex")
os.system("cp tut_stochastic_poisson.pdf " + srcdir + "/examples/poisson_uncertainty/.")

#Blood flow Tutorial
os.chdir(srcdir + "/doc/tutorials/blood_flow/")
os.system("pdflatex tut_blood_flow.tex")
os.system("makeindex tut_blood_flow.tex")
os.system("bibtex tut_blood_flow.aux")
os.system("pdflatex tut_blood_flow.tex")
os.system("pdflatex tut_blood_flow.tex")
os.system("cp tut_blood_flow.pdf " + srcdir + "/examples/blood_flow/.")

#Poisson adaptive Tutorial
os.chdir(srcdir + "/doc/tutorials/poisson_adaptive/")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("makeindex tut_Poisson_adaptive.tex")
os.system("bibtex tut_Poisson_adaptive.aux")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("pdflatex tut_Poisson_adaptive.tex")
os.system("cp tut_Poisson_adaptive.pdf " + srcdir + "/examples/poisson_adaptive/.")

#rm *tex files in tutorial directories
os.chdir(srcdir + "/doc/tutorials/")
os.system("rm */*.tex */*.tex~ */*.bib */*.bib~ */*.png */emcl.pdf */*.txt */*.aux */*.bbl */*.blg */*idx */*ilg */*ind */*log */*out */*toc */*lof */*lot */*sty */*cc */*.h ; rm -rf */fig/; rm -rf */Matlab/")

#rm POOSC10/ directory
os.chdir(srcdir + "/doc/")
shutil.rmtree("POOSC10/")

#rm mesh tutorial
os.chdir(srcdir + "/doc/")
os.remove("mesh_tutorial.html")
os.remove("mesh_tutorial.org")
os.remove("mesh_tutorial.txt")

os.chdir(srcdir)
