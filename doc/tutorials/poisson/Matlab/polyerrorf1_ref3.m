%refinement_level = 3
x = [1 2 3 4];
l2 = [0.0255247 0.00193251 8.81413*10^(-5) 0.334932*10^(-6)];
h1 = [0.999634 0.101971 0.00675344 0.000334022];
[Ax,H1,H2]=plotyy(x,l2,x,h1,'plot');
legend('Error in the L^2-norm','Error in the H^1-norm');
xlabel('Polynomial degree');
