x = [25 81 289 1089 4225];
l2 = [0.0330237 0.00818547 0.00204158 0.000510089 0.000127503];
h1 = [0.599637 0.29856 0.149123 0.0745421 0.0372686];
[AX,H1,H2] = plotyy(x,l2,x,h1,'semilogx'); 
legend('Error in the L^2-norm','Error in the H^1-norm');
%ylabel(AX(1), 'Error in the L^2-norm');
%ylabel(AX(2), 'Error in the H^1-norm'); 
xlabel('Number of DoF');