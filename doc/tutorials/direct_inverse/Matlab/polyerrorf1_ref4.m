%refinement_level = 4
x = [1 2 3 4];
l2 = [0.0064131 0.00024512 0.0000055641 0.00000010535];
h1 = [0.50264 0.025525 0.00084664 0.000020942];
[AX,H1,H2] = plotyy(x,l2,x,h1,'plot');
legend('Error in the L^2-norm','Error in the H^1-norm');
%ylabel(AX(1), 'Error in the L^2-norm');
%ylabel(AX(2), 'Error in the H^1-norm'); 
xlabel('Polynomial degree')