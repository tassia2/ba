x = [1 2 3 4];
h1sin = [0.999634 0.101971 0.00675344 0.000334022];
h1 = [0.29856 0 2.368*10^(-9) 3.90031*10^(-9)];
[Ax,H1,H2]=plotyy(x,h1sin,x,h1,'plot');
legend('Error in the H^1-norm for the first example','Error in the H^1-norm for the second example');
xlabel('Polynomial degree');