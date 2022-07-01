Include "channel_params.geo";

// resolution near to the outer frame
res_bdy = resolution_bdy * dx;

// resolution near to the obstacle
res_foil = resolution_foil * dx;

// frame
Point(1) = {0, 0, 0, res_bdy};
Point(2) = {dx, 0, 0, res_bdy};
Point(3) = {dx, dy, 0, res_bdy};
Point(4) = {0, dy, 0, res_bdy};

midx = cx;
midy = cy;

// center of circle
Point(5) = {midx, midy, 0, res_foil};

// 4 points on circle: bottom . right, top, left
Point(6) = {midx, midy-r, 0, res_foil};
Point(7) = {midx+r, midy, 0, res_foil};
Point(8) = {midx, midy+r, 0, res_foil};
Point(9) = {midx-r, midy, 0, res_foil};

// frame
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
Line Loop(2) = {5,6,7,8};

// create a surface from the outer frame with the obstacle as a hole
Plane Surface(1) = {1,2};

// Labels for the boundary conditions on the outer frame

// frame
// left 
Physical Line(11) = {4};
// right
Physical Line(12) = {2};
// top and bottom
Physical Line(13) = {1,3};


// foil
Physical Line(21) = {5,6,7,8};

// surface
Physical Surface (31) = {1};
