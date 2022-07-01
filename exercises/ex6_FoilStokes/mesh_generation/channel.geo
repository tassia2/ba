Include "channel_params.geo";

r = 0.5 * dx;

// position of foil
x_ref_foil = 0.;
y_ref_foil = 0.5*dy;

scale_foil = scaling_foil * dx;

// shape of foil: determined by splines through 3 points (in relative coordinates between [-1,1] x [-1,1] ), (0,0) <-> (x_ref_foil, y_ref_foil)
front_x = -1.;
front_y = 0.;

rear_x = 1.;
rear_y = 0.;

top_x = -0.75;
top_y = 0.5;

// resolution near to the outer frame
res_bdy = resolution_bdy * scale_foil;

// resolution near to the obstacle
res_foil = resolution_foil * scale_foil;


// #############################
// definition of the outer frame
//      -   x2----------x3      ^
//   .                  |       |y
//  /                   |       |
// x6 <-r-> x1          | dy    |
//  \                   |       |
//   .                  |       |
//      _   x5----------x4      0
//                dx
// ---------0------------>x

Point(1) = {0, 0.5*dy, 0, res_foil};
Point(2) = {0, dy, 0, res_bdy};
Point(3) = {dx, dy, 0, res_bdy};
Point(4) = {dx, 0, 0, res_bdy};
Point(5) = {0, 0, 0, res_bdy};
Point(6) = {-r, 0.5*dy, 0, res_bdy};


Line(1) = {2,3};
Line(2) = {3,4};
Line(3) = {4,5};
Circle(4) = {5, 1, 6};
Circle(5) = {6, 1, 2};

Line Loop(1) = {1,2,3,4,5};

//Plane Surface(1) = {1};

// definition of the obstacle
cos_a = Cos( -foil_rot );
sin_a = Sin( -foil_rot );

p1_x = x_ref_foil + 0.5 * scale_foil * (cos_a * front_x - sin_a * front_y ); 
p1_y = y_ref_foil + 0.5 * scale_foil * (sin_a * front_x + cos_a * front_y );

p2_x = x_ref_foil + 0.5 * scale_foil * (cos_a * top_x - sin_a * top_y ); 
p2_y = y_ref_foil + 0.5 * scale_foil * (sin_a * top_x + cos_a * top_y );

p3_x = x_ref_foil + 0.5 * scale_foil * (cos_a * rear_x - sin_a * rear_y ); 
p3_y = y_ref_foil + 0.5 * scale_foil * (sin_a * rear_x + cos_a * rear_y );

Point(7) = {p1_x, p1_y , 0, res_foil};
Point(8) = {p2_x, p2_y , 0, res_foil};
Point(9) = {p3_x, p3_y , 0, res_foil};

// Two options for creating the obstacle:

// 1. via straight lines
//Line(6) = {7,8};
//Line(7) = {8,9};
//Line(8) = {9,7};
//Line Loop(2) = {6,7,8};

// 2. via a Spline, guaranteed to form a closed loop
Spline(6) = {7,8,9,7};
Line Loop(2) = {6};

//Plane Surface(2) = {2};

// create a surface from the outer frame with the obstacle as a hole
Plane Surface(1) = {1,2};

// Labels for the boundary conditions on the outer frame

// foil
Physical Line(10) = {6};

// top and bottom
Physical Line(11) = {1,3};

// right
Physical Line(21) = {2};

// left half circle
Physical Line(22) = {4,5};

// surface
Physical Surface (33) = {1};
