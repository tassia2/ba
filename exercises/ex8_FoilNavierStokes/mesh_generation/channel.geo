Include "channel_params.geo";

r = 0.5 * dx;

// position of foil
x_ref_foil1 = 0.;
y_ref_foil1 = 0.5*dy;

x_ref_foil2 = x_ref_foil1 + foil_distance*dx;
y_ref_foil2 = 0.5*dy;

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
// Foil 1
cos_a1 = Cos( -foil_rot1 );
sin_a1 = Sin( -foil_rot1 );

p11_x = x_ref_foil1 + 0.5 * scale_foil * (cos_a1 * front_x - sin_a1 * front_y ); 
p11_y = y_ref_foil1 + 0.5 * scale_foil * (sin_a1 * front_x + cos_a1 * front_y );

p12_x = x_ref_foil1 + 0.5 * scale_foil * (cos_a1 * top_x - sin_a1 * top_y ); 
p12_y = y_ref_foil1 + 0.5 * scale_foil * (sin_a1 * top_x + cos_a1 * top_y );

p13_x = x_ref_foil1 + 0.5 * scale_foil * (cos_a1 * rear_x - sin_a1 * rear_y ); 
p13_y = y_ref_foil1 + 0.5 * scale_foil * (sin_a1 * rear_x + cos_a1 * rear_y );

Point(7) = {p11_x, p11_y , 0, res_foil};
Point(8) = {p12_x, p12_y , 0, res_foil};
Point(9) = {p13_x, p13_y , 0, res_foil};

// Foil 2
cos_a2 = Cos( -foil_rot2 );
sin_a2 = Sin( -foil_rot2 );

p21_x = x_ref_foil2 + 0.5 * scale_foil * (cos_a2 * front_x - sin_a2 * front_y ); 
p21_y = y_ref_foil2 + 0.5 * scale_foil * (sin_a2 * front_x + cos_a2 * front_y );

p22_x = x_ref_foil2 + 0.5 * scale_foil * (cos_a2 * top_x - sin_a2 * top_y ); 
p22_y = y_ref_foil2 + 0.5 * scale_foil * (sin_a2 * top_x + cos_a2 * top_y );

p23_x = x_ref_foil2 + 0.5 * scale_foil * (cos_a2 * rear_x - sin_a2 * rear_y ); 
p23_y = y_ref_foil2 + 0.5 * scale_foil * (sin_a2 * rear_x + cos_a2 * rear_y );

Point(10) = {p21_x, p21_y , 0, res_foil};
Point(11) = {p22_x, p22_y , 0, res_foil};
Point(12) = {p23_x, p23_y , 0, res_foil};



// Two options for creating the obstacle:

// 2. via a Spline, guaranteed to form a closed loop
Spline(6) = {7,8,9,7};
Line Loop(2) = {6};

Spline(7) = {10,11,12,10};
Line Loop(3) = {7};

// create a surface from the outer frame with the obstacle as a hole
Plane Surface(1) = {1,2,3};

// Labels for the boundary conditions on the outer frame

// foil
Physical Line(9) = {6};
Physical Line(10) = {7};

// top and bottom
Physical Line(11) = {1,3};

// right
Physical Line(21) = {2};

// left half circle
Physical Line(22) = {4,5};

// surface
Physical Surface (33) = {1};
