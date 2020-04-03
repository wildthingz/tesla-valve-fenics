// Gmsh project created on Tue Mar 31 19:36:54 2020
//+
Point(1) = {0, -0.5, 0, 1.0};
//+
Point(2) = {0, 0, 0, 1.0};
//+
Point(3) = {0, 0.5, 0, 1.0};
//+
Point(5) = {-2, -2, 0, 1.0};
//+
Point(6) = {-2, 2, 0, 1.0};
//+
Point(9) = {5, -2, -0, 1.0};
//+
Point(10) = {5, 2, -0, 1.0};
//+
Circle(1) = {1, 2, 3};
//+
Circle(2) = {3, 2, 1};
//+
Line(3) = {5, 9};
//+
Line(4) = {9, 10};
//+
Line(5) = {10, 6};
//+
Line(6) = {6, 5};
//+
Line Loop(1) = {5, 6, 3, 4};
//+
Line Loop(2) = {1, 2};
//+
Plane Surface(1) = {1, 2};
//+
Physical Line("inflow") = {6};
//+
Physical Line("outflow") = {4};
//+
Physical Line("boundaries") = {5, 3};
//+
Physical Line("cylinder") = {1, 2};
//+
Physical Surface("fluid_domain") = {1};
