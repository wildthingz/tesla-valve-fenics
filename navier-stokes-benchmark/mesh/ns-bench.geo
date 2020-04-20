Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 0.41, 0, 1.0};
//+
Point(3) = {2.2, 0.41, 0, 1.0};
//+
Point(4) = {2.2, 0, 0, 1.0};
//+
Point(5) = {0.15, 0.2, 0, 1.0};
//+
Point(6) = {0.2, 0.2, 0, 1.0};
//+
Point(7) = {0.25, 0.2, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Circle(5) = {5, 6, 7};
//+
Circle(6) = {7, 6, 5};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Line Loop(2) = {5, 6};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("inflow") = {1};
//+
Physical Curve("outflow") = {3};
//+
Physical Curve("wall") = {2, 4};
//+
Physical Curve("cylinder") = {6, 5};
//+
Physical Surface("fluid_domain") = {1};
