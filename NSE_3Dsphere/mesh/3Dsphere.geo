//+
SetFactory("OpenCASCADE");
Box(1) = {0.0, 0.0, 0.0, 2.2, 0.41, 0.41};
//+
Sphere(2) = {0.2, 0.2, 0.2, 0.05, -Pi/2, Pi/2, 2*Pi};
//+
Physical Volume(1) = {1};
//+
Physical Volume(2) = {2};
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Physical Volume("domain") = {1};
//+
Physical Surface("inflow") = {8};
//+
Physical Surface("noslip") = {11, 9, 10, 12};
//+
Physical Surface("outflow") = {13};
//+
Physical Surface("sphere") = {7};
