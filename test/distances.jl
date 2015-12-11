using FactCheck, Base.Test, Images

srand(2015)

facts("Distances") do
  context("Hausdorff") do
    A = eye(3); B = copy(A); C = copy(A)
    B[1,2] = 1; C[1,3] = 1
    @fact MHD(A,A) == 0 --> true
    @fact MHD(A,B) == MHD(B,A) --> true
    @fact MHD(A,B) < MHD(A,C) --> true
    A = rand([0,1],10,10)
    B = rand([0,1],10,10)
    @fact MHD(A,B) ≥ 0 --> true
  end
end
