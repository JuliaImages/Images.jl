using Base.Test, Images

info("From this point on there will be deprectation warnings")

srand(2015)

@testset "Distances" begin
    @testset "Hausdorff" begin
        A = eye(3); B = copy(A); C = copy(A)
        B[1,2] = 1; C[1,3] = 1
        @test hausdorff_distance(A,A) == 0
        @test hausdorff_distance(A,B) == hausdorff_distance(B,A)
        @test hausdorff_distance(A,B) < hausdorff_distance(A,C)
        A = rand([0,1],10,10)
        B = rand([0,1],10,10)
        @test hausdorff_distance(A,B) ≥ 0
    end
end

nothing
