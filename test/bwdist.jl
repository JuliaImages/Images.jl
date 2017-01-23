using Base.Test, Images

@testset "bwdist" begin
    @testset "Square Images" begin
        # (1)
        A = [true false; false true]
        (F, D) = bwdist(A)
        @test F == [1 1; 1 4]
        @test D == [0 1; 1 0]

        # (2)
        A = [true true; false true]
        (F, D) = bwdist(A)
        @test F == [1 3; 1 4]
        @test D == [0 0; 1 0]

        # (3)
        A = [false false; false true]
        (F, D) = bwdist(A)
        @test F == [4 4; 4 4]
        @test isapprox(D, [1.41421 1.0; 1.0 0.0], rtol=0.01)

        # (4)
        A = [true false true; false true false; true true false]
        (F, D) = bwdist(A)
        @test F == [1 1 7; 1 5 5; 3 6 6]
        @test D == [0 1 0; 1 0 1; 0 0 1]

        # (5)
        A = [false false true; true true false; true true true]
        (F, D) = bwdist(A)
        @test F == [2 5 7; 2 5 5; 3 6 9]
        @test D == [1 1 0; 0 0 1; 0 0 0]

        # (6)
        A = [true false true true; false true false false; false true true false; true false false false]
        (F, D) = bwdist(A)
        @test F == [1 1 9 13; 1 6 6 13; 4 7 11 11; 4 4 11 11]
        @test isapprox(D, [0.0 1.0 0.0 0.0; 1.0 0.0 1.0 1.0; 1.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.41421], rtol=0.01)
    end

    @testset "Rectangular Images" begin
        # (1)
        A = [true false true; false true false]
        (F, D) = bwdist(A)
        @test F == [1 1 5; 1 4 4]
        @test D == [0 1 0; 1 0 1]

        # (2)
        A = [true false; false false; false true]
        (F, D) = bwdist(A)
        @test F == [1 1; 1 6; 6 6]
        @test D == [0 1; 1 1; 1 0]

        # (3)
        A = [true false false; true false false; false true true; true true true; false true false]
        (F, D) = bwdist(A)
        @test F == [1 1 1; 2 2 13; 2 8 13; 4 9 14; 4 10 10]
        @test D == [0.0 1.0 2.0; 0.0 1.0 1.0; 1.0 0.0 0.0; 0.0 0.0 0.0; 1.0 0.0 1.0]
    end

    @testset "Corner Case Images" begin
        # (1)
        A = [false]
        (F, D) = bwdist(A)
        @test F == [0]
        @test D == [1]

        # (2)
        A = [true]
        (F, D) = bwdist(A)
        @test F == [1]
        @test D == [0]

        # (3)
        A = [true false]
        (F, D) = bwdist(A)
        @test F == [1 1]
        @test D == [0 1]

        # (4)
        A = [false; false]
        (F, D) = bwdist(A)
        @test F == [0; 0]
        @test D == [1.0; 2.0]

        # (5)
        A = [true; true]
        (F, D) = bwdist(A)
        @test F == [1; 2]
        @test D == [0; 0]

        # (6)
        A = [true; true; false]
        (F, D) = bwdist(A)
        @test F == [1; 2; 2]
        @test D == [0; 0; 1]
    end
end
