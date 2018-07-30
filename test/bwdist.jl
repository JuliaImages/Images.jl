using Test, Images

@testset "bwdist" begin
    function ind2cart(F)
        s = CartesianIndices(axes(F))
        map(i->CartesianIndex(s[i]), F)
    end
    @testset "Square Images" begin
        # (1)
        A = [true false; false true]
        F = feature_transform(A)
        @test F == ind2cart([1 1; 1 4])
        D = distance_transform(F)
        @test D == [0 1; 1 0]

        # (2)
        A = [true true; false true]
        F = feature_transform(A)
        @test F == ind2cart([1 3; 1 4])
        D = distance_transform(F)
        @test D == [0 0; 1 0]

        # (3)
        A = [false false; false true]
        F = feature_transform(A)
        @test F == ind2cart([4 4; 4 4])
        D = distance_transform(F)
        @test D ≈ [sqrt(2) 1.0; 1.0 0.0]

        # (4)
        A = [true false true; false true false; true true false]
        F = feature_transform(A)
        @test F == ind2cart([1 1 7; 1 5 5; 3 6 6])
        D = distance_transform(F)
        @test D == [0 1 0; 1 0 1; 0 0 1]

        # (5)
        A = [false false true; true true false; true true true]
        F = feature_transform(A)
        @test F == ind2cart([2 5 7; 2 5 5; 3 6 9])
        D = distance_transform(F)
        @test D == [1 1 0; 0 0 1; 0 0 0]

        # (6)
        A = [true false true true; false true false false; false true true false; true false false false]
        F = feature_transform(A)
        @test F == ind2cart([1 1 9 13; 1 6 6 13; 4 7 11 11; 4 4 11 11])
        D = distance_transform(F)
        @test D ≈ [0.0 1.0 0.0 0.0; 1.0 0.0 1.0 1.0; 1.0 0.0 0.0 1.0; 0.0 1.0 1.0 sqrt(2)]
    end

    @testset "Rectangular Images" begin
        # (1)
        A = [true false true; false true false]
        F = feature_transform(A)
        @test F == ind2cart([1 1 5; 1 4 4])
        D = distance_transform(F)
        @test D == [0 1 0; 1 0 1]

        # (2)
        A = [true false; false false; false true]
        F = feature_transform(A)
        @test F == ind2cart([1 1; 1 6; 6 6])
        D = distance_transform(F)
        @test D == [0 1; 1 1; 1 0]

        # (3)
        A = [true false false; true false false; false true true; true true true; false true false]
        F = feature_transform(A)
        @test F == ind2cart([1 1 1; 2 2 13; 2 8 13; 4 9 14; 4 10 10])
        D = distance_transform(F)
        @test D == [0.0 1.0 2.0; 0.0 1.0 1.0; 1.0 0.0 0.0; 0.0 0.0 0.0; 1.0 0.0 1.0]
    end

    @testset "Corner Case Images" begin
        null1 = CartesianIndex((typemin(Int),))
        null2 = CartesianIndex((typemin(Int),typemin(Int)))
        # (1)
        A = [false]
        F = feature_transform(A)
        @test F == [null1]
        D = distance_transform(F)
        @test D == [Inf]

        # (2)
        A = [true]
        F = feature_transform(A)
        @test F == ind2cart([1])
        D = distance_transform(F)
        @test D == [0]

        # (3)
        A = [true false]
        F = feature_transform(A)
        @test F == ind2cart([1 1])
        D = distance_transform(F)
        @test D == [0 1]

        # (4)
        A = [false; false]
        F = feature_transform(A)
        @test F == [null1; null1]
        D = distance_transform(F)
        @test D == [Inf; Inf]

        # (5)
        A = [true; true]
        F = feature_transform(A)
        @test F == ind2cart([1; 2])
        D = distance_transform(F)
        @test D == [0; 0]

        # (6)
        A = [true; true; false]
        F = feature_transform(A)
        @test F == ind2cart([1; 2; 2])
        D = distance_transform(F)
        @test D == [0; 0; 1]

        # (7)
        A = falses(3,3)
        F = feature_transform(A)
        @test all(x->x==null2, F)
        D = distance_transform(F)
        @test all(x->x==Inf, D)

        # (8)
        A = trues(4,2,3)
        F = feature_transform(A)
        @test F == ind2cart(reshape(1:length(A), size(A)))
        D = distance_transform(F)
        @test all(x->x==0, D)
    end

    @testset "Anisotropic images" begin
        A, w = [false false; false true], (3,1)
        F = feature_transform(A, w)
        @test F == ind2cart([4 4; 4 4])
        D = distance_transform(F, w)
        @test D ≈ [sqrt(10) 3.0; 1.0 0.0]

        A, w = [false false; false true], (1,3)
        F = feature_transform(A, w)
        @test F == ind2cart([4 4; 4 4])
        D = distance_transform(F, w)
        @test D ≈ [sqrt(10) 1.0; 3.0 0.0]

        A, w = [true false; false true], (3,1)
        F = feature_transform(A, w)
        @test F == ind2cart([1 1; 4 4])
        D = distance_transform(F, w)
        @test D == [0 1; 1 0]

        A, w = [true false; false true], (1,3)
        F = feature_transform(A, w)
        @test F == ind2cart([1 4; 1 4])
        D = distance_transform(F, w)
        @test D == [0 1; 1 0]
    end
end

nothing
