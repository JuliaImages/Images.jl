using Images
using Base.Test

@testset "Algorithms" begin
    @testset "Features" begin
        A = zeros(Int, 9, 9); A[5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test length(blobs) == 1
        blob = blobs[1]
        @test blob.amplitude ≈ 0.3183098861837907
        @test blob.σ === 1.0
        @test blob.location == CartesianIndex((5,5))
        @test blob_LoG(A, [1.0]) == blobs
        @test blob_LoG(A, [1.0], (true, false, false)) == blobs
        @test isempty(blob_LoG(A, [1.0], false))
        A = zeros(Int, 9, 9); A[1, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0,0.5,1])
        A = zeros(Int, 9, 9); A[1,5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test all(b.amplitude < 1e-16 for b in blobs)
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], true))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], (true, true, false))) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], (false, true, false)))
        blobs = blob_LoG(A, 2.0.^[0,0.5,1], (true, false, true))
        @test all(b.amplitude < 1e-16 for b in blobs)
        A = zeros(Int, 9, 9); A[[1:2;5],5]=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5)),CartesianIndex((2,5)),CartesianIndex((5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5)),CartesianIndex((5,5))]
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5]=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5,5)),CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
    end
end

nothing
