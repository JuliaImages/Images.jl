# These tests once lived in other files, but related functions are declared legacy or moved to
# other packages. For regression test purpose, we still keep them here.

@testset "legacy" begin

    # Moved to ColorVectorSpace
    @testset "Complement" begin
        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test complement(Gray(0.5)) == Gray(0.5)
        @test complement(Gray(0.2)) == Gray(0.8)
        @test all(complement.(img) .== 1 .- img)

        @test complement.([Gray(0.2)]) == [Gray(0.8)]
        @test complement.([Gray{N0f8}(0.2)]) == [Gray{N0f8}(0.8)]
        @test complement.([RGB(0,0.3,1)]) == [RGB(1,0.7,0)]
        @test complement.([RGBA(0,0.3,1,0.7)]) == [RGBA(1.0,0.7,0.0,0.7)]
        @test complement.([RGBA{N0f8}(0,0.6,1,0.7)]) == [RGBA{N0f8}(1.0,0.4,0.0,0.7)]
    end

end
