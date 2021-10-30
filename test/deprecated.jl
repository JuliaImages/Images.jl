# When we deprecate old APIs, we copy the old tests here to make sure they still work

@testset "deprecations" begin

    @testset "Complement" begin
        # deprecated (#690)
        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test all(complement(img) .== 1 .- img)
    end

    @testset "Restrict with vector dimensions" begin
        imgcol = colorview(RGB, rand(3,5,6))
        imgmeta = ImageMeta(imgcol, myprop=1)
        @test isa(restrict(imgmeta, [1, 2]), ImageMeta)
    end
end
