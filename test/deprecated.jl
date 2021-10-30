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

    @testset "Interpolations" begin
        img = zeros(Float64, 5, 5)
        @test bilinear_interpolation(img, 4.5, 5.5) == 0.0
        @test bilinear_interpolation(img, 4.5, 3.5) == 0.0

        for i in [1.0, 2.0, 5.0, 7.0, 9.0]
            img = ones(Float64, 5, 5) * i
            @test (bilinear_interpolation(img, 3.5, 4.5) == i)
            @test (bilinear_interpolation(img, 3.2, 4) == i)  # X_MAX == X_MIN
            @test (bilinear_interpolation(img, 3.2, 4) == i)  # Y_MAX == Y_MIN
            @test (bilinear_interpolation(img, 3.2, 4) == i)  # BOTH EQUAL
            @test (bilinear_interpolation(img, 2.8, 1.9) == i)
            # One dim out of bounds
            @test isapprox(bilinear_interpolation(img, 0.5, 1.5), 0.5 * i)
            @test isapprox(bilinear_interpolation(img, 0.5, 1.6), 0.5 * i)
            @test isapprox(bilinear_interpolation(img, 0.5, 1.8), 0.5 * i)
            # Both out of bounds (corner)
            @test isapprox(bilinear_interpolation(img, 0.5, 0.5), 0.25 * i)
        end

        img = reshape(1.0:1.0:25.0, 5, 5)
        @test bilinear_interpolation(img, 1.5, 2) == 6.5
        @test bilinear_interpolation(img, 2, 1.5) == 4.5
        @test bilinear_interpolation(img, 2, 1) == 2.0
        @test bilinear_interpolation(img, 1.5, 2.5) == 9.0
        @test bilinear_interpolation(img, 1.5, 3.5) == 14.0
        @test bilinear_interpolation(img, 1.5, 4.5) == 19.0
        @test bilinear_interpolation(img, 1.5, 5.5) == 10.75
    end
end
