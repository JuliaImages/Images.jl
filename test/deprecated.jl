# When we deprecate old APIs, we copy the old tests here to make sure they still work

@testset "deprecations" begin

    @testset "Complement" begin
        # deprecated (#690)
        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test all(complement(img) .== 1 .- img)
    end

    @testset "Stats" begin
        a = [0.1, 0.2, 0.1]
        @test var(Gray.(a)) == Gray(var(a))
        @test std(Gray.(a)) == Gray(std(a))
        an0f8 = N0f8.(a)
        @test var(Gray.(an0f8)) == Gray(var(an0f8))
        @test std(Gray.(an0f8)) == Gray(std(an0f8))
        c = [RGB(1, 0, 0), RGB(0, 0, 1)]
        @test var(c) == varmult(⊙, c)
        @test std(c) ≈ mapc(sqrt, varmult(⊙, c))
        # We don't want to support this next one at all:
        chsv = HSV.(c)
        @test var(chsv) == HSV{Float32}(28800.0f0, 0.0f0, 0.0f0)
        @test std(chsv) == mapc(sqrt, HSV{Float32}(28800.0f0, 0.0f0, 0.0f0))
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

    @testset "boxdiff" begin
        a = zeros(10, 10)
        int_img = integral_image(a)
        @test all(int_img == a)

        a = ones(10,10)
        int_img = integral_image(a)
        chk = Array(1:10)
        @test all([vec(int_img[i, :]) == chk * i for i in 1:10])

        int_sum = boxdiff(int_img, 1, 1, 5, 2)
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, 1:5, 1:2)
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((5, 2)))
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, 1, 1, 2, 5)
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, 1:2, 1:5)
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((2, 5)))
        @test int_sum == 10.0
        int_sum = boxdiff(int_img, 4, 4, 8, 8)
        @test int_sum == 25.0
        int_sum = boxdiff(int_img, 4:8, 4:8)
        @test int_sum == 25.0
        int_sum = boxdiff(int_img, CartesianIndex((4, 4)), CartesianIndex((8, 8)))
        @test int_sum == 25.0

        a = reshape(1:100, 10, 10)
        int_img = integral_image(a)
        @test int_img[diagind(int_img)] == Array([1, 26,  108,  280,  575, 1026, 1666, 2528, 3645, 5050])

        int_sum = boxdiff(int_img, 1, 1, 3, 3)
        @test int_sum == 108
        int_sum = boxdiff(int_img, 1:3, 1:3)
        @test int_sum == 108
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((3, 3)))
        @test int_sum == 108
        int_sum = boxdiff(int_img, 1, 1, 5, 2)
        @test int_sum == 80
        int_sum = boxdiff(int_img, 1:5, 1:2)
        @test int_sum == 80
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((5, 2)))
        @test int_sum == 80
        int_sum = boxdiff(int_img, 4, 4, 8, 8)
        @test int_sum == 1400
        int_sum = boxdiff(int_img, 4:8, 4:8)
        @test int_sum == 1400
        int_sum = boxdiff(int_img, CartesianIndex((4, 4)), CartesianIndex((8, 8)))
        @test int_sum == 1400
    end
end
