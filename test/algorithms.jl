using Images, TestImages
using Statistics, Random, LinearAlgebra, FFTW
using Test, Suppressor
using ImageBase.FiniteDiff: fdiff

@testset "Algorithms" begin
    @testset "Statistics" begin
        # issue #187
        for T in (N0f8, Float32)
            A = rand(RGB{T}, 5, 4)
            Ac = channelview(A)
            s = stdmult(⊙, A)
            @test red(s) ≈ std(Ac[1,:,:])
            @test green(s) ≈ std(Ac[2,:,:])
            @test blue(s) ≈ std(Ac[3,:,:])
        end
    end

    @testset "Reductions" begin
        a = rand(15,15)
        @test_throws ErrorException (@test_approx_eq_sigma_eps a rand(13,15) [1,1] 0.01)
        @test_throws ErrorException (@test_approx_eq_sigma_eps a rand(15,15) [1,1] 0.01)
        @test (@test_approx_eq_sigma_eps a a [1,1] 0.01) == nothing
        @test (@test_approx_eq_sigma_eps a a+0.01*rand(eltype(a),size(a)) [1,1] 0.01) == nothing
        @test_throws ErrorException (@test_approx_eq_sigma_eps a a+0.5*rand(eltype(a),size(a)) [1,1] 0.01)
        a = colorview(RGB, rand(3,15,15))
        @test (@test_approx_eq_sigma_eps a a [1,1] 0.01) == nothing
        @test_throws ErrorException (@test_approx_eq_sigma_eps a colorview(RGB, rand(3,15,15)) [1,1] 0.01)

        a = rand(15,15)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(13,15), [1,1], 0.01)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(15,15), [1,1], 0.01)
        @test Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) == 0.0
        @test Images.test_approx_eq_sigma_eps(a, a+0.01*rand(eltype(a),size(a)), [1,1], 0.01) > 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, a+0.5*rand(eltype(a),size(a)), [1,1], 0.01)
        a = colorview(RGB, rand(3,15,15))
        @test Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) == 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, colorview(RGB, rand(3,15,15)), [1,1], 0.01)

        @test Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.1) < 0.1
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.01)

        img = zeros(70, 70)
        img[20:51, 20:51] .= 1
        pyramid = gaussian_pyramid(img, 3, 2, 1.0)
        @test size(pyramid[1]) == (70, 70)
        @test size(pyramid[2]) == (35, 35)
        @test size(pyramid[3]) == (18, 18)
        @test size(pyramid[4]) == (9, 9)
        @test pyramid[1][35, 35] == 1.0
        @test isapprox(pyramid[2][18, 18], 1.0, atol = 1e-5)
        @test isapprox(pyramid[3][9, 9], 1.0, atol = 1e-3)
        @test isapprox(pyramid[4][5, 5], 0.99, atol = 0.01)

        for p in pyramid
            h, w = size(p)
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[1, :]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[:, 1]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[h, :]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[:, w]])
        end

        #608
        pyramidlevel1 = gaussian_pyramid(rand(32,32), 1, 2, 1.0)
        @test length(pyramidlevel1) == 2
        @test size.(pyramidlevel1) == [(32,32), (16,16)]
    end

    @testset "gaussian_pyramid" begin
        #Tests for OffsetArrays
        img = zeros(70, 70)
        img[20:51, 20:51] .= 1
        imgo = OffsetArray(img, 0, 0)
        pyramid = gaussian_pyramid(imgo, 3, 2, 1.0)
        @test size.(axes(pyramid[1])) == ((70,), (70,))
        @test size.(axes(pyramid[2])) == ((35,), (35,))
        @test size.(axes(pyramid[3])) == ((18,), (18,))
        @test size.(axes(pyramid[4])) == ((9,), (9,))
        @test pyramid[1][35, 35] == 1.0
        @test isapprox(pyramid[2][18, 18], 1.0, atol = 1e-5)
        @test isapprox(pyramid[3][9, 9], 1.0, atol = 1e-3)
        @test isapprox(pyramid[4][5, 5], 0.99, atol = 0.01)

        for p in pyramid
            h, w = axes(p)
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[first(h), :]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[:, first(w)]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[last(h), :]])
            @test all(Bool[isapprox(v, 0, atol = 0.01) for v in p[:, last(w)]])
        end
    end

    @testset "fft and ifft" begin
        A = rand(Float32, 3, 5, 6)
        img = colorview(RGB, A)
        imgfft = fft(channelview(img), 2:3)
        @test imgfft ≈ fft(A, 2:3)
        img2 = ifft(imgfft, 2:3)
        @test img2 ≈ A
    end

    # functionality moved to ImageTransformations
    # tests are here as well to make sure everything
    # is exported properly.
    @testset "Image resize" begin
        img = zeros(10,10)
        img2 = imresize(img, (5,5))
        @test length(img2) == 25
        img = rand(RGB{Float32}, 10, 10)
        img2 = imresize(img, (6,7))
        @test size(img2) == (6,7)
        @test eltype(img2) == RGB{Float32}
    end

    @testset "Convex Hull" begin
        A = zeros(50, 30)
        A= convert(Array{Bool}, A)
        A[25,1]=1
        A[1,10]=1
        A[10,10]=1
        A[10,30]=1
        A[40,30]=1
        A[40,10]=1
        A[50,10]=1
        B = @inferred convexhull(A)
        C = CartesianIndex{}[]
        push!(C, CartesianIndex{}(25,1))
        push!(C, CartesianIndex{}(1,10))
        push!(C, CartesianIndex{}(10,30))
        push!(C, CartesianIndex{}(40,30))
        push!(C, CartesianIndex{}(50,10))
        @test typeof(B)==Array{CartesianIndex{2},1}
        @test sort(B)==sort(C)

        A = [0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 1.0, 1.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0]
        A = reshape(A, 5, 5)
        A = convert(Array{Bool}, A)
        B = B = @inferred convexhull(A)
        C = CartesianIndex{}[]
        push!(C, CartesianIndex{}(1,3))
        push!(C, CartesianIndex{}(3,1))
        push!(C, CartesianIndex{}(3,5))
        push!(C, CartesianIndex{}(5,3))
        @test typeof(B)==Array{CartesianIndex{2},1}
        @test sort(B)==sort(C)
    end

end

nothing
