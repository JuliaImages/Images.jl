using Images, OffsetArrays, TestImages
using Statistics, Random, LinearAlgebra, FFTW
using Test, Suppressor
using ImageBase.FiniteDiff: fdiff

@testset "Algorithms" begin
    @testset "Statistics" begin
        # issue #187
        for T in (N0f8, Float32)
            A = rand(RGB{T}, 5, 4)
            Ac = channelview(A)
            s = @suppress_err std(A)
            @test red(s) ≈ std(Ac[1,:,:])
            @test green(s) ≈ std(Ac[2,:,:])
            @test blue(s) ≈ std(Ac[3,:,:])
        end
    end

    Random.seed!(1234)

    @testset "Entropy" begin
        img = rand(1:10,10,10)
        img2 = rand(1:2,10,10)
        img3 = colorview(Gray, normedview(rand(UInt8,10,10)))
        @test all([entropy(img, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img2, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img3, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
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

    @testset "Erode/ dilate" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ae = erode(A)
        @test Ae == zeros(size(A))
        Ad = dilate(A, 1:2)
        Ar = [0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0.6 0.6;
              0 0 0.6 0.6]
        @test Ad == cat(Ar, Ag, zeros(4,4), dims=3)
        Ae = erode(Ad, 1:2)
        Ar = [0.8 0.8 0 0;
              0.8 0.8 0 0;
              0 0 0 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0 0;
              0 0 0 0.6]
        @test Ae == cat(Ar, Ag, zeros(4,4), dims=3)
        # issue #311
        @test dilate(trues(3)) == trues(3)
        # ImageMeta
        @test arraydata(dilate(ImageMeta(A))) == dilate(A)
        @test arraydata(dilate(ImageMeta(A), 1:2)) == dilate(A, 1:2)
        @test arraydata(erode(ImageMeta(A))) == erode(A)
        @test arraydata(erode(ImageMeta(A), 1:2)) == erode(A, 1:2)
    end

    @testset "Opening / closing" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ao = opening(A)
        @test Ao == zeros(size(A))
        A = zeros(10,10)
        A[4:7,4:7] .= 1
        B = copy(A)
        A[5,5] = 0
        Ac = closing(A)
        @test Ac == B
    end

    @testset "Morphological Top-hat" begin
        A = zeros(13, 13)
        A[2:3, 2:3] .= 1
        Ae = copy(A)
        A[5:9, 5:9] .= 1
        Ao = tophat(A)
        @test Ao == Ae
        Aoo = tophat(Ae)
        @test Aoo == Ae
    end

    @testset "Morphological Bottom-hat" begin
        A = ones(13, 13)
        A[2:3, 2:3] .= 0
        Ae = 1 .- copy(A)
        A[5:9, 5:9] .= 0
        Ao = bothat(A)
        @test Ao == Ae
    end

    @testset "Morphological Gradient" begin
        A = zeros(13, 13)
        A[5:9, 5:9] .= 1
        Ao = morphogradient(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] .= 1
        Ae[6:8, 6:8] .= 0
        @test Ao == Ae
        Aee = dilate(A) - erode(A)
        @test Aee == Ae
    end

    @testset "Morphological Laplacian" begin
        A = zeros(13, 13)
        A[5:9, 5:9] .= 1
        Ao = morpholaplace(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] .= 1
        Ae[5:9, 5:9] .= -1
        Ae[6:8, 6:8] .= 0
        @test Ao == Ae
        Aee = dilate(A) + erode(A) - 2A
        @test Aee == Ae
    end

    @testset "Label components" begin
        A = [true  true  false true;
             true  false true  true]
        lbltarget = [1 1 0 2;
                     1 0 2 2]
        lbltarget1 = [1 2 0 4;
                      1 0 3 4]
        @test label_components(A) == lbltarget
        @test label_components(A, [1]) == lbltarget1
        connectivity = [false true  false;
                        true  false true;
                        false true  false]
        @test label_components(A, connectivity) == lbltarget
        connectivity = trues(3,3)
        lbltarget2 = [1 1 0 1;
                      1 0 1 1]
        @test label_components(A, connectivity) == lbltarget2
        @test component_boxes(lbltarget) == Vector{Tuple}[[(1,2),(2,3)],[(1,1),(2,2)],[(1,3),(2,4)]]
        @test component_lengths(lbltarget) == [2,3,3]
        @test component_indices(lbltarget) == Array{Int64}[[4,5],[1,2,3],[6,7,8]]
        @test component_subscripts(lbltarget) == Array{Tuple}[[(2,2),(1,3)],[(1,1),(2,1),(1,2)],[(2,3),(1,4),(2,4)]]
        @test @inferred(component_centroids(lbltarget)) == Tuple[(1.5,2.5),(4/3,4/3),(5/3,11/3)]
    end

    # deprecated
    @suppress_err @testset "Phantoms" begin
        P = [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.2  0.3  0.3  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.2  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.0  0.2  0.0  0.0;
              0.0  0.0  0.2  0.2  0.2  0.2  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ]
        Q = Images.shepp_logan(8)
        @test norm((P-Q)[:]) < 1e-10
        P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
              0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]
        Q = Images.shepp_logan(8, highContrast=false)
        @test norm((P-Q)[:]) < 1e-10
    end

    # functionality moved to ImageTransformations
    # tests are here as well to make sure everything
    # is exported properly.
    @testset "Image resize" begin
        img = zeros(10,10)
        img2 = Images.imresize(img, (5,5))
        @test length(img2) == 25
        img = rand(RGB{Float32}, 10, 10)
        img2 = Images.imresize(img, (6,7))
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

    # deprecated
    @suppress_err @testset "Thresholding" begin

        #otsu_threshold
        img = testimage("cameraman")
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 87/256), atol=eps(N0f8))
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 87/256), atol=eps(N0f8))

        img = map(x->convert(Gray{Float64}, x), img)
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)

        img = map(x->convert(Float64, x), img)
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)

        #test for multidimension arrays
        img = rand(Float64, 10, 10, 3)
        @test otsu_threshold(img) == otsu_threshold(cat(img[:,:,1], img[:,:,2], img[:,:,3], dims=1))

        #yen_threshold
        img = testimage("cameraman")
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 199/256), atol=eps(N0f8))
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 199/256), atol=eps(N0f8))

        img = map(x->convert(Gray{Float64}, x), img)
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)

        img = map(x->convert(Float64, x), img)
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)

        img = rand(Float64, 10, 10, 3)
        @test yen_threshold(img) == yen_threshold(cat(img[:,:,1], img[:,:,2], img[:,:,3], dims=1))

        img = zeros(Gray{Float64},10,10,3)
        @test yen_threshold(img) == 0
    end
end

nothing
