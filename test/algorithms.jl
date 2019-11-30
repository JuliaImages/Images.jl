using Images, Colors, FixedPointNumbers, OffsetArrays, TestImages
using Statistics, Random, LinearAlgebra, FFTW
using Test

@testset "Algorithms" begin
    @testset "Statistics" begin
        # issue #187
        for T in (N0f8, Float32)
            A = rand(RGB{T}, 5, 4)
            Ac = channelview(A)
            s = std(A)
            @test red(s) ≈ std(Ac[1,:,:])
            @test green(s) ≈ std(Ac[2,:,:])
            @test blue(s) ≈ std(Ac[3,:,:])
        end
    end

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
        A = zeros(Int, 9, 9); A[[1:2;5],5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5)),CartesianIndex((2,5)),CartesianIndex((5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5)),CartesianIndex((5,5))]
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5,5)),CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        # stub test for N-dimensional blob_LoG:
        A = zeros(Int, 9, 9, 9); A[5, 5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5, 0, 1])
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        # kinda anisotropic image
        A = zeros(Int,9,9,9); A[5,4:6,5] .= 1;
        blobs = blob_LoG(A,2 .^ [1.,0,0.5], [1.,3.,1.])
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        A = zeros(Int,9,9,9); A[1,1,4:6] .= 1;
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], true, [1.,1.,3.]))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], (true, true, true, false), [1.,1.,3.])) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], (false, true, false, false), [1.,1.,3.]))

    end

    Random.seed!(1234)

    @testset "Complement" begin
        @test complement.([Gray(0.2)]) == [Gray(0.8)]
        @test complement.([Gray{N0f8}(0.2)]) == [Gray{N0f8}(0.8)]
        @test complement.([RGB(0,0.3,1)]) == [RGB(1,0.7,0)]
        @test complement.([RGBA(0,0.3,1,0.7)]) == [RGBA(1.0,0.7,0.0,0.7)]
        @test complement.([RGBA{N0f8}(0,0.6,1,0.7)]) == [RGBA{N0f8}(1.0,0.4,0.0,0.7)]
    end

    @testset "Entropy" begin
        img = rand(1:10,10,10)
        img2 = rand(1:2,10,10)
        img3 = colorview(Gray, normedview(rand(UInt8,10,10)))
        @test all([entropy(img, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img2, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img3, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0)
    end

    @testset "Reductions" begin
        A = rand(5,5,3)
        img = colorview(RGB, permuteddimsview(A, (3,1,2)))
        s12 = sum(img, dims=(1,2))
        @test eltype(s12) <: RGB
        A = [NaN, 1, 2, 3]
        @test meanfinite(A, 1) ≈ [2]
        A = [NaN 1 2 3;
             NaN 6 5 4]
        mf = meanfinite(A, 1)
        @test isnan(mf[1])
        @test mf[1,2:end] ≈ [3.5,3.5,3.5]
        @test meanfinite(A, 2) ≈ reshape([2, 5], 2, 1)
        @test meanfinite(A, (1,2)) ≈ [3.5]
        @test minfinite(A) == 1
        @test maxfinite(A) == 6
        @test maxabsfinite(A) == 6
        A = rand(10:20, 5, 5)
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = reinterpret(N0f8, rand(0x00:0xff, 5, 5))
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = rand(Float32,3,5,5)
        img = colorview(RGB, A)
        dc = meanfinite(img, 1)-reshape(reinterpretc(RGB{Float32}, mean(A, dims=2)), (1,5))
        @test maximum(map(abs, dc)) < 1e-6
        dc = minfinite(img)-RGB{Float32}(minimum(A, dims=(2,3))...)
        @test abs(dc) < 1e-6
        dc = maxfinite(img)-RGB{Float32}(maximum(A, dims=(2,3))...)
        @test abs(dc) < 1e-6

        a = rand(15,15)
        @test_throws ErrorException (@test_approx_eq_sigma_eps a rand(13,15) [1,1] 0.01)
        @test_throws ErrorException (@test_approx_eq_sigma_eps a rand(15,15) [1,1] 0.01)
        @test (@test_approx_eq_sigma_eps a a [1,1] 0.01) == nothing
        @test (@test_approx_eq_sigma_eps a a+0.01*rand(Float64,size(a)) [1,1] 0.01) == nothing
        @test_throws ErrorException (@test_approx_eq_sigma_eps a a+0.5*rand(Float64,size(a)) [1,1] 0.01)
        a = colorview(RGB, rand(3,15,15))
        @test (@test_approx_eq_sigma_eps a a [1,1] 0.01) == nothing
        @test_throws ErrorException (@test_approx_eq_sigma_eps a colorview(RGB, rand(3,15,15)) [1,1] 0.01)

        a = rand(15,15)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(13,15), [1,1], 0.01)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(15,15), [1,1], 0.01)
        @test Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) == 0.0
        @test Images.test_approx_eq_sigma_eps(a, a+0.01*rand(Float64,size(a)), [1,1], 0.01) > 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, a+0.5*rand(Float64,size(a)), [1,1], 0.01)
        a = colorview(RGB, rand(3,15,15))
        @test Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) == 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a, colorview(RGB, rand(3,15,15)), [1,1], 0.01)

        @test Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.1) < 0.1
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.01)

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
    @testset "Restriction" begin
        imgcol = colorview(RGB, rand(3,5,6))
        A = reshape([convert(UInt16, i) for i = 1:60], 4, 5, 3)
        B = restrict(A, (1,2))
        Btarget = cat([ 0.96875  4.625   5.96875;
                           2.875   10.5    12.875;
                           1.90625  5.875   6.90625],
                      [ 8.46875  14.625 13.46875;
                        17.875    30.5   27.875;
                        9.40625  15.875 14.40625],
                      [15.96875  24.625 20.96875;
                       32.875    50.5   42.875;
                       16.90625  25.875 21.90625], dims=3)
        @test B ≈ Btarget
        Argb = reinterpretc(RGB, reinterpret(N0f16, permutedims(A, (3,1,2))))
        B = restrict(Argb)
        Bf = permutedims(reinterpretc(eltype(eltype(B)), B), (2,3,1))
        @test isapprox(Bf, Btarget/reinterpret(one(N0f16)), atol=1e-10)
        Argba = reinterpretc(RGBA{N0f16}, reinterpret(N0f16, A))
        B = restrict(Argba)
        @test isapprox(reinterpretc(eltype(eltype(B)), B), restrict(A, (2,3))/reinterpret(one(N0f16)), atol=1e-10)
        A = reshape(1:60, 5, 4, 3)
        B = restrict(A, (1,2,3))
        @test cat([ 2.6015625  8.71875 6.1171875;
                       4.09375   12.875   8.78125;
                       3.5390625 10.59375 7.0546875],
                     [10.1015625 23.71875 13.6171875;
                      14.09375   32.875   18.78125;
                      11.0390625 25.59375 14.5546875], dims=3) ≈ B
        imgcolax = AxisArray(imgcol, :y, :x)
        imgr = restrict(imgcolax, (1,2))
        @test pixelspacing(imgr) == (2,2)
        @test pixelspacing(imgcolax) == (1,1)  # issue #347
        @inferred(restrict(imgcolax, Axis{:y}))
        @inferred(restrict(imgcolax, Axis{:x}))
        restrict(imgcolax, Axis{:y})  # FIXME #628
        restrict(imgcolax, Axis{:x})
        imgmeta = ImageMeta(imgcol, myprop=1)
        @test isa(restrict(imgmeta, [1, 2]), ImageMeta)
        # Issue #395
        img1 = colorview(RGB, fill(0.9, 3, 5, 5))
        img2 = colorview(RGB, fill(N0f8(0.9), 3, 5, 5))
        @test isapprox(channelview(restrict(img1)), channelview(restrict(img2)), rtol=0.01)
        # Issue #655
        tmp = AxisArray(rand(1080,1120,5,10), (:x, :y, :z, :t), (0.577, 0.5770, 5, 2));
        @test size(restrict(tmp, 2), 2) == 561

        # restricting OffsetArrays
        A = ones(4, 5)
        Ao = OffsetArray(A, 0:3, -2:2)
        Aor = restrict(Ao)
        @test axes(Aor) == axes(OffsetArray(restrict(A), 0:2, -1:1))

        # restricting AxisArrays with offset axes
        AA = AxisArray(Ao, Axis{:y}(0:3), Axis{:x}(-2:2))
        @test axes(AA) === axes(Ao)
        AAr = restrict(AA)
        axs = axisvalues(AAr)
        @test axes(axs[1])[1] == axes(Aor)[1]
        @test axes(axs[2])[1] == axes(Aor)[2]
        AAA = AxisArray(Aor, axs)  # just test that it's constructable (another way of enforcing agreement)
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
        @test data(dilate(ImageMeta(A))) == dilate(A)
        @test data(dilate(ImageMeta(A), 1:2)) == dilate(A, 1:2)
        @test data(erode(ImageMeta(A))) == erode(A)
        @test data(erode(ImageMeta(A), 1:2)) == erode(A, 1:2)
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

    @testset "Phantoms" begin
        P = [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.2  0.3  0.3  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.2  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.0  0.2  0.0  0.0;
              0.0  0.0  0.2  0.2  0.2  0.2  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ]
        Q = shepp_logan(8)
        @test norm((P-Q)[:]) < 1e-10
        P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
              0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]
        Q = shepp_logan(8, highContrast=false)
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

    @testset "Thresholding" begin

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

    @testset "imROF" begin
        # Test that -div is the adjoint of forwarddiff
        p = rand(3,3,2)
        u = rand(3,3)
        gu = cat(Images.forwarddiffy(u), Images.forwarddiffx(u), dims=3)
        @test sum(-Images.div(p) .* u) ≈ sum(p .* gu)

        img = [0.1 0.2 0.1 0.8 0.9 0.7;
               0.2 0.1 0.1 0.8 0.1 0.8;
               0.1 0.2 0.1 0.7 0.9 0.8]
        # # Ground truth
        # using Optim
        # diff1(u) = [u[2:end,:]; u[end:end,:]] - u
        # diff2(u) = [u[:,2:end] u[:,end:end]] - u
        # obj(Avec, img, λ) = (A = reshape(Avec, size(img)); sum(abs2, A - img)/2 + λ*sum(sqrt.(diff1(A).^2 + diff2(A).^2)))
        # res = optimize(v->obj(v, img, 0.2), vec(img); iterations=10^4)
        # imgtv = reshape(res.minimizer, size(img))
        target = [fill(0.2, (3,3)) fill(0.656, (3,3))]
        @test all(map((x,y)->isapprox(x, y, atol=0.001), imROF(img, 0.2, 1000), target))
        imgc = colorview(RGB, img, img, img)
        targetc = colorview(RGB, target, target, target)
        @test all(map((x,y)->isapprox(x, y, atol=0.001), channelview(imROF(imgc, 0.2, 1000)), channelview(targetc)))
    end

    @testset "clearborder" begin
        #Case when given border width is more than image size
        img = [1 0 1 1 1 1
               0 1 1 1 0 0
               1 1 0 0 0 1
               0 1 0 1 0 1
               1 1 0 0 0 1
               0 0 1 1 0 0]
        @test_throws ArgumentError clearborder(img,7)

        #Normal Case
        img = [0 0 0 0 0 0 0 1 0
               0 0 0 0 1 0 0 0 0
               1 0 0 1 0 1 0 0 0
               0 0 1 1 1 1 1 0 0
               0 1 1 1 1 1 1 1 0
               0 0 0 0 0 0 0 0 0]
        cleared_img = clearborder(img)
        check_img = copy(img)
        check_img[3,1] = 0
        check_img[1,8] = 0
        @test cleared_img == check_img

        cleared_img = clearborder(img,2)
        @test cleared_img == fill!(similar(img), zero(eltype(img)))

        cleared_img = clearborder(img,2,10)
        @test cleared_img == 10*fill!(similar(img), one(eltype(img)))

        #Multidimentional Case
        img = cat([0 0 0 0;
                     0 0 0 0;
                     0 0 0 0;
                     1 0 0 0],
                    [0 0 0 0;
                     0 1 1 0;
                     0 0 1 0;
                     0 0 0 0],
                    [0 0 0 0;
                     0 0 0 0;
                     0 0 0 0;
                     0 0 0 0], dims=3)
        cleared_img = clearborder(img)
        check_img = copy(img)
        check_img[4,1,1] = 0
        @test cleared_img == check_img

        cleared_img = clearborder(img,2)
        @test cleared_img == fill!(similar(img), zero(eltype(img)))

        cleared_img = clearborder(img,2,10)
        @test cleared_img == 10*fill!(similar(img), one(eltype(img)))

        #Grayscale input image Case
        img = [1 2 3 1 2
               3 3 5 4 2
               3 4 5 4 2
               3 3 2 1 2]
        cleared_img = clearborder(img)
        check_img = [0 0 0 0 0
                     0 0 5 4 0
                     0 4 5 4 0
                     0 0 0 0 0]
        @test cleared_img == check_img
    end
end

nothing
