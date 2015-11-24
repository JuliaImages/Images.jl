using FactCheck, Base.Test, Images, Colors, FixedPointNumbers

srand(1234)

facts("Algorithms") do
    # Comparison of each element in arrays with a scalar
    approx_equal(ar, v) = all(abs(ar.-v) .< sqrt(eps(v)))
    approx_equal(ar::Images.AbstractImage, v) = approx_equal(Images.data(ar), v)

    context("Arithmetic") do
        img = convert(Images.Image, zeros(3,3))
        img2 = (img .+ 3)/2
        @fact all(img2 .== 1.5) --> true
        img3 = 2img2
        @fact all(img3 .== 3) --> true
        img3 = copy(img2)
        img3[img2 .< 4] = -1
        @fact all(img3 .== -1) --> true
        img = convert(Images.Image, rand(3,4))
        A = rand(3,4)
        img2 = img .* A
        @fact all(Images.data(img2) == Images.data(img).*A) --> true
        img2 = convert(Images.Image, A)
        img2 = img2 .- 0.5
        img3 = 2img .* img2
        img2 = img ./ A
        img2 = (2img).^2
        # Same operations with Color images
        img = Images.colorim(zeros(Float32,3,4,5))
        img2 = (img .+ RGB{Float32}(1,1,1))/2
        @fact all(img2 .== RGB{Float32}(1,1,1)/2) --> true
        img3 = 2img2
        @fact all(img3 .== RGB{Float32}(1,1,1)) --> true
        A = fill(2, 4, 5)
        @fact all(A.*img2 .== fill(RGB{Float32}(1,1,1), 4, 5)) --> true
        img2 = img2 .- RGB{Float32}(1,1,1)/2
        A = rand(UInt8,3,4)
        img = reinterpret(Gray{UFixed8}, Images.grayim(A))
        imgm = mean(img)
        imgn = img/imgm
        @fact reinterpret(Float64, Images.data(imgn)) --> roughly(convert(Array{Float64}, A/mean(A)))
        img = rand(1:10,10,10)
        img2 = rand(1:2,10,10)
        img3 = reinterpret(Gray{U8}, grayim(rand(UInt8,10,10)))
        @fact all([entropy(img, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true
        @fact all([entropy(img2, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true
        @fact all([entropy(img3, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true
    end

    context("Reductions") do
        A = rand(5,5,3)
        img = Images.colorim(A, "RGB")
        s12 = sum(img, (1,2))
        @fact Images.colorspace(s12) --> "RGB"
        s3 = sum(img, (3,))
        @fact Images.colorspace(s3) --> "Unknown"
        A = [NaN, 1, 2, 3]
        @fact Images.meanfinite(A, 1) --> roughly([2])
        A = [NaN 1 2 3;
             NaN 6 5 4]
        @test_approx_eq Images.meanfinite(A, 1) [NaN 3.5 3.5 3.5]
        @test_approx_eq Images.meanfinite(A, 2) [2, 5]'
        @test_approx_eq Images.meanfinite(A, (1,2)) [3.5]
        @fact Images.minfinite(A) --> 1
        @fact Images.maxfinite(A) --> 6
        @fact Images.maxabsfinite(A) --> 6
        A = rand(10:20, 5, 5)
        @fact minfinite(A) --> minimum(A)
        @fact maxfinite(A) --> maximum(A)
        A = reinterpret(UFixed8, rand(0x00:0xff, 5, 5))
        @fact minfinite(A) --> minimum(A)
        @fact maxfinite(A) --> maximum(A)
        A = rand(Float32,3,5,5)
        img = Images.colorim(A, "RGB")
        dc = Images.data(Images.meanfinite(img, 1))-reinterpret(RGB{Float32}, mean(A, 2), (1,5))
        @fact maximum(map(abs, dc)) --> less_than(1e-6)
        dc = Images.minfinite(img)-RGB{Float32}(minimum(A, (2,3))...)
        @fact abs(dc) --> less_than(1e-6)
        dc = Images.maxfinite(img)-RGB{Float32}(maximum(A, (2,3))...)
        @fact abs(dc) --> less_than(1e-6)

        a = convert(Array{UInt8}, [1, 1, 1])
        b = convert(Array{UInt8}, [134, 252, 4])
        @fact Images.sad(a, b) --> 387
        @fact Images.ssd(a, b) --> 80699
        af = reinterpret(UFixed8, a)
        bf = reinterpret(UFixed8, b)
        @fact Images.sad(af, bf) --> roughly(387f0/255)
        @fact Images.ssd(af, bf) --> roughly(80699f0/255^2)
        ac = reinterpret(RGB{UFixed8}, a)
        bc = reinterpret(RGB{UFixed8}, b)
        @fact Images.sad(ac, bc) --> roughly(387f0/255)
        @fact Images.ssd(ac, bc) --> roughly(80699f0/255^2)
        ag = reinterpret(RGB{UFixed8}, a)
        bg = reinterpret(RGB{UFixed8}, b)
        @fact Images.sad(ag, bg) --> roughly(387f0/255)
        @fact Images.ssd(ag, bg) --> roughly(80699f0/255^2)

        a = rand(15,15)
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a rand(13,15) [1,1] 0.01)
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a rand(15,15) [1,1] 0.01)
        @fact (Images.@test_approx_eq_sigma_eps a a [1,1] 0.01) --> nothing
        @fact (Images.@test_approx_eq_sigma_eps a a+0.01*rand(size(a)) [1,1] 0.01) --> nothing
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a a+0.5*rand(size(a)) [1,1] 0.01)
        a = colorim(rand(3,15,15))
        @fact (Images.@test_approx_eq_sigma_eps a a [1,1] 0.01) --> nothing
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a colorim(rand(3,15,15)) [1,1] 0.01)

        a = rand(15,15)
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(13,15), [1,1], 0.01)
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(15,15), [1,1], 0.01)
        @fact Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) --> 0.0
        @fact Images.test_approx_eq_sigma_eps(a, a+0.01*rand(size(a)), [1,1], 0.01) --> greater_than(0.0)
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, a+0.5*rand(size(a)), [1,1], 0.01)
        a = colorim(rand(3,15,15))
        @fact Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) --> 0.0
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, colorim(rand(3,15,15)), [1,1], 0.01)

        @fact Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.1) --> less_than(0.1)
        @fact_throws Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.01)
    end

    context("fft and ifft") do
        A = rand(Float32, 3, 5, 6)
        img = Images.colorim(A)
        imgfft = fft(img)
        @fact Images.data(imgfft) --> roughly(fft(A, 2:3))
        @fact Images.colordim(imgfft) --> 1
        img2 = ifft(imgfft)
        @fact img2 --> roughly(reinterpret(Float32, img))
    end

    context("Array padding") do
        A = [1 2; 3 4]
        @fact Images.padindexes(A, 1, 0, 0, "replicate") --> [1,2]
        @fact Images.padindexes(A, 1, 1, 0, "replicate") --> [1,1,2]
        @fact Images.padindexes(A, 1, 0, 1, "replicate") --> [1,2,2]
        @fact Images.padarray(A, (0,0), (0,0), "replicate") --> A
        @fact Images.padarray(A, (1,2), (2,0), "replicate") --> [1 1 1 2; 1 1 1 2; 3 3 3 4; 3 3 3 4; 3 3 3 4]
        @fact Images.padindexes(A, 1, 1, 0, "circular") --> [2,1,2]
        @fact Images.padindexes(A, 1, 0, 1, "circular") --> [1,2,1]
        @fact Images.padarray(A, [2,1], [0,2], "circular") --> [2 1 2 1 2; 4 3 4 3 4; 2 1 2 1 2; 4 3 4 3 4]
        @fact Images.padindexes(A, 1, 1, 0, "symmetric") --> [1,1,2]
        @fact Images.padindexes(A, 1, 0, 1, "symmetric") --> [1,2,2]
        @fact Images.padarray(A, (1,2), (2,0), "symmetric") --> [2 1 1 2; 2 1 1 2; 4 3 3 4; 4 3 3 4; 2 1 1 2]
        @fact Images.padarray(A, (1,2), (2,0), "value", -1) --> [-1 -1 -1 -1; -1 -1 1 2; -1 -1 3 4; -1 -1 -1 -1; -1 -1 -1 -1]
        A = [1 2 3; 4 5 6]
        @fact Images.padindexes(A, 2, 1, 0, "reflect") --> [2,1,2,3]
        @fact Images.padindexes(A, 2, 0, 1, "reflect") --> [1,2,3,2]
        @fact Images.padarray(A, (1,2), (2,0), "reflect") --> [6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6]
        A = [1 2; 3 4]
        @fact Images.padarray(A, (1,1)) --> [1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
        @fact Images.padarray(A, (1,1), "replicate", "both") --> [1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
        @fact Images.padarray(A, (1,1), "circular", "pre") --> [4 3 4; 2 1 2; 4 3 4]
        @fact Images.padarray(A, (1,1), "symmetric", "post") --> [1 2 2; 3 4 4; 3 4 4]
        A = ["a" "b"; "c" "d"]
        @fact Images.padarray(A, (1,1)) --> ["a" "a" "b" "b"; "a" "a" "b" "b"; "c" "c" "d" "d"; "c" "c" "d" "d"]
        @fact_throws ErrorException Images.padindexes(A, 1, 1, 1, "unknown")
        @fact_throws ErrorException Images.padarray(A, (1,1), "unknown")
        # issue #292
        A = trues(3,3)
        @fact typeof(Images.padarray(A, (1,2), (2,1), "replicate")) --> BitArray{2}
        @fact typeof(Images.padarray(Images.grayim(A), (1,2), (2,1), "replicate")) --> BitArray{2}
    end

    context("Filtering") do
        EPS = 1e-14
        imgcol = Images.colorim(rand(3,5,6))
        imgcolf = convert(Images.Image{RGB{UFixed8}}, imgcol)
        for T in (Float64, Int)
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3,3)
            @fact maximum(abs(Images.imfilter(A, kern) - rot180(kern))) --> less_than(EPS)
            kern = rand(2,3)
            @fact maximum(abs(Images.imfilter(A, kern)[1:2,:] - rot180(kern))) --> less_than(EPS)
            kern = rand(3,2)
            @fact maximum(abs(Images.imfilter(A, kern)[:,1:2] - rot180(kern))) --> less_than(EPS)
        end
        kern = zeros(3,3); kern[2,2] = 1
        @fact maximum(map(abs, imgcol - Images.imfilter(imgcol, kern))) --> less_than(EPS)
        @fact maximum(map(abs, imgcolf - Images.imfilter(imgcolf, kern))) --> less_than(EPS)
        for T in (Float64, Int)
            # Separable kernels
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3).*rand(3)'
            @fact maximum(abs(Images.imfilter(A, kern) - rot180(kern))) --> less_than(EPS)
            kern = rand(2).*rand(3)'
            @fact maximum(abs(Images.imfilter(A, kern)[1:2,:] - rot180(kern))) --> less_than(EPS)
            kern = rand(3).*rand(2)'
            @fact maximum(abs(Images.imfilter(A, kern)[:,1:2] - rot180(kern))) --> less_than(EPS)
        end
        A = zeros(3,3); A[2,2] = 1
        kern = rand(3,3)
        @fact maximum(abs(Images.imfilter_fft(A, kern) - rot180(kern))) --> less_than(EPS)
        kern = rand(2,3)
        @fact maximum(abs(Images.imfilter_fft(A, kern)[1:2,:] - rot180(kern))) --> less_than(EPS)
        kern = rand(3,2)
        @fact maximum(abs(Images.imfilter_fft(A, kern)[:,1:2] - rot180(kern))) --> less_than(EPS)
        kern = zeros(3,3); kern[2,2] = 1
        @fact maximum(map(abs, imgcol - Images.imfilter_fft(imgcol, kern))) --> less_than(EPS)
        @fact maximum(map(abs, imgcolf - Images.imfilter_fft(imgcolf, kern))) --> less_than(EPS)

        @fact approx_equal(Images.imfilter(ones(4,4), ones(3,3)), 9.0) --> true
        @fact approx_equal(Images.imfilter(ones(3,3), ones(3,3)), 9.0) --> true
        @fact approx_equal(Images.imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0) --> true
        img = convert(Images.Image, ones(4,4))
        @fact approx_equal(Images.imfilter(img, ones(3,3)), 9.0) --> true
        A = zeros(5,5,3); A[3,3,[1,3]] = 1
        @fact Images.colordim(A) --> 3
        kern = rand(3,3)
        kernpad = zeros(5,5); kernpad[2:4,2:4] = kern
        Af = Images.imfilter(A, kern)

        @fact cat(3, rot180(kernpad), zeros(5,5), rot180(kernpad)) --> roughly(Af)
        Aimg = permutedims(convert(Images.Image, A), [3,1,2])
        @fact Images.imfilter(Aimg, kern) --> roughly(permutedims(Af, [3,1,2]))
        @fact approx_equal(Images.imfilter(ones(4,4),ones(1,3),"replicate"), 3.0) --> true

        A = zeros(5,5); A[3,3] = 1
        kern = rand(3,3)
        Af = Images.imfilter(A, kern, "inner")
        @fact Af --> rot180(kern)
        Afft = Images.imfilter_fft(A, kern, "inner")
        @fact Af --> roughly(Afft)
        h = [0.24,0.87]
        hfft = Images.imfilter_fft(eye(3), h, "inner")
        hfft[abs(hfft) .< 3eps()] = 0
        @fact Images.imfilter(eye(3), h, "inner") --> roughly(hfft)  # issue #204

        # circular
        A = zeros(3, 3)
        A[3,2] = 1
        kern = rand(3,3)
        @fact Images.imfilter_fft(A, kern, "circular") --> roughly(kern[[1,3,2],[3,2,1]])

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @fact Images.imfilter_fft(A, kern, "circular")[[1,4,5],2:4] --> roughly(kern[[1,3,2],[3,2,1]])

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @fact Images.imfilter(A, kern, "circular")[[1,4,5],2:4] --> roughly(kern[[1,3,2],[3,2,1]])

        @fact approx_equal(Images.imfilter_gaussian(ones(4,4), [5,5]), 1.0) --> true
        A = fill(convert(Float32, NaN), 3, 3)
        A[1,1] = 1
        A[2,1] = 2
        A[3,1] = 3
        @fact Images.imfilter_gaussian(A, [0,0]) --> exactly(A)
        @test_approx_eq Images.imfilter_gaussian(A, [0,3]) A
        B = copy(A)
        B[isfinite(B)] = 2
        @test_approx_eq Images.imfilter_gaussian(A, [10^3,0]) B
        @fact maximum(map(abs, Images.imfilter_gaussian(imgcol, [10^3,10^3]) - mean(imgcol))) --> less_than(1e-4)
        @fact maximum(map(abs, Images.imfilter_gaussian(imgcolf, [10^3,10^3]) - mean(imgcolf))) --> less_than(1e-4)
        A = rand(4,5)
        img = reinterpret(Images.Gray{Float64}, Images.grayim(A))
        imgf = Images.imfilter_gaussian(img, [2,2])
        @fact reinterpret(Float64, Images.data(imgf)) --> roughly(Images.imfilter_gaussian(A, [2,2]))
        A = rand(3,4,5)
        img = Images.colorim(A)
        imgf = Images.imfilter_gaussian(img, [2,2])
        @fact reinterpret(Float64, Images.data(imgf)) --> roughly(Images.imfilter_gaussian(A, [0,2,2]))

        A = zeros(Int, 9, 9); A[5, 5] = 1
        @fact maximum(abs(Images.imfilter_LoG(A, [1,1]) - Images.imlog(1.0))) --> less_than(EPS)
        @fact maximum(Images.imfilter_LoG([0 0 0 0 1 0 0 0 0], [1,1]) - sum(Images.imlog(1.0),1)) --> less_than(EPS)
        @fact maximum(Images.imfilter_LoG([0 0 0 0 1 0 0 0 0]', [1,1]) - sum(Images.imlog(1.0),2)) --> less_than(EPS)

        @fact Images.imaverage() --> fill(1/9, 3, 3)
        @fact Images.imaverage([3,3]) --> fill(1/9, 3, 3)
        @fact_throws ErrorException Images.imaverage([5])
    end

    context("Restriction") do
        imgcol = Images.colorim(rand(3,5,6))
        A = reshape([convert(UInt16, i) for i = 1:60], 4, 5, 3)
        B = Images.restrict(A, (1,2))
        Btarget = cat(3, [ 0.96875  4.625   5.96875;
                           2.875   10.5    12.875;
                           1.90625  5.875   6.90625],
                      [ 8.46875  14.625 13.46875;
                        17.875    30.5   27.875;
                        9.40625  15.875 14.40625],
                      [15.96875  24.625 20.96875;
                       32.875    50.5   42.875;
                       16.90625  25.875 21.90625])
        @fact B --> roughly(Btarget)
        Argb = reinterpret(RGB, reinterpret(UFixed16, permutedims(A, (3,1,2))))
        B = Images.restrict(Argb)
        Bf = permutedims(reinterpret(Float64, B), (2,3,1))
        @fact Bf --> roughly(Btarget/reinterpret(one(UFixed16)), 1e-12)
        Argba = reinterpret(RGBA{UFixed16}, reinterpret(UFixed16, A))
        B = Images.restrict(Argba)
        @fact reinterpret(Float64, B) --> roughly(Images.restrict(A, (2,3))/reinterpret(one(UFixed16)), 1e-12)
        A = reshape(1:60, 5, 4, 3)
        B = Images.restrict(A, (1,2,3))
        @fact cat(3, [ 2.6015625  8.71875 6.1171875;
                       4.09375   12.875   8.78125;
                       3.5390625 10.59375 7.0546875],
                     [10.1015625 23.71875 13.6171875;
                      14.09375   32.875   18.78125;
                      11.0390625 25.59375 14.5546875]) --> roughly(B)
        imgcol["pixelspacing"] = [1,1]
        imgr = Images.restrict(imgcol, (1,2))
        @fact pixelspacing(imgr) --> [2,2]
        @fact pixelspacing(imgcol) --> [1,1]  # issue #347
    end

    context("Erode/ dilate") do
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ae = Images.erode(A)
        @fact Ae --> zeros(size(A))
        Ad = Images.dilate(A)
        Ar = [0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0.6 0.6;
              0 0 0.6 0.6]
        @fact Ad --> cat(3, Ar, Ag, zeros(4,4))
        Ae = Images.erode(Ad)
        Ar = [0.8 0.8 0 0;
              0.8 0.8 0 0;
              0 0 0 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0 0;
              0 0 0 0.6]
        @fact Ae --> cat(3, Ar, Ag, zeros(4,4))
        # issue #311
        @fact Images.dilate(trues(3)) --> trues(3)
    end

    context("Opening / closing") do
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ao = Images.opening(A)
        @fact Ao --> zeros(size(A))
        A = zeros(10,10)
        A[4:7,4:7] = 1
        B = copy(A)
        A[5,5] = 0
        Ac = Images.closing(A)
        @fact Ac --> B
    end

    context("Morphological Top-hat") do
        A = zeros(13, 13)
        A[2:3, 2:3] = 1
        Ae = copy(A)
        A[5:9, 5:9] = 1
        Ao = Images.tophat(A)
        @fact Ao --> Ae
        Aoo = Images.tophat(Ae)
        @fact Aoo --> Ae
    end

    context("Morphological Bottom-hat") do
        A = ones(13, 13)
        A[2:3, 2:3] = 0
        Ae = 1 - copy(A)
        A[5:9, 5:9] = 0
        Ao = Images.bothat(A)
        @fact Ao --> Ae
    end

    context("Morphological Gradient") do
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morphogradient(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[6:8, 6:8] = 0
        @fact Ao --> Ae
        Aee = Images.dilate(A) - Images.erode(A)
        @fact Aee --> Ae
    end

    context("Morphological Laplacian") do
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morpholaplace(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[5:9, 5:9] = -1
        Ae[6:8, 6:8] = 0
        @fact Ao --> Ae
        Aee = Images.dilate(A) + Images.erode(A) - 2A
        @fact Aee --> Ae
    end

    context("Label components") do
        A = [true  true  false true;
             true  false true  true]
        lbltarget = [1 1 0 2;
                     1 0 2 2]
        lbltarget1 = [1 2 0 4;
                      1 0 3 4]
        @fact Images.label_components(A) --> lbltarget
        @fact Images.label_components(A, [1]) --> lbltarget1
        connectivity = [false true  false;
                        true  false true;
                        false true  false]
        @fact Images.label_components(A, connectivity) --> lbltarget
        connectivity = trues(3,3)
        lbltarget2 = [1 1 0 1;
                      1 0 1 1]
        @fact Images.label_components(A, connectivity) --> lbltarget2
    end

    context("Phantoms") do
        P = [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.2  0.3  0.3  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.2  0.2  0.0  0.0;
              0.0  0.0  0.2  0.0  0.0  0.2  0.0  0.0;
              0.0  0.0  0.2  0.2  0.2  0.2  0.0  0.0;
              0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ]
        Q = Images.shepp_logan(8)
        @fact norm((P-Q)[:]) --> less_than(1e-10)
        P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
              0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]
        Q = Images.shepp_logan(8, highContrast=false)
        @fact norm((P-Q)[:]) --> less_than(1e-10)
    end

    context("Image resize") do
        img = convert(Images.Image, zeros(10,10))
        img2 = Images.imresize(img, (5,5))
        @fact length(img2) --> 25
    end

context("Contrast") do
    # Issue #282
    img = convert(Images.Image{Gray{UFixed8}}, eye(2,2))
    imgs = Images.imstretch(img, 0.3, 0.4)
    @fact data(imgs) --> roughly(1./(1 + (0.3./(eye(2,2) + eps())).^0.4))
end

end
