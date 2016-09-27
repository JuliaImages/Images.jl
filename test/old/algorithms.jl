using FactCheck, Base.Test, Images, Colors, FixedPointNumbers, Compat, OffsetArrays
using Compat.view

srand(1234)

facts("Algorithms") do
    # Comparison of each element in arrays with a scalar
    approx_equal(ar, v) = @compat all(abs.(ar.-v) .< sqrt(eps(v)))
    approx_equal(ar::Images.AbstractImage, v) = approx_equal(Images.data(ar), v)

	context("Flip dimensions") do
	    A = grayim(UInt8[200 150; 50 1])
	    @fact raw(flipy(A)) --> raw(flipdim(A, 1)) "test vRs7Gx"
	    @fact raw(flipx(A)) --> raw(flipdim(A, 2)) "test omDVHe"
	end

    context("Arithmetic") do
        img = convert(Images.Image, zeros(3,3))
        img2 = (img .+ 3)/2
        @fact all(img2 .== 1.5) --> true "test mzgz4D"
        img3 = 2img2
        @fact all(img3 .== 3) --> true "test ECrt7w"
        img3 = copy(img2)
        img3[img2 .< 4] = -1
        @fact all(img3 .== -1) --> true "test 5OclVr"
        img = convert(Images.Image, rand(3,4))
        A = rand(3,4)
        img2 = img .* A
        @fact all(Images.data(img2) == Images.data(img).*A) --> true "test Txfm6a"
        img2 = convert(Images.Image, A)
        img2 = img2 .- 0.5
        img3 = 2img .* data(img2)
        img2 = img ./ A
        img2 = (2img).^2
        # Same operations with Color images
        img = Images.colorim(zeros(Float32,3,4,5))
        img2 = (img .+ RGB{Float32}(1,1,1))/2
        @fact all(img2 .== RGB{Float32}(1,1,1)/2) --> true "test yBGjhP"
        img3 = 2img2
        @fact all(img3 .== RGB{Float32}(1,1,1)) --> true "test ZaNzgJ"
        A = fill(2, 4, 5)
        @fact all(A.*img2 .== fill(RGB{Float32}(1,1,1), 4, 5)) --> true "test OT1P7V"
        img2 = img2 .- RGB{Float32}(1,1,1)/2
        A = rand(UInt8,3,4)
        img = reinterpret(Gray{UFixed8}, Images.grayim(A))
        imgm = mean(img)
        imgn = img/imgm
        @fact reinterpret(Float64, Images.data(imgn)) --> roughly(convert(Array{Float64}, A/mean(A))) "test NTAvmj"
        @fact imcomplement([Gray(0.2)]) --> [Gray(0.8)] "test Gu9b6h"
        @fact imcomplement([Gray{U8}(0.2)]) --> [Gray{U8}(0.8)] "test Kle9oP"
        @fact imcomplement([RGB(0,0.3,1)]) --> [RGB(1,0.7,0)] "test jgxQxd"
        @fact imcomplement([RGBA(0,0.3,1,0.7)]) --> [RGBA(1.0,0.7,0.0,0.7)] "test tEAo1x"
        @fact imcomplement([RGBA{U8}(0,0.6,1,0.7)]) --> [RGBA{U8}(1.0,0.4,0.0,0.7)] "test MGiodE"

        img = rand(1:10,10,10)
        img2 = rand(1:2,10,10)
        img3 = reinterpret(Gray{U8}, grayim(rand(UInt8,10,10)))
        @fact all([entropy(img, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true "test lwgD8l"
        @fact all([entropy(img2, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true "test L7wFfa"
        @fact all([entropy(img3, kind=kind) for kind in [:shannon,:nat,:hartley]] .≥ 0) --> true "test OHwUzO"
    end

    context("Reductions") do
        A = rand(5,5,3)
        img = Images.colorim(A, "RGB")
        s12 = sum(img, (1,2))
        @fact Images.colorspace(s12) --> "RGB" "test gjp91q"
        A = [NaN, 1, 2, 3]
        @fact Images.meanfinite(A, 1) --> roughly([2]) "test jy9vvu"
        A = [NaN 1 2 3;
             NaN 6 5 4]
        @test_approx_eq Images.meanfinite(A, 1) [NaN 3.5 3.5 3.5]
        @test_approx_eq Images.meanfinite(A, 2) [2, 5]'
        @test_approx_eq Images.meanfinite(A, (1,2)) [3.5]
        @fact Images.minfinite(A) --> 1 "test ctthKJ"
        @fact Images.maxfinite(A) --> 6 "test 1TSlCM"
        @fact Images.maxabsfinite(A) --> 6 "test w7j5sr"
        A = rand(10:20, 5, 5)
        @fact minfinite(A) --> minimum(A) "test k5GtIe"
        @fact maxfinite(A) --> maximum(A) "test 8qaTAa"
        A = reinterpret(UFixed8, rand(0x00:0xff, 5, 5))
        @fact minfinite(A) --> minimum(A) "test RCl2VS"
        @fact maxfinite(A) --> maximum(A) "test eKwX2u"
        A = rand(Float32,3,5,5)
        img = Images.colorim(A, "RGB")
        dc = Images.data(Images.meanfinite(img, 1))-reinterpret(RGB{Float32}, mean(A, 2), (1,5))
        @fact maximum(map(abs, dc)) --> less_than(1e-6) "test e62u7Q"
        dc = Images.minfinite(img)-RGB{Float32}(minimum(A, (2,3))...)
        @fact abs(dc) --> less_than(1e-6) "test ynXLYd"
        dc = Images.maxfinite(img)-RGB{Float32}(maximum(A, (2,3))...)
        @fact abs(dc) --> less_than(1e-6) "test Nu5oAt"

        a = convert(Array{UInt8}, [1, 1, 1])
        b = convert(Array{UInt8}, [134, 252, 4])
        @fact Images.sad(a, b) --> 387 "test sx70B8"
        @fact Images.ssd(a, b) --> 80699 "test aFz7hO"
        af = reinterpret(UFixed8, a)
        bf = reinterpret(UFixed8, b)
        @fact Images.sad(af, bf) --> roughly(387f0/255) "test R3U9a6"
        @fact Images.ssd(af, bf) --> roughly(80699f0/255^2) "test WtPNxa"
        ac = reinterpret(RGB{UFixed8}, a)
        bc = reinterpret(RGB{UFixed8}, b)
        @fact Images.sad(ac, bc) --> roughly(387f0/255) "test wtRHsd"
        @fact Images.ssd(ac, bc) --> roughly(80699f0/255^2) "test Ti1QN0"
        ag = reinterpret(RGB{UFixed8}, a)
        bg = reinterpret(RGB{UFixed8}, b)
        @fact Images.sad(ag, bg) --> roughly(387f0/255) "test jaMtWn"
        @fact Images.ssd(ag, bg) --> roughly(80699f0/255^2) "test Gc9gbr"

        a = rand(15,15)
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a rand(13,15) [1,1] 0.01) "test Pkx94Y"
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a rand(15,15) [1,1] 0.01) "test xrZo3m"
        @fact (Images.@test_approx_eq_sigma_eps a a [1,1] 0.01) --> nothing "test ccbApu"
        @fact (Images.@test_approx_eq_sigma_eps a a+0.01*rand(size(a)) [1,1] 0.01) --> nothing "test JhUXjB"
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a a+0.5*rand(size(a)) [1,1] 0.01) "test a1yPd4"
        a = colorim(rand(3,15,15))
        @fact (Images.@test_approx_eq_sigma_eps a a [1,1] 0.01) --> nothing "test 9DBljW"
        @fact_throws ErrorException (Images.@test_approx_eq_sigma_eps a colorim(rand(3,15,15)) [1,1] 0.01) "test 4LhXLK"

        a = rand(15,15)
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(13,15), [1,1], 0.01) "test vMzGQH"
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, rand(15,15), [1,1], 0.01) "test w51zmO"
        @fact Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) --> 0.0 "test 8NfaXn"
        @fact Images.test_approx_eq_sigma_eps(a, a+0.01*rand(size(a)), [1,1], 0.01) --> greater_than(0.0) "test S9POMO"
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, a+0.5*rand(size(a)), [1,1], 0.01) "test SaFtHr"
        a = colorim(rand(3,15,15))
        @fact Images.test_approx_eq_sigma_eps(a, a, [1,1], 0.01) --> 0.0 "test wtlfUO"
        @fact_throws ErrorException Images.test_approx_eq_sigma_eps(a, colorim(rand(3,15,15)), [1,1], 0.01) "test bccntj"

        @fact Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.1) --> less_than(0.1) "test j0aGQD"
        @fact_throws Images.test_approx_eq_sigma_eps(a[:,1:end-1], a[1:end-1,:], [3,3], 0.01) "test qJcPdD"

        a = zeros(10, 10)
        int_img = integral_image(a)
        @fact all(int_img == a) --> true "test UjtTze"

        a = ones(10,10)
        int_img = integral_image(a)
        chk = Array(1:10)
        @fact all([vec(int_img[i, :]) == chk * i for i in 1:10]) --> true "test TWnkpp"

        int_sum = boxdiff(int_img, 1, 1, 5, 2)
        @fact int_sum --> 10.0 "test CNwchG"
        int_sum = boxdiff(int_img, 1:5, 1:2)
        @fact int_sum --> 10.0 "test sbCuZQ"
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((5, 2)))
        @fact int_sum --> 10.0 "test UWb90L"
        int_sum = boxdiff(int_img, 1, 1, 2, 5)
        @fact int_sum --> 10.0 "test C4yK8H"
        int_sum = boxdiff(int_img, 1:2, 1:5)
        @fact int_sum --> 10.0 "test 9mTOGN"
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((2, 5)))
        @fact int_sum --> 10.0 "test 6B1IU2"
        int_sum = boxdiff(int_img, 4, 4, 8, 8)
        @fact int_sum --> 25.0 "test 4BiU7v"
        int_sum = boxdiff(int_img, 4:8, 4:8)
        @fact int_sum --> 25.0 "test GGj1cU"
        int_sum = boxdiff(int_img, CartesianIndex((4, 4)), CartesianIndex((8, 8)))
        @fact int_sum --> 25.0 "test vM8fF1"

        a = reshape(1:100, 10, 10)
        int_img = integral_image(a)
        @fact int_img[diagind(int_img)] == Array([1, 26,  108,  280,  575, 1026, 1666, 2528, 3645, 5050]) --> true "test oPHNmm"

        int_sum = boxdiff(int_img, 1, 1, 3, 3)
        @fact int_sum --> 108 "test eGx0PK"
        int_sum = boxdiff(int_img, 1:3, 1:3)
        @fact int_sum --> 108 "test KKhF8q"
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((3, 3)))
        @fact int_sum --> 108 "test K5Rnhs"
        int_sum = boxdiff(int_img, 1, 1, 5, 2)
        @fact int_sum --> 80 "test HM7SD3"
        int_sum = boxdiff(int_img, 1:5, 1:2)
        @fact int_sum --> 80 "test fBsFaz"
        int_sum = boxdiff(int_img, CartesianIndex((1, 1)), CartesianIndex((5, 2)))
        @fact int_sum --> 80 "test MhkNgv"
        int_sum = boxdiff(int_img, 4, 4, 8, 8)
        @fact int_sum --> 1400 "test 3s3WYt"
        int_sum = boxdiff(int_img, 4:8, 4:8)
        @fact int_sum --> 1400 "test uYHL4A"
        int_sum = boxdiff(int_img, CartesianIndex((4, 4)), CartesianIndex((8, 8)))
        @fact int_sum --> 1400 "test V3Bw7v"

        img = zeros(40, 40)
        img[10:30, 10:30] = 1
        pyramid = gaussian_pyramid(img, 3, 2, 1.0)
        @fact size(pyramid[1]) == (40, 40) --> true "test 7PkJim"
        @fact size(pyramid[2]) == (20, 20) --> true "test sFshCA"
        @fact size(pyramid[3]) == (10, 10) --> true "test GoaiPs"
        @fact size(pyramid[4]) == (5, 5) --> true "test 7x7aQ3"
        @fact isapprox(pyramid[1][20, 20], 1.0, atol = 0.01) --> true "test 87udZ1"
        @fact isapprox(pyramid[2][10, 10], 1.0, atol = 0.01) --> true "test HgqMZk"
        @fact isapprox(pyramid[3][5, 5], 1.0, atol = 0.05) --> true "test 98smZ2"
        @fact isapprox(pyramid[4][3, 3], 0.9, atol = 0.025) --> true "test NXNu7G"

        for p in pyramid
            h, w = size(p)
            @fact all(Bool[isapprox(v, 0, atol = 0.06) for v in p[1, :]]) --> true "test n8h7aF"
            @fact all(Bool[isapprox(v, 0, atol = 0.06) for v in p[:, 1]]) --> true "test EMA9Jn"
            @fact all(Bool[isapprox(v, 0, atol = 0.06) for v in p[h, :]]) --> true "test cG3BfH"
            @fact all(Bool[isapprox(v, 0, atol = 0.06) for v in p[:, w]]) --> true "test LcETLO"
        end
    end

    context("fft and ifft") do
        A = rand(Float32, 3, 5, 6)
        img = Images.colorim(A)
        imgfft = fft(channelview(img), 2:3)
        @fact Images.data(imgfft) --> roughly(fft(A, 2:3)) "test LlCbyo"
        img2 = ifft(imgfft, 2:3)
        @fact img2 --> roughly(A) "test D74cjO"
    end

    context("Features") do
        A = zeros(Int, 9, 9); A[5, 5] = 1
        @fact all(x->x<eps(),[blob_LoG(A, 2.0.^[0.5,0,1])[1]...] - [0.3183098861837907,sqrt(2),5,5]) --> true "test IIHdVc"
        @fact all(x->x<eps(),[blob_LoG(A, [1])[1]...] - [0.3183098861837907,sqrt(2),5,5]) --> true "test CMOJKr"
        A = zeros(Int, 9, 9); A[[1:2;5],5]=1
        @fact findlocalmaxima(A) --> [(5,5)] "test xyncRi"
        @fact findlocalmaxima(A,2) --> [(1,5),(2,5),(5,5)] "test vRMsHj"
        @fact findlocalmaxima(A,2,false) --> [(2,5),(5,5)] "test NpwSel"
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5]=1
        @fact findlocalmaxima(A) --> [(5,5,5)] "test lSv2tA"
        @fact findlocalmaxima(A,2) --> [(1,5,5),(2,5,5),(5,5,5)] "test 4jrt8N"
        @fact findlocalmaxima(A,2,false) --> [(2,5,5),(5,5,5)] "test 2awiPo"
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
        @fact B --> roughly(Btarget) "test g0lXjp"
        Argb = reinterpret(RGB, reinterpret(UFixed16, permutedims(A, (3,1,2))))
        B = Images.restrict(Argb)
        Bf = permutedims(reinterpret(eltype(eltype(B)), B), (2,3,1))
        @fact Bf --> roughly(Btarget/reinterpret(one(UFixed16)), 1e-12) "test IVByaq"
        Argba = reinterpret(RGBA{UFixed16}, reinterpret(UFixed16, A))
        B = Images.restrict(Argba)
        @fact reinterpret(eltype(eltype(B)), B) --> roughly(Images.restrict(A, (2,3))/reinterpret(one(UFixed16)), 1e-12) "test z8K24e"
        A = reshape(1:60, 5, 4, 3)
        B = Images.restrict(A, (1,2,3))
        @fact cat(3, [ 2.6015625  8.71875 6.1171875;
                       4.09375   12.875   8.78125;
                       3.5390625 10.59375 7.0546875],
                     [10.1015625 23.71875 13.6171875;
                      14.09375   32.875   18.78125;
                      11.0390625 25.59375 14.5546875]) --> roughly(B)  "test ClRQpv"
        imgcolax = AxisArray(imgcol, :y, :x)
        imgr = Images.restrict(imgcolax, (1,2))
        @fact pixelspacing(imgr) --> (2,2) "test tu0DXK"
        @fact pixelspacing(imgcolax) --> (1,1)  # issue #347 "test JR7awG"
        @inferred(restrict(imgcolax, Axis{:y}))
        @inferred(restrict(imgcolax, Axis{:x}))
        # Issue #395
        img1 = colorim(fill(0.9, 3, 5, 5))
        img2 = colorim(fill(U8(0.9), 3, 5, 5))
        @fact separate(restrict(img1)) --> roughly(separate(restrict(img2)), 0.01) "test TH8OoL"
    end

    context("Erode/ dilate") do
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ae = Images.erode(A)
        @fact Ae --> zeros(size(A)) "test kT2gnC"
        Ad = Images.dilate(A, 1:2)
        Ar = [0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0.6 0.6;
              0 0 0.6 0.6]
        @fact Ad --> cat(3, Ar, Ag, zeros(4,4)) "test AnS1W2"
        Ae = Images.erode(Ad, 1:2)
        Ar = [0.8 0.8 0 0;
              0.8 0.8 0 0;
              0 0 0 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0 0;
              0 0 0 0.6]
        @fact Ae --> cat(3, Ar, Ag, zeros(4,4)) "test vs0TRg"
        # issue #311
        @fact Images.dilate(trues(3)) --> trues(3) "test Eykrqy"
    end

    context("Extrema_filter") do
        # 2d case
        A = zeros(5,5)
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.8, 0.6] "test nQ22cy"
        # 3d case
        A = zeros(5,5,5)
        A[2,2,2] = 0.7
        A[4,4,3] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.7, 0.5] "test NXCGwQ"
        # 4d case
        A = zeros(5,5,5,5)
        A[2,2,2,2] = 0.7
        A[4,4,3,1] = 0.4
        A[3,4,3,2] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.4,0.7,0.5] "test Cw5Mbc"
        x, y, z, t = ind2sub(size(A), find(A .== 0.4))
        @fact x[1] --> 4 "test ptjTjw"
        @fact y[1] --> 4 "test KKwmrK"
        @fact z[1] --> 3 "test Pz5dnk"
        @fact t[1] --> 1 "test GFMLRG"
        # 2d case
        A = rand(5,5)/10
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, [2, 2])
        matching = falses(A)
        matching[2:end, 2:end] = maxval .== A[2:end, 2:end]
        @fact sort(A[matching])[end-1:end] --> [0.6, 0.8] "test 1SqoTD"
        # 3d case
        A = rand(5,5,5)/10
        A[2,2,2] = 0.7
        A[4,4,2] = 0.4
        A[2,2,4] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end]
        @fact sort(A[matching])[end-2:end] --> [0.4, 0.5, 0.7] "test XJKNe1"
        # 4d case
        A = rand(5,5,5,5)/10
        A[2,2,2,2] = 0.7
        A[4,4,2,3] = 0.4
        A[2,2,4,3] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end, 2:end]
        @fact sort(A[matching])[end-2:end] --> [0.4, 0.5, 0.7] "test TqtAKq"
    end

    context("Opening / closing") do
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ao = Images.opening(A)
        @fact Ao --> zeros(size(A)) "test dRBjeP"
        A = zeros(10,10)
        A[4:7,4:7] = 1
        B = copy(A)
        A[5,5] = 0
        Ac = Images.closing(A)
        @fact Ac --> B "test NFEnY3"
    end

    context("Morphological Top-hat") do
        A = zeros(13, 13)
        A[2:3, 2:3] = 1
        Ae = copy(A)
        A[5:9, 5:9] = 1
        Ao = Images.tophat(A)
        @fact Ao --> Ae "test VxhCpF"
        Aoo = Images.tophat(Ae)
        @fact Aoo --> Ae "test xA9B1t"
    end

    context("Morphological Bottom-hat") do
        A = ones(13, 13)
        A[2:3, 2:3] = 0
        Ae = 1 - copy(A)
        A[5:9, 5:9] = 0
        Ao = Images.bothat(A)
        @fact Ao --> Ae "test olSLjv"
    end

    context("Morphological Gradient") do
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morphogradient(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[6:8, 6:8] = 0
        @fact Ao --> Ae "test FjldLK"
        Aee = Images.dilate(A) - Images.erode(A)
        @fact Aee --> Ae "test PMRzxP"
    end

    context("Morphological Laplacian") do
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morpholaplace(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[5:9, 5:9] = -1
        Ae[6:8, 6:8] = 0
        @fact Ao --> Ae "test 52XRJX"
        Aee = Images.dilate(A) + Images.erode(A) - 2A
        @fact Aee --> Ae "test J1QcHc"
    end

    context("Label components") do
        A = [true  true  false true;
             true  false true  true]
        lbltarget = [1 1 0 2;
                     1 0 2 2]
        lbltarget1 = [1 2 0 4;
                      1 0 3 4]
        @fact Images.label_components(A) --> lbltarget "test JroBn0"
        @fact Images.label_components(A, [1]) --> lbltarget1 "test QJVu6m"
        connectivity = [false true  false;
                        true  false true;
                        false true  false]
        @fact Images.label_components(A, connectivity) --> lbltarget "test vz5uon"
        connectivity = trues(3,3)
        lbltarget2 = [1 1 0 1;
                      1 0 1 1]
        @fact Images.label_components(A, connectivity) --> lbltarget2 "test CFEvKh"
        @fact component_boxes(lbltarget) --> Vector{Tuple}[[(1,2),(2,3)],[(1,1),(2,2)],[(1,3),(2,4)]] "test X6qLkS"
        @fact component_lengths(lbltarget) --> [2,3,3] "test BoBrn8"
        @fact component_indices(lbltarget) --> Array{Int64}[[4,5],[1,2,3],[6,7,8]] "test iHJcSR"
        @fact component_subscripts(lbltarget) --> Array{Tuple}[[(2,2),(1,3)],[(1,1),(2,1),(1,2)],[(2,3),(1,4),(2,4)]] "test gyzoNi"
        @fact component_centroids(lbltarget) --> Tuple[(1.5,2.5),(4/3,4/3),(5/3,11/3)] "test RMvhLP"
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
        @fact norm((P-Q)[:]) --> less_than(1e-10) "test Xl6BcQ"
        P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
              0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]
        Q = Images.shepp_logan(8, highContrast=false)
        @fact norm((P-Q)[:]) --> less_than(1e-10) "test MFwM6o"
    end

    context("Image resize") do
        img = convert(Images.Image, zeros(10,10))
        img2 = Images.imresize(img, (5,5))
        @fact length(img2) --> 25 "test l5xQhW"
    end

    context("Interpolations") do

        img = zeros(Float64, 5, 5)
        @fact bilinear_interpolation(img, 4.5, 5.5) --> 0.0 "test bhkr9P"
        @fact bilinear_interpolation(img, 4.5, 3.5) --> 0.0 "test In1KLU"

        for i in [1.0, 2.0, 5.0, 7.0, 9.0]
            img = ones(Float64, 5, 5) * i
            @fact (bilinear_interpolation(img, 3.5, 4.5) == i) --> true "test YiwKrn"
            @fact (bilinear_interpolation(img, 3.2, 4) == i) --> true # X_MAX == X_MIN "test zzKRPq"
            @fact (bilinear_interpolation(img, 3.2, 4) == i) --> true # Y_MAX == Y_MIN "test RbjV3L"
            @fact (bilinear_interpolation(img, 3.2, 4) == i) --> true # BOTH EQUAL "test WVPgJQ"
            @fact (bilinear_interpolation(img, 2.8, 1.9) == i) --> true "test S4fl6z"
            # One dim out of bounds
            @fact isapprox(bilinear_interpolation(img, 0.5, 1.5), 0.5 * i) --> true "test 5vocwI"
            @fact isapprox(bilinear_interpolation(img, 0.5, 1.6), 0.5 * i) --> true "test ssXoVz"
            @fact isapprox(bilinear_interpolation(img, 0.5, 1.8), 0.5 * i) --> true "test Mg0DVi"
            # Both out of bounds (corner)
            @fact isapprox(bilinear_interpolation(img, 0.5, 0.5), 0.25 * i) --> true "test CtbuP5"
        end

        img = reshape(1.0:1.0:25.0, 5, 5)
        @fact bilinear_interpolation(img, 1.5, 2) --> 6.5 "test J81zIL"
        @fact bilinear_interpolation(img, 2, 1.5) --> 4.5 "test fqXV9r"
        @fact bilinear_interpolation(img, 2, 1) --> 2.0 "test 4T1SSb"
        @fact bilinear_interpolation(img, 1.5, 2.5) --> 9.0 "test mBjvW9"
        @fact bilinear_interpolation(img, 1.5, 3.5) --> 14.0 "test EHqnCK"
        @fact bilinear_interpolation(img, 1.5, 4.5) --> 19.0 "test FGWq37"
        @fact bilinear_interpolation(img, 1.5, 5.5) --> 10.75 "test z2ejkZ"

    end

end
