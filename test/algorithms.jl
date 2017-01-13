using Images, Colors, FixedPointNumbers
using Compat.view

srand(1234)

@testset "Algorithms" begin
    # Comparison of each element in arrays with a scalar
    approx_equal(ar, v) = @compat all(abs.(ar.-v) .< sqrt(eps(v)))
    approx_equal(ar::Images.AbstractImage, v) = approx_equal(Images.data(ar), v)

	@testset "Flip dimensions" begin
		A = UInt8[200 150; 50 1]
		img_x = grayim(A)
		img_y = permutedims(img_x, [2, 1])

		@test raw(flipdim(img_x,"x")) == raw(flipdim(img_x,1))
		@test raw(flipdim(img_x,"x")) == flipdim(A,1)
		@test raw(flipdim(img_y,"x")) == raw(flipdim(img_y,2))
		@test raw(flipdim(img_y,"x")) == flipdim(A',2)

		@test raw(flipdim(img_x,"y")) == raw(flipdim(img_x,2))
		@test raw(flipdim(img_x,"y")) == flipdim(A,2)
		@test raw(flipdim(img_y,"y")) == raw(flipdim(img_y,1))
		@test raw(flipdim(img_y,"y")) == flipdim(A',1)

		@test raw(flipx(img_x)) == raw(flipdim(img_x,"x"))
		@test raw(flipx(img_y)) == raw(flipdim(img_y,"x"))

		@test raw(flipy(img_x)) == raw(flipdim(img_x,"y"))
		@test raw(flipy(img_y)) == raw(flipdim(img_y,"y"))
	end

    @testset "Arithmetic" begin
        img = convert(Images.Image, zeros(3,3))
        img2 = (img .+ 3)/2
        @test all(img2 .== 1.5)
        img3 = 2img2
        @test all(img3 .== 3)
        img3 = copy(img2)
        img3[img2 .< 4] = -1
        @test all(img3 .== -1)
        img = convert(Images.Image, rand(3,4))
        A = rand(3,4)
        img2 = img .* A
        @test all(Images.data(img2) == Images.data(img) .* A)
        img2 = convert(Images.Image, A)
        img2 = img2 .- 0.5
        img3 = 2img .* img2
        img2 = img ./ A
        img2 = (2img).^2
        # Same operations with Color images
        img = Images.colorim(zeros(Float32,3,4,5))
        img2 = (img .+ RGB{Float32}(1,1,1))/2
        @test all(img2 .== RGB{Float32}(1,1,1) / 2)
        img3 = 2img2
        @test all(img3 .== RGB{Float32}(1,1,1))
        A = fill(2, 4, 5)
        @test all(A .* img2 .== fill(RGB{Float32}(1,1,1),4,5))
        img2 = img2 .- RGB{Float32}(1,1,1)/2
        A = rand(UInt8,3,4)
        img = reinterpret(Gray{UFixed8}, Images.grayim(A))
        imgm = mean(img)
        imgn = img/imgm
        @test reinterpret(Float64,Images.data(imgn)) ≈ convert(Array{Float64},A / mean(A))
        @test imcomplement([Gray(0.2)]) == [Gray(0.8)]
        @test imcomplement([Gray{U8}(0.2)]) == [Gray{U8}(0.8)]
        @test imcomplement([RGB(0,0.3,1)]) == [RGB(1,0.7,0)]
        @test imcomplement([RGBA(0,0.3,1,0.7)]) == [RGBA(1.0,0.7,0.0,0.7)]
        @test imcomplement([RGBA{U8}(0,0.6,1,0.7)]) == [RGBA{U8}(1.0,0.4,0.0,0.7)]

        img = rand(1:10,10,10)
        img2 = rand(1:2,10,10)
        img3 = reinterpret(Gray{U8}, grayim(rand(UInt8,10,10)))
        @test all([entropy(img,kind=kind) for kind = [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img2,kind=kind) for kind = [:shannon,:nat,:hartley]] .≥ 0)
        @test all([entropy(img3,kind=kind) for kind = [:shannon,:nat,:hartley]] .≥ 0)
    end

    @testset "Reductions" begin
        A = rand(5,5,3)
        img = Images.colorim(A, "RGB")
        s12 = sum(img, (1,2))
        @test Images.colorspace(s12) == "RGB"
        s3 = sum(img, (3,))
        @test Images.colorspace(s3) == "Unknown"
        A = [NaN, 1, 2, 3]
        @test Images.meanfinite(A,1) ≈ [2]
        A = [NaN 1 2 3;
             NaN 6 5 4]
        @test_approx_eq Images.meanfinite(A, 1) [NaN 3.5 3.5 3.5]
        @test_approx_eq Images.meanfinite(A, 2) [2, 5]'
        @test_approx_eq Images.meanfinite(A, (1,2)) [3.5]
        @test Images.minfinite(A) == 1
        @test Images.maxfinite(A) == 6
        @test Images.maxabsfinite(A) == 6
        A = rand(10:20, 5, 5)
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = reinterpret(UFixed8, rand(0x00:0xff, 5, 5))
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = rand(Float32,3,5,5)
        img = Images.colorim(A, "RGB")
        dc = Images.data(Images.meanfinite(img, 1))-reinterpret(RGB{Float32}, mean(A, 2), (1,5))
        @test maximum(map(abs,dc)) < 1.0e-6
        dc = Images.minfinite(img)-RGB{Float32}(minimum(A, (2,3))...)
        @test abs(dc) < 1.0e-6
        dc = Images.maxfinite(img)-RGB{Float32}(maximum(A, (2,3))...)
        @test abs(dc) < 1.0e-6

        a = convert(Array{UInt8}, [1, 1, 1])
        b = convert(Array{UInt8}, [134, 252, 4])
        @test Images.sad(a,b) == 387
        @test Images.ssd(a,b) == 80699
        af = reinterpret(UFixed8, a)
        bf = reinterpret(UFixed8, b)
        @test Images.sad(af,bf) ≈ 387.0f0 / 255
        @test Images.ssd(af,bf) ≈ 80699.0f0 / 255 ^ 2
        ac = reinterpret(RGB{UFixed8}, a)
        bc = reinterpret(RGB{UFixed8}, b)
        @test Images.sad(ac,bc) ≈ 387.0f0 / 255
        @test Images.ssd(ac,bc) ≈ 80699.0f0 / 255 ^ 2
        ag = reinterpret(RGB{UFixed8}, a)
        bg = reinterpret(RGB{UFixed8}, b)
        @test Images.sad(ag,bg) ≈ 387.0f0 / 255
        @test Images.ssd(ag,bg) ≈ 80699.0f0 / 255 ^ 2

        a = rand(15,15)
        @test_throws ErrorException Images.@test_approx_eq_sigma_eps(a,rand(13,15),[1,1],0.01)
        @test_throws ErrorException Images.@test_approx_eq_sigma_eps(a,rand(15,15),[1,1],0.01)
        @test Images.@test_approx_eq_sigma_eps(a,a,[1,1],0.01) == nothing
        @test Images.@test_approx_eq_sigma_eps(a,a + 0.01 * rand(size(a)),[1,1],0.01) == nothing
        @test_throws ErrorException Images.@test_approx_eq_sigma_eps(a,a + 0.5 * rand(size(a)),[1,1],0.01)
        a = colorim(rand(3,15,15))
        @test Images.@test_approx_eq_sigma_eps(a,a,[1,1],0.01) == nothing
        @test_throws ErrorException Images.@test_approx_eq_sigma_eps(a,colorim(rand(3,15,15)),[1,1],0.01)

        a = rand(15,15)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a,rand(13,15),[1,1],0.01)
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a,rand(15,15),[1,1],0.01)
        @test Images.test_approx_eq_sigma_eps(a,a,[1,1],0.01) == 0.0
        @test Images.test_approx_eq_sigma_eps(a,a + 0.01 * rand(size(a)),[1,1],0.01) > 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a,a + 0.5 * rand(size(a)),[1,1],0.01)
        a = colorim(rand(3,15,15))
        @test Images.test_approx_eq_sigma_eps(a,a,[1,1],0.01) == 0.0
        @test_throws ErrorException Images.test_approx_eq_sigma_eps(a,colorim(rand(3,15,15)),[1,1],0.01)

        @test Images.test_approx_eq_sigma_eps(a[:,1:end - 1],a[1:end - 1,:],[3,3],0.1) < 0.1
        @test_throws Exception Images.test_approx_eq_sigma_eps(a[:,1:end - 1],a[1:end - 1,:],[3,3],0.01)

        a = zeros(10, 10)
        int_img = integral_image(a)
        @test all(int_img == a)

        a = ones(10,10)
        int_img = integral_image(a)
        chk = Array(1:10)
        @test all([vec(int_img[i,:]) == chk * i for i = 1:10])

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
        @test int_img[diagind(int_img)] == Array([1,26,108,280,575,1026,1666,2528,3645,5050])

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

        img = zeros(40, 40)
        img[10:30, 10:30] = 1
        pyramid = gaussian_pyramid(img, 3, 2, 1.0)
        @test size(pyramid[1]) == (40,40)
        @test size(pyramid[2]) == (20,20)
        @test size(pyramid[3]) == (10,10)
        @test size(pyramid[4]) == (5,5)
        @test isapprox((pyramid[1])[20,20],1.0,atol=0.01)
        @test isapprox((pyramid[2])[10,10],1.0,atol=0.01)
        @test isapprox((pyramid[3])[5,5],1.0,atol=0.05)
        @test isapprox((pyramid[4])[3,3],0.9,atol=0.025)

        for p in pyramid
            h, w = size(p)
            @test all(Bool[isapprox(v,0,atol=0.06) for v = p[1,:]])
            @test all(Bool[isapprox(v,0,atol=0.06) for v = p[:,1]])
            @test all(Bool[isapprox(v,0,atol=0.06) for v = p[h,:]])
            @test all(Bool[isapprox(v,0,atol=0.06) for v = p[:,w]])
        end
    end

    @testset "fft and ifft" begin
        A = rand(Float32, 3, 5, 6)
        img = Images.colorim(A)
        imgfft = fft(img)
        @test Images.data(imgfft) ≈ fft(A,2:3)
        @test Images.colordim(imgfft) == 1
        img2 = ifft(imgfft)
        @test img2 ≈ reinterpret(Float32,img)
    end

    @testset "Array padding" begin
        A = [1 2; 3 4]
        @test Images.padindexes(A,1,0,0,"replicate") == [1,2]
        @test Images.padindexes(A,1,1,0,"replicate") == [1,1,2]
        @test Images.padindexes(A,1,0,1,"replicate") == [1,2,2]
        @test Images.padarray(A,(0,0),(0,0),"replicate") == A
        @test Images.padarray(A,(1,2),(2,0),"replicate") == [1 1 1 2;1 1 1 2;3 3 3 4;3 3 3 4;3 3 3 4]
        @test Images.padindexes(A,1,1,0,"circular") == [2,1,2]
        @test Images.padindexes(A,1,0,1,"circular") == [1,2,1]
        @test Images.padarray(A,[2,1],[0,2],"circular") == [2 1 2 1 2;4 3 4 3 4;2 1 2 1 2;4 3 4 3 4]
        @test Images.padindexes(A,1,1,0,"symmetric") == [1,1,2]
        @test Images.padindexes(A,1,0,1,"symmetric") == [1,2,2]
        @test Images.padarray(A,(1,2),(2,0),"symmetric") == [2 1 1 2;2 1 1 2;4 3 3 4;4 3 3 4;2 1 1 2]
        @test Images.padarray(A,(1,2),(2,0),"value",-1) == [-1 -1 -1 -1;-1 -1 1 2;-1 -1 3 4;-1 -1 -1 -1;-1 -1 -1 -1]
        A = [1 2 3; 4 5 6]
        @test Images.padindexes(A,2,1,0,"reflect") == [2,1,2,3]
        @test Images.padindexes(A,2,0,1,"reflect") == [1,2,3,2]
        @test Images.padarray(A,(1,2),(2,0),"reflect") == [6 5 4 5 6;3 2 1 2 3;6 5 4 5 6;3 2 1 2 3;6 5 4 5 6]
        A = [1 2; 3 4]
        @test Images.padarray(A,(1,1)) == [1 1 2 2;1 1 2 2;3 3 4 4;3 3 4 4]
        @test Images.padarray(A,(1,1),"replicate","both") == [1 1 2 2;1 1 2 2;3 3 4 4;3 3 4 4]
        @test Images.padarray(A,(1,1),"circular","pre") == [4 3 4;2 1 2;4 3 4]
        @test Images.padarray(A,(1,1),"symmetric","post") == [1 2 2;3 4 4;3 4 4]
        A = ["a" "b"; "c" "d"]
        @test Images.padarray(A,(1,1)) == ["a" "a" "b" "b";"a" "a" "b" "b";"c" "c" "d" "d";"c" "c" "d" "d"]
        @test_throws ErrorException Images.padindexes(A,1,1,1,"unknown")
        @test_throws ErrorException Images.padarray(A,(1,1),"unknown")
        # issue #292
        A = trues(3,3)
        @test typeof(Images.padarray(A,(1,2),(2,1),"replicate")) == BitArray{2}
        @test typeof(Images.padarray(Images.grayim(A),(1,2),(2,1),"replicate")) == BitArray{2}
        # issue #525
        A = falses(10,10,10)
        B = view(A,1:8,1:8,1:8)
        @test isa(padarray(A,ones(Int,3),ones(Int,3),"replicate"),BitArray{3})
        @test isa(padarray(B,ones(Int,3),ones(Int,3),"replicate"),BitArray{3})
    end

    @compat @testset "Filtering" begin
        EPS = 10*eps(float(UFixed8(1)))
        imgcol = Images.colorim(rand(3,5,6))
        imgcolf = convert(Images.Image{RGB{UFixed8}}, imgcol)
        for T in (Float64, Int)
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3,3)
            @test maximum(abs.(Images.imfilter(A,kern) - rot180(kern))) < EPS
            kern = rand(2,3)
            @test maximum(abs.((Images.imfilter(A,kern))[1:2,:] - rot180(kern))) < EPS
            kern = rand(3,2)
            @test maximum(abs.((Images.imfilter(A,kern))[:,1:2] - rot180(kern))) < EPS
        end
        kern = zeros(3,3); kern[2,2] = 1
        @test maximum(map(abs,imgcol - Images.imfilter(imgcol,kern))) < EPS
        @test maximum(map(abs,imgcolf - Images.imfilter(imgcolf,kern))) < EPS
        for T in (Float64, Int)
            # Separable kernels
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3).*rand(3)'
            @test maximum(abs.(Images.imfilter(A,kern) - rot180(kern))) < EPS
            kern = rand(2).*rand(3)'
            @test maximum(abs.((Images.imfilter(A,kern))[1:2,:] - rot180(kern))) < EPS
            kern = rand(3).*rand(2)'
            @test maximum(abs.((Images.imfilter(A,kern))[:,1:2] - rot180(kern))) < EPS
        end
        A = zeros(3,3); A[2,2] = 1
        kern = rand(3,3)
        @test maximum(abs.(Images.imfilter_fft(A,kern) - rot180(kern))) < EPS
        kern = rand(2,3)
        @test maximum(abs.((Images.imfilter_fft(A,kern))[1:2,:] - rot180(kern))) < EPS
        kern = rand(3,2)
        @test maximum(abs.((Images.imfilter_fft(A,kern))[:,1:2] - rot180(kern))) < EPS
        kern = zeros(3,3); kern[2,2] = 1
        @test maximum(map(abs,imgcol - Images.imfilter_fft(imgcol,kern))) < EPS
        @test maximum(map(abs,imgcolf - Images.imfilter_fft(imgcolf,kern))) < EPS

        @test approx_equal(Images.imfilter(ones(4,4),ones(3,3)),9.0)
        @test approx_equal(Images.imfilter(ones(3,3),ones(3,3)),9.0)
        @test approx_equal(Images.imfilter(ones(3,3),[1 1 1;1 0.0 1;1 1 1]),8.0)
        img = convert(Images.Image, ones(4,4))
        @test approx_equal(Images.imfilter(img,ones(3,3)),9.0)
        A = zeros(5,5,3); A[3,3,[1,3]] = 1
        @test Images.colordim(A) == 3
        kern = rand(3,3)
        kernpad = zeros(5,5); kernpad[2:4,2:4] = kern
        Af = Images.imfilter(A, kern)

        @test cat(3,rot180(kernpad),zeros(5,5),rot180(kernpad)) ≈ Af
        Aimg = permutedims(convert(Images.Image, A), [3,1,2])
        @test Images.imfilter(Aimg,kern) ≈ permutedims(Af,[3,1,2])
        @test approx_equal(Images.imfilter(ones(4,4),ones(1,3),"replicate"),3.0)

        A = zeros(5,5); A[3,3] = 1
        kern = rand(3,3)
        Af = Images.imfilter(A, kern, "inner")
        @test Af == rot180(kern)
        Afft = Images.imfilter_fft(A, kern, "inner")
        @test Af ≈ Afft
        h = [0.24,0.87]
        hfft = Images.imfilter_fft(eye(3), h, "inner")
        hfft[abs.(hfft) .< 3eps()] = 0
        @test Images.imfilter(eye(3),h,"inner") ≈ hfft # issue #204

        # circular
        A = zeros(3, 3)
        A[3,2] = 1
        kern = rand(3,3)
        @test Images.imfilter_fft(A,kern,"circular") ≈ kern[[1,3,2],[3,2,1]]

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @test (Images.imfilter_fft(A,kern,"circular"))[[1,4,5],2:4] ≈ kern[[1,3,2],[3,2,1]]

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @test (Images.imfilter(A,kern,"circular"))[[1,4,5],2:4] ≈ kern[[1,3,2],[3,2,1]]

        @test approx_equal(Images.imfilter_gaussian(ones(4,4),[5,5]),1.0)
        A = fill(convert(Float32, NaN), 3, 3)
        A[1,1] = 1
        A[2,1] = 2
        A[3,1] = 3
        @test Images.imfilter_gaussian(A,[0,0]) === A
        @test_approx_eq Images.imfilter_gaussian(A, [0,3]) A
        B = copy(A)
        B[isfinite.(B)] = 2
        @test_approx_eq Images.imfilter_gaussian(A, [10^3,0]) B
        @test maximum(map(abs,Images.imfilter_gaussian(imgcol,[10 ^ 3,10 ^ 3]) - mean(imgcol))) < 0.0001
        @test maximum(map(abs,Images.imfilter_gaussian(imgcolf,[10 ^ 3,10 ^ 3]) - mean(imgcolf))) < 0.0001
        A = rand(4,5)
        img = reinterpret(Images.Gray{Float64}, Images.grayim(A))
        imgf = Images.imfilter_gaussian(img, [2,2])
        @test reinterpret(Float64,Images.data(imgf)) ≈ Images.imfilter_gaussian(A,[2,2])
        A = rand(3,4,5)
        img = Images.colorim(A)
        imgf = Images.imfilter_gaussian(img, [2,2])
        @test reinterpret(Float64,Images.data(imgf)) ≈ Images.imfilter_gaussian(A,[0,2,2])

        A = zeros(Int, 9, 9); A[5, 5] = 1
        @test maximum(abs.(Images.imfilter_LoG(A,[1,1]) - Images.imlog(1.0))) < EPS
        @test maximum(Images.imfilter_LoG([0 0 0 0 1 0 0 0 0],[1,1]) - sum(Images.imlog(1.0),1)) < EPS
        @test maximum(Images.imfilter_LoG(([0 0 0 0 1 0 0 0 0])',[1,1]) - sum(Images.imlog(1.0),2)) < EPS

        @test Images.imaverage() == fill(1 / 9,3,3)
        @test Images.imaverage([3,3]) == fill(1 / 9,3,3)
        @test_throws ErrorException Images.imaverage([5])
    end

    @testset "Features" begin
        A = zeros(Int, 9, 9); A[5, 5] = 1
        @test all((x->begin
                x < eps()
            end),[(blob_LoG(A,2.0 .^ [0.5,0,1]))[1]...] - [0.3183098861837907,sqrt(2),5,5])
        @test all((x->begin
                x < eps()
            end),[(blob_LoG(A,[1]))[1]...] - [0.3183098861837907,sqrt(2),5,5])
        A = zeros(Int, 9, 9); A[[1:2;5],5]=1
        @test findlocalmaxima(A) == [(5,5)]
        @test findlocalmaxima(A,2) == [(1,5),(2,5),(5,5)]
        @test findlocalmaxima(A,2,false) == [(2,5),(5,5)]
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5]=1
        @test findlocalmaxima(A) == [(5,5,5)]
        @test findlocalmaxima(A,2) == [(1,5,5),(2,5,5),(5,5,5)]
        @test findlocalmaxima(A,2,false) == [(2,5,5),(5,5,5)]
    end

    @testset "Restriction" begin
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
        @test B ≈ Btarget
        Argb = reinterpret(RGB, reinterpret(UFixed16, permutedims(A, (3,1,2))))
        B = Images.restrict(Argb)
        Tr = eltype(eltype(B))
        Bf = permutedims(reinterpret(Tr, B), (2,3,1))
        @test isapprox(Bf,Btarget / reinterpret(one(UFixed16)),atol=eps(Tr) ^ (2 / 3))
        Argba = reinterpret(RGBA{UFixed16}, reinterpret(UFixed16, A))
        B = Images.restrict(Argba)
        Tr = eltype(eltype(B))
        @test isapprox(reinterpret(Tr,B),Images.restrict(A,(2,3)) / reinterpret(one(UFixed16)),atol=eps(Tr) ^ (2 / 3))
        A = reshape(1:60, 5, 4, 3)
        B = Images.restrict(A, (1,2,3))
        @test cat(3, [ 2.6015625  8.71875 6.1171875;
                       4.09375   12.875   8.78125;
                       3.5390625 10.59375 7.0546875],
                     [10.1015625 23.71875 13.6171875;
                      14.09375   32.875   18.78125;
                      11.0390625 25.59375 14.5546875]) ≈ B
        imgcol["pixelspacing"] = [1,1]
        imgr = Images.restrict(imgcol, (1,2))
        @test pixelspacing(imgr) == [2,2]
        @test pixelspacing(imgcol) == [1,1] # issue #347
        # Issue #395
        img1 = colorim(fill(0.9, 3, 5, 5))
        img2 = colorim(fill(U8(0.9), 3, 5, 5))
        @test isapprox(separate(restrict(img1)),separate(restrict(img2)),atol=0.01)
    end

    @testset "Erode/ dilate" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ae = Images.erode(A)
        @test Ae == zeros(size(A))
        Ad = Images.dilate(A)
        Ar = [0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0.6 0.6;
              0 0 0.6 0.6]
        @test Ad == cat(3,Ar,Ag,zeros(4,4))
        Ae = Images.erode(Ad)
        Ar = [0.8 0.8 0 0;
              0.8 0.8 0 0;
              0 0 0 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0 0;
              0 0 0 0.6]
        @test Ae == cat(3,Ar,Ag,zeros(4,4))
        # issue #311
        @test Images.dilate(trues(3)) == trues(3)
    end

    @testset "Extrema_filter" begin
        # 2d case
        A = zeros(5,5)
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @test A[reshape(matching,size(A))] == [0.8,0.6]
        # 3d case
        A = zeros(5,5,5)
        A[2,2,2] = 0.7
        A[4,4,3] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @test A[reshape(matching,size(A))] == [0.7,0.5]
        # 4d case
        A = zeros(5,5,5,5)
        A[2,2,2,2] = 0.7
        A[4,4,3,1] = 0.4
        A[3,4,3,2] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @test A[reshape(matching,size(A))] == [0.4,0.7,0.5]
        x, y, z, t = ind2sub(size(A), find(A .== 0.4))
        @test x[1] == 4
        @test y[1] == 4
        @test z[1] == 3
        @test t[1] == 1
        # 2d case
        A = rand(5,5)/10
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, [2, 2])
        matching = falses(A)
        matching[2:end, 2:end] = maxval .== A[2:end, 2:end]
        @test (sort(A[matching]))[end - 1:end] == [0.6,0.8]
        # 3d case
        A = rand(5,5,5)/10
        A[2,2,2] = 0.7
        A[4,4,2] = 0.4
        A[2,2,4] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end]
        @test (sort(A[matching]))[end - 2:end] == [0.4,0.5,0.7]
        # 4d case
        A = rand(5,5,5,5)/10
        A[2,2,2,2] = 0.7
        A[4,4,2,3] = 0.4
        A[2,2,4,3] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end, 2:end]
        @test (sort(A[matching]))[end - 2:end] == [0.4,0.5,0.7]
    end

    @testset "Opening / closing" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ao = Images.opening(A)
        @test Ao == zeros(size(A))
        A = zeros(10,10)
        A[4:7,4:7] = 1
        B = copy(A)
        A[5,5] = 0
        Ac = Images.closing(A)
        @test Ac == B
    end

    @testset "Morphological Top-hat" begin
        A = zeros(13, 13)
        A[2:3, 2:3] = 1
        Ae = copy(A)
        A[5:9, 5:9] = 1
        Ao = Images.tophat(A)
        @test Ao == Ae
        Aoo = Images.tophat(Ae)
        @test Aoo == Ae
    end

    @testset "Morphological Bottom-hat" begin
        A = ones(13, 13)
        A[2:3, 2:3] = 0
        Ae = 1 - copy(A)
        A[5:9, 5:9] = 0
        Ao = Images.bothat(A)
        @test Ao == Ae
    end

    @testset "Morphological Gradient" begin
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morphogradient(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[6:8, 6:8] = 0
        @test Ao == Ae
        Aee = Images.dilate(A) - Images.erode(A)
        @test Aee == Ae
    end

    @testset "Morphological Laplacian" begin
        A = zeros(13, 13)
        A[5:9, 5:9] = 1
        Ao = Images.morpholaplace(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] = 1
        Ae[5:9, 5:9] = -1
        Ae[6:8, 6:8] = 0
        @test Ao == Ae
        Aee = Images.dilate(A) + Images.erode(A) - 2A
        @test Aee == Ae
    end

    @testset "Label components" begin
        A = [true  true  false true;
             true  false true  true]
        lbltarget = [1 1 0 2;
                     1 0 2 2]
        lbltarget1 = [1 2 0 4;
                      1 0 3 4]
        @test Images.label_components(A) == lbltarget
        @test Images.label_components(A,[1]) == lbltarget1
        connectivity = [false true  false;
                        true  false true;
                        false true  false]
        @test Images.label_components(A,connectivity) == lbltarget
        connectivity = trues(3,3)
        lbltarget2 = [1 1 0 1;
                      1 0 1 1]
        @test Images.label_components(A,connectivity) == lbltarget2
        @test component_boxes(lbltarget) == Vector{Tuple}[[(1,2),(2,3)],[(1,1),(2,2)],[(1,3),(2,4)]]
        @test component_lengths(lbltarget) == [2,3,3]
        @test component_indices(lbltarget) == Array{Int64}[[4,5],[1,2,3],[6,7,8]]
        @test component_subscripts(lbltarget) == Array{Tuple}[[(2,2),(1,3)],[(1,1),(2,1),(1,2)],[(2,3),(1,4),(2,4)]]
        @test component_centroids(lbltarget) == Tuple[(1.5,2.5),(4 / 3,4 / 3),(5 / 3,11 / 3)]
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
        Q = Images.shepp_logan(8)
        @test norm((P - Q)[:]) < 1.0e-10
        P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
              0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
              0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
              0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
              0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]
        Q = Images.shepp_logan(8, highContrast=false)
        @test norm((P - Q)[:]) < 1.0e-10
    end

    @testset "Image resize" begin
        img = convert(Images.Image, zeros(10,10))
        img2 = Images.imresize(img, (5,5))
        @test length(img2) == 25
    end

    @testset "Interpolations" begin

        img = zeros(Float64, 5, 5)
        @test bilinear_interpolation(img,4.5,5.5) == 0.0
        @test bilinear_interpolation(img,4.5,3.5) == 0.0

        for i in [1.0, 2.0, 5.0, 7.0, 9.0]
            img = ones(Float64, 5, 5) * i
            @test bilinear_interpolation(img,3.5,4.5) == i
            @test bilinear_interpolation(img,3.2,4) == i # X_MAX == X_MIN
            @test bilinear_interpolation(img,3.2,4) == i # Y_MAX == Y_MIN
            @test bilinear_interpolation(img,3.2,4) == i # BOTH EQUAL
            @test bilinear_interpolation(img,2.8,1.9) == i
            # One dim out of bounds
            @test isapprox(bilinear_interpolation(img,0.5,1.5),0.5i)
            @test isapprox(bilinear_interpolation(img,0.5,1.6),0.5i)
            @test isapprox(bilinear_interpolation(img,0.5,1.8),0.5i)
            # Both out of bounds (corner)
            @test isapprox(bilinear_interpolation(img,0.5,0.5),0.25i)
        end

        img = reshape(1.0:1.0:25.0, 5, 5)
        @test bilinear_interpolation(img,1.5,2) == 6.5
        @test bilinear_interpolation(img,2,1.5) == 4.5
        @test bilinear_interpolation(img,2,1) == 2.0
        @test bilinear_interpolation(img,1.5,2.5) == 9.0
        @test bilinear_interpolation(img,1.5,3.5) == 14.0
        @test bilinear_interpolation(img,1.5,4.5) == 19.0
        @test bilinear_interpolation(img,1.5,5.5) == 10.75

    end

end
