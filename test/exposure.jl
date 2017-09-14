using Base.Test, Images, Colors, FixedPointNumbers

@testset "Exposure" begin

    @testset "Histogram" begin

        # Many of these test integer values, but of course most images
        # should have pixel values between 0 and 1.  Still, it doesn't
        # hurt to get the integer case right too.
        img = 1:10
        bins, hist = imhist(img, 10)
        @test length(hist) == length(bins)+1
        @test bins == 1.0:1.0:11.0
        @test hist == [0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0]
        bins, hist = imhist(img, 5, 2, 6)
        @test length(hist) == length(bins)+1
        @test bins == 2.0:1.0:7.0
        @test hist == [1, 1, 1, 1, 1, 1, 4]

        img = reshape(0:99, 10, 10)
        bins, hist = imhist(img, 10)
        @test length(hist) == length(bins)+1
        @test bins == 0.0:10.0:100.0
        @test hist == [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0]
        bins, hist = imhist(img, 7, 25, 59)
        @test length(hist) == length(bins)+1
        @test bins == 25.0:5.0:60.0
        @test hist == [25, 5, 5, 5, 5, 5, 5, 5, 40]

        # Test the more typical case
        img = reinterpret(Gray{N0f8}, [0x20,0x40,0x80,0xd0])
        @test imhist(img, 5) == (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{N0f8}, [0x00,0x40,0x80,0xd0])
        @test imhist(img, 5) == (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{N0f8}, [0x00,0x40,0x80,0xff])
        @test imhist(img, 6) == (0.0:0.2:1.2,[0,1,1,1,0,0,1,0])

    end

    @testset "Histogram Equalisation" begin

        #DataTypes
        img = ones(Gray{Float64}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f8}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f16}, 10, 10)
        ret = histeq(img, 100)
        @test eltype(ret) == eltype(img)

        img = ones(AGray{N0f8}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f8}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f16}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{Float64}, 10, 10)
        ret = histeq(img, 100)
        @test all(map((i, r) -> isapprox(i, r), img, ret))
        @test eltype(ret) == eltype(img)

        img = ones(ARGB{N0f8}, 10, 10)
        ret = histeq(img, 100)
        @test img == ret
        @test eltype(ret) == eltype(img)

        #Working

        img = zeros(10, 10)
        for i in 1:10
            img[i, :] = 10 * (i - 1)
        end
        @test img == histeq(img, 10, 0, 90)

        ret = histeq(img, 2, 0, 90)
        @test all(ret[1:5, :] .== 0)
        @test all(ret[6:10, :] .== 90)

        ret = histeq(img, 5, 0, 90)
        for i in 1:2:10
            @test all(ret[i:i+1, :] .== 22.5 * floor(i / 2))
        end

        img = [0.0,  21.0,  29.0,  38.0,  38.0,  51.0,  66.0,  79.0,  79.0,  91.0,
               1.0,  21.0,  28.0,  39.0,  39.0,  52.0,  63.0,  77.0,  77.0,  91.0,
               2.0,  22.0,  21.0,  35.0,  35.0,  56.0,  64.0,  73.0,  73.0,  94.0,
               3.0,  21.0,  22.0,  33.0,  33.0,  53.0,  63.0,  72.0,  72.0,  93.0,
               4.0,  22.0,  27.0,  32.0,  32.0,  52.0,  64.0,  78.0,  78.0,  92.0,
               11.0,  21.0,  31.0,  31.0,  42.0,  66.0,  71.0,  71.0,  88.0,  91.0,
               12.0,  22.0,  32.0,  32.0,  44.0,  62.0,  75.0,  75.0,  88.0,  91.0,
               13.0,  23.0,  34.0,  34.0,  49.0,  67.0,  74.0,  74.0,  82.0,  94.0,
               14.0,  26.0,  35.0,  35.0,  43.0,  68.0,  74.0,  74.0,  83.0,  93.0,
               15.0,  27.0,  36.0,  36.0,  44.0,  69.0,  74.0,  74.0,  86.0,  92.0]
        img = reshape(img, 10, 10)'

        ret = histeq(img, 10, 0, 99)
        cdf = cumsum(imhist(img, 10)[2][2:end-1])
        @test all(ret[1:cdf[1]] .== 0.0)
        for i in 1:(size(cdf)[1]-1)
            @test all(ret[cdf[i] + 1 : cdf[i + 1]] .== (cdf[i + 1] - cdf[1]) * 99.0 / (cdf[end] - cdf[1]))
        end

    end

    @testset "Gamma Correction" begin

        img = ones(Gray{Float64}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f16}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test eltype(ret) == eltype(img)

        img = ones(AGray{N0f8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f16}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = ones(RGB{Float64}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test all(map((i, r) -> isapprox(i, r), img, ret))
        @test eltype(ret) == eltype(img)

        img = ones(ARGB{N0f8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @test img == ret

        #Working

        img = reshape(1:1:100, 10, 10)
        ret = adjust_gamma(img, 2)
        @test ret == img .^ 2
        ret = adjust_gamma(img, 0.3)
        @test ret == img .^ 0.3
        ret = adjust_gamma(img, 1, 1, 100)
        @test ret == img

        img = zeros(10, 10)
        ret = adjust_gamma(img, 2)
        @test all(ret .== 0)

        a = ARGB(0.2, 0.3, 0.4, 0.5)
        r = Images._gamma_pixel_rescale(a, 1)
        @test isapprox(a, r, rtol = 0.0001)
        r = Images._gamma_pixel_rescale(a, 2)
        @test alpha(r) == alpha(a)

        b = AGray(0.2, 0.6)
        r = Images._gamma_pixel_rescale(b, 1)
        @test b == r
        r = Images._gamma_pixel_rescale(b, 2)
        @test alpha(r) == alpha(b)
        @test isapprox(r.val, b.val ^ 2)

    end

    @testset "Histogram Matching" begin

        #DataTypes
        img = ones(Gray{Float64}, 10, 10)
        ret = histmatch(img, img)
        @test all(ret .== zero(eltype(img)))
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f8}, 10, 10)
        ret = histmatch(img, img)
        @test all(ret .== zero(eltype(img)))
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f16}, 10, 10)
        ret = histmatch(img, img)
        @test eltype(ret) == eltype(img)

        img = ones(AGray{N0f8}, 10, 10)
        ret = histmatch(img, img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f8}, 10, 10)
        ret = histmatch(img, img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f16}, 10, 10)
        ret = histmatch(img, img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{Float64}, 10, 10)
        ret = histmatch(img, img)
        @test all(map((i, r) -> isapprox(zero(RGB), r, atol = 0.001), img, ret))
        @test eltype(ret) == eltype(img)

        img = ones(ARGB{N0f8}, 10, 10)
        ret = histmatch(img, img)
        @test eltype(ret) == eltype(img)

        img = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges, hist = imhist(img, 2)
        himg = Images._histmatch(img, edges, hist)
        @test himg == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5]
        edges, hist = imhist(img, 5)
        himg = Images._histmatch(img, edges, hist)
        @test himg == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
    end

    @testset "CLAHE" begin

        #DataTypes
        img = ones(Gray{Float64}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f8}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(Gray{N0f16}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(AGray{N0f8}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f8}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{N0f16}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(RGB{Float64}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = ones(ARGB{N0f8}, 10, 10)
        ret = clahe(img, 100)
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        #Working

        cdf = Array(0.1:0.1:1.0)
        res_pix = Images._clahe_pixel_rescale(0.5654, cdf, cdf, cdf, cdf, 0.1:0.1:1.0, 1, 4, 10, 10)
        @test isapprox(res_pix, 0.5)
        res_pix = Images._clahe_pixel_rescale(0.2344, cdf, cdf, cdf, cdf, 0.1:0.1:1.0, 1, 4, 10, 10)
        @test isapprox(res_pix, 0.2)
        res_pix = Images._clahe_pixel_rescale(0.8123, cdf, cdf, cdf, cdf, 0.1:0.1:1.0, 1, 4, 10, 10)
        @test isapprox(res_pix, 0.8)

        cdf2 = [0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0]

        res_pix = Images._clahe_pixel_rescale(0.3, cdf2, cdf, 0.1:0.1:1, 3, 10)
        @test isapprox(res_pix, 0.144444444)

        res_pix = Images._clahe_pixel_rescale(0.3, cdf2, cdf, 0.1:0.1:1, 6, 10)
        @test isapprox(res_pix, 0.211111111)

        res_pix = Images._clahe_pixel_rescale(0.5654, cdf2, 0.1:0.1:1.0)
        @test isapprox(res_pix, 0.2)

        cdf3 = [0.0, 0.0, 0.1, 0.3, 0.5, 0.5, 0.6, 0.9, 1.0, 1.0]
        cdf4 = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.8, 1.0]

        res_pix = Images._clahe_pixel_rescale(0.8123, cdf, cdf2, cdf3, cdf4, 0.1:0.1:1.0, 3, 4, 10, 10)
        @test isapprox(res_pix, 0.781481481)
        res_pix = Images._clahe_pixel_rescale(0.3, cdf, cdf2, cdf3, cdf4, 0.1:0.1:1.0, 5, 4, 10, 10)
        @test isapprox(res_pix, 0.2037037037)

        img = [zeros(4,4) 0.1*ones(4,4); 0.2*ones(4,4) 0.3*ones(4,4)]
        hist_eq_img = clahe(img, xblocks = 2, yblocks = 2, clip = 5)
        # TODO: it would be good to understand where these numbers come from
        @test all(x->x ≈ 0.2960140679953115, hist_eq_img[1:3, 1:3])
        @test all(x->x ≈ 0.4196951934349368, hist_eq_img[1:3, end-2:end])
        @test all(x->x ≈ 0.4841735052754997, hist_eq_img[end-2:end, 1:3])
        @test all(x->x ≈ 0.5486518171160624, hist_eq_img[end-2:end, end-2:end])
        # TODO: the "boundary regions" img[4:5, :] and img[:, 4:5] are not
        # "symmetric," is this a problem?

        # The following test is sensitive to any change in the
        # algorithm. Is there way to make this more generic?
        # img = [ 0.907976  0.0601377   0.747873   0.938683   0.935189  0.49492    0.508545  0.394573   0.991136   0.210218
        #         0.219737  0.727806    0.606995   0.25272    0.279777  0.584529   0.708244  0.958645   0.979217   0.418678
        #         0.793384  0.314803    0.324222   0.840858   0.49438   0.102003   0.724399  0.133663   0.312085   0.662211
        #         0.905336  0.423511    0.564616   0.692544   0.367656  0.630386   0.133439  0.529039   0.0175596  0.644065
        #         0.255092  0.00242822  0.723724   0.673323   0.244508  0.518068   0.204353  0.34222    0.106092   0.0331161
        #         0.432963  0.383491    0.903465   0.579605   0.719874  0.571533   0.728544  0.1864     0.950187   0.470226
        #         0.877475  0.554114    0.987133   0.947148   0.710115  0.131948   0.711611  0.0221843  0.470008   0.952806
        #         0.231911  0.177463    0.742054   0.0333307  0.481319  0.716638   0.332057  0.978177   0.0610481  0.439462
        #         0.935457  0.159602    0.178357   0.163585   0.275052  0.0557963  0.066368  0.199349   0.9238     0.85161
        #         0.692148  0.503563    0.0918827  0.0206237  0.702344  0.546088   0.04163   0.174103   0.45499    0.90019
        #         ]
        # hist_eq_img = clahe(img, xblocks = 2, yblocks = 2, clip = 5)
        # ret_expected = [ 0.972222  0.154261  0.672734  0.962918  0.960518  0.603715  0.559332  0.684868  0.919632  0.11213
        #                  0.26227   0.577172  0.615628  0.420623  0.356921  0.581208  0.845865  0.888908  0.908469  0.518953
        #                  0.79315   0.398521  0.454306  0.692093  0.404984  0.287649  0.619472  0.47354   0.442781  0.826681
        #                  0.875951  0.325109  0.561585  0.746554  0.355356  0.542615  0.308523  0.303799  0.12818   0.662404
        #                  0.218908  0.145377  0.627466  0.703423  0.255746  0.543485  0.270196  0.240487  0.119202  0.0762326
        #                  0.443491  0.419296  0.857736  0.778583  0.78256   0.660936  0.66722   0.420796  0.793904  0.50695
        #                  0.765105  0.583873  0.870947  0.825773  0.755839  0.346564  0.598017  0.329962  0.564487  0.897893
        #                  0.512372  0.328346  0.50314   0.339795  0.493125  0.476082  0.45704   0.597256  0.385295  0.814294
        #                  0.866771  0.256846  0.140199  0.118129  0.287474  0.175723  0.115673  0.343809  0.735832  0.928661
        #                  0.833503  0.481259  0.135795  0.116606  0.737954  0.57691   0.103692  0.132731  0.539984  0.999525
        #                  ]
        # @test all(map((i, j) -> isapprox(i, j, rtol = 0.00001), hist_eq_img, ret_expected))

    end

    @testset "Other" begin

        # Issue #282
        img = Gray{N0f8}.(eye(2,2))
        imgs = imstretch(img, 0.3, 0.4)
        @test imgs ≈ 1./(1 + (0.3./(eye(2,2) + eps())).^0.4)

        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test imadjustintensity(img,[0.0103761, 0.0252166])[2,1] == 0.0
        @test eltype(imadjustintensity(img)) == Gray{N0f16}

        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test complement(Gray(0.5)) == Gray(0.5)
        @test complement(Gray(0.2)) == Gray(0.8)
        @test all(complement.(img) .== 1 - img)

        hist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        clipped_hist = cliphist(hist, 2)
        @test all(clipped_hist .== 2.0)
        clipped_hist = cliphist(hist, 3)
        @test all(clipped_hist .== 3.0)
        clipped_hist = cliphist(hist, 4)
        @test all(clipped_hist .== 4.0)
        clipped_hist = cliphist(hist, 5)
        @test all(clipped_hist .== 5.0)
        clipped_hist = cliphist(hist, 6)
        @test clipped_hist == vec([3, 4, 6, 6, 6, 6, 6, 6, 6, 6])
        clipped_hist = cliphist(hist, 7)
        @test clipped_hist == vec([2.6, 3.6, 3.6, 4.6, 5.6, 7, 7, 7, 7, 7])
        clipped_hist = cliphist(hist, 8)
        @test clipped_hist == vec([2.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8, 8, 8])
        clipped_hist = cliphist(hist, 9)
        @test clipped_hist == vec([2.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9, 9])
        clipped_hist = cliphist(hist, 10)
        @test clipped_hist == hist
    end

end

nothing
