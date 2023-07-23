using Test, Images, TestImages

# why use 3 chars when many will do?
eye(m,n) = Matrix{Float64}(I,m,n)

@testset "Exposure" begin
    oneunits(::Type{C}, dims...) where C = fill(oneunit(C), dims)

    @testset "Histogram" begin

        # Many of these test integer values, but of course most images
        # should have pixel values between 0 and 1.  Still, it doesn't
        # hurt to get the integer case right too.
        img = 1:10
        bins, hist = build_histogram(img, 10)
        @test length(hist) == length(bins)+1
        @test_broken bins == 1.0:1.0:11.0
        @test_broken hist == [0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0]
        bins, hist = build_histogram(img, 5; minval=2, maxval=6)
        @test length(hist) == length(bins)+1
        @test_broken bins == 2.0:1.0:7.0
        @test_broken hist == [1, 1, 1, 1, 1, 1, 4]

        img = reshape(0:99, 10, 10)
        bins, hist = build_histogram(img, 10)
        @test length(hist) == length(bins)+1
        @test_broken bins == 0.0:10.0:100.0
        @test_broken hist == [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0]
        bins, hist = build_histogram(img, 7; minval=25, maxval=59)
        @test length(hist) == length(bins)+1
        @test_broken bins == 25.0:5.0:60.0
        @test_broken hist == [25, 5, 5, 5, 5, 5, 5, 5, 40]

        # Test the more typical case
        img = reinterpret(Gray{N0f8}, [0x20,0x40,0x80,0xd0])
        @test_broken build_histogram(img, 5) == (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{N0f8}, [0x00,0x40,0x80,0xd0])
        @test_broken build_histogram(img, 5) == (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{N0f8}, [0x00,0x40,0x80,0xff])
        @test_broken all((≈).(build_histogram(img, 6), (0.0:0.2:1.2,[0,1,1,1,0,0,1,0])))

        # Consider an image where each intensity occurs only once and vary the number
        # of bins used in the histogram in powers of two. With the exception of the
        # first bin (with index 0), all other bins should have equal counts.
        expected_counts = [2^i for i = 0:7]
        bins = [2^i for i = 8:-1:1]
        for i = 1:length(bins)
            for T in (Gray{N0f8}, Gray{N0f16}, Gray{Float32}, Gray{Float64})
                edges, counts  = build_histogram(T.(collect(0:1/255:1)), bins[i]; minval=0, maxval=1)
                @test length(edges) == length(counts) - 1
                @test all(counts[1:end] .== expected_counts[i]) && counts[0] == 0
                @test axes(counts) == (0:length(edges),)
            end

            # Verify that the function can also take a color image as an input.
            for T in (RGB{N0f8}, RGB{N0f16}, RGB{Float32}, RGB{Float64})
                imgg = collect(0:1/255:1)
                img = colorview(RGB,imgg,imgg,imgg)
                edges, counts  = build_histogram(T.(img), bins[i]; minval=0, maxval=1)
                @test length(edges) == length(counts) - 1
                @test all(counts[1:end] .== expected_counts[i]) && counts[0] == 0
                @test axes(counts) == (0:length(edges),)
            end

            # Consider also integer-valued images.
            edges, counts  = build_histogram(0:1:255, bins[i]; minval=0, maxval=255)
            @test length(edges) == length(counts) - 1
            @test all(counts[1:end] .== expected_counts[i]) && counts[0] == 0
            @test axes(counts) == (0:length(edges),)
        end

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred build_histogram(img, 10; minval = 0.0, maxval = 1.0)

    end

    @testset "Histogram Equalisation" begin

        #DataTypes
        img = oneunits(Gray{Float64}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f8}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f16}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test eltype(ret) == eltype(img)

        @info "Test skippped due to lack of support for `AGray` in adjust_histogram"
        # img = oneunits(AGray{N0f8}, 10, 10)
        # ret = adjust_histogram(img, Equalization(;nbins=100))
        # @test img == ret
        # @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f8}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f16}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test img == ret
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{Float64}, 10, 10)
        ret = adjust_histogram(img, Equalization(;nbins=100))
        @test all(map((i, r) -> isapprox(i, r), img, ret))
        @test eltype(ret) == eltype(img)

        @info "Test skipped due to lack of support for `ARGB` in adjust_histogram"
        # img = oneunits(ARGB{N0f8}, 10, 10)
        # ret = adjust_histogram(img, Equalization(;nbins=100))
        # @test img == ret
        # @test eltype(ret) == eltype(img)

        img = zeros(10, 10)
        for i in 1:10
            img[i, :] .= 10 * (i - 1)
        end
        @test_broken img == adjust_histogram(img, Equalization(;nbins=10, minval=0, maxval=90))

        ret = adjust_histogram(img, Equalization(;nbins=2, minval=0, maxval=90))
        @test all(ret[1:5, :] .== 0)
        @test all(ret[6:10, :] .== 90)

        ret = adjust_histogram(img, Equalization(;nbins=5, minval=0, maxval=90))
        for i in 1:2:10
            if i ∈ (1, 9)
                @test all(ret[i:i+1, :] .== 22.5 * floor(i / 2))
            else
                @test_broken all(ret[i:i+1, :] .== 22.5 * floor(i / 2))
            end
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

        ret = adjust_histogram(img, Equalization(; nbins=10, minval=0, maxval=99))
        cdf = cumsum(build_histogram(img, 10)[2][2:end-1])
        @test all(ret[1:cdf[1]] .== 0.0)
        for i in 1:(size(cdf)[1]-1)
            @test_broken all(ret[cdf[i] + 1 : cdf[i + 1]] .== (cdf[i + 1] - cdf[1]) * 99.0 / (cdf[end] - cdf[1]))
        end

        for T in (Gray{N0f8}, Gray{N0f16}, Gray{Float32}, Gray{Float64})
            #=
            Create an image that spans a narrow graylevel range. Then quantize
            the 256 bins down to 32 and determine how many bins have non-zero
            counts.
            =#

            img = Gray{Float32}.([i/255.0 for i = 64:128, j = 1:10])
            img = T.(img)
            _, counts_before = build_histogram(img,32; minval=0, maxval=1)
            nonzero_before = sum(counts_before .!= 0)

            #=
            Equalize the image histogram. Then quantize the 256 bins down to 32
            and verify that all 32 bins have non-zero counts. This will confirm
            that the dynamic range of the original image has been increased.
            =#
            imgeq = adjust_histogram(img, Equalization(256,0,1))
            edges, counts_after = build_histogram(imgeq,32; minval=0, maxval=1)
            nonzero_after = sum(counts_after .!= 0)
            @test nonzero_before < nonzero_after
            @test nonzero_after == 32
        end


        for T in (RGB{N0f8}, RGB{N0f16}, RGB{Float32}, RGB{Float64})
            #=
            Create a color image that spans a narrow graylevel range.  Then
            quantize the 256 bins down to 32 and determine how many bins have
            non-zero counts.
            =#

            imgg = Gray{Float32}.([i/255.0 for i = 64:128, j = 1:10])
            img = colorview(RGB,imgg,imgg,imgg)
            img = T.(img)
            _, counts_before = build_histogram(img,32; minval=0, maxval=1)
            nonzero_before = sum(counts_before .!= 0)

            #=
            Equalize the histogram. Then quantize the 256 bins down to 32 and
            verify that all 32 bins have non-zero counts. This will confirm that
            the dynamic range of the original image has been increased.
            =#
            imgeq = adjust_histogram(img, Equalization(256,0,1))
            edges, counts_after = build_histogram(imgeq,32; minval=0, maxval=1)
            nonzero_after = sum(counts_after .!= 0)
            @test nonzero_before < nonzero_after
            @test nonzero_after == 32
        end

        # Verify that the minimum and maximum values of the equalised image match the
        # specified minimum and maximum values, i.e. that the intensities of the equalised
        # image are in the interval [minvalue, maxvalue].
        imgeq = adjust_histogram(collect(0:1:255), Equalization(256,64,128))
        @test all(imgeq[1:65] .== 64)
        @test all(imgeq[128+1:end] .== 128)

        imgeq = adjust_histogram(collect(0:1/255:1), Equalization(256,64/255,128/255))
        @test all(imgeq[1:65] .== 64/255)
        @test all(imgeq[128+1:end] .== 128/255)

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred adjust_histogram(img, Equalization())
        @inferred adjust_histogram!(img, Equalization())

    end

    @testset "Gamma Correction" begin

        img = oneunits(Gray{Float64}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test img == ret
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(0, (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        img = oneunits(Gray{N0f8}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test img == ret
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(0, (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        img = oneunits(Gray{N0f16}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(0, (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        @info "Test skipped due to lack of support for `AGray` in adjust_histogram"
        # img = oneunits(AGray{N0f8}, 10, 10)
        # ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        # @test img == ret
        # @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(0, (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        img = oneunits(RGB{N0f8}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test img == ret
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(zero(eltype(img)), (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        img = oneunits(RGB{N0f16}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test img == ret
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(zero(eltype(img)), (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test imgp == retp
        @test eltype(retp) == eltype(imgp)

        img = oneunits(RGB{Float64}, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        @test all(map((i, r) -> isapprox(i, r), img, ret))
        @test eltype(ret) == eltype(img)

        imgp = padarray(img, Fill(zero(eltype(img)), (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test all(map((i, r) -> isapprox(i, r), imgp, retp))
        @test eltype(retp) == eltype(imgp)

        @info "Test skipped due to lack of support for `ARGB` in adjust_histogram"
        # img = oneunits(ARGB{N0f8}, 10, 10)
        # ret = adjust_histogram(img, GammaCorrection(; gamma=1))
        # @test img == ret

        imgp = padarray(img, Fill(zero(eltype(img)), (2,2)))
        retp = adjust_histogram(imgp, GammaCorrection(; gamma=1))
        @test_broken imgp == retp

        img = reshape(1:1:100, 10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=2))
        @test ret == img .^ 2
        ret = adjust_histogram(img, GammaCorrection(; gamma=0.3))
        @test_broken ret == img .^ 0.3
        @info "Test skipped due to lack of support for `minval` and `maxval` in `GammaCorrection`"
        # ret = adjust_histogram(img, GammaCorrection(; gamma=1, minval=1, maxval=100))
        # @test ret == img

        img = zeros(10, 10)
        ret = adjust_histogram(img, GammaCorrection(; gamma=2))
        @test all(ret .== 0)

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred adjust_histogram(img, GammaCorrection(0.1))
        @inferred adjust_histogram!(img, GammaCorrection(0.1))

    end

    @testset "Histogram Matching" begin

        #DataTypes
        img = oneunits(Gray{Float64}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test_broken all(ret .== zero(eltype(img)))
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f8}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test_broken all(ret .== zero(eltype(img)))
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f16}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test eltype(ret) == eltype(img)

        @info "Test skipped due to lack of support for `AGray` in adjust_histogram"
        # img = oneunits(AGray{N0f8}, 10, 10)
        # ret = adjust_histogram(img, Matching(targetimg=img))
        # @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f8}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f16}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{Float64}, 10, 10)
        ret = adjust_histogram(img, Matching(targetimg=img))
        @test_broken all(map((i, r) -> isapprox(zero(RGB), r, atol = 0.001), img, ret))
        @test eltype(ret) == eltype(img)

        @info "Test skipped due to lack of support for `ARGB` in adjust_histogram"
        # img = oneunits(ARGB{N0f8}, 10, 10)
        # ret = adjust_histogram(img, Matching(targetimg=img))
        # @test eltype(ret) == eltype(img)

        img = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges, hist = build_histogram(img, 2)
        himg = adjust_histogram(img, Matching(; targetimg=img, edges))
        @test_broken himg == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5]
        edges, hist = build_histogram(img, 5)
        himg = adjust_histogram(img, Matching(; targetimg=img, edges))
        @test_broken himg == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]

        @test_throws ErrorException Matching()

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred adjust_histogram(img, Matching(targetimg = img))
        @inferred adjust_histogram!(img, Matching(targetimg = img))
    end

    @testset "AdaptiveEqualization" begin

        #DataTypes
        img = oneunits(Gray{Float64}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f8}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = oneunits(Gray{N0f16}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        @info "Test skipped for lack of support for `AGray` in adjust_histogram"
        # img = oneunits(AGray{N0f8}, 10, 10)
        # ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        # @test size(ret) == size(img)
        # @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f8}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{N0f16}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        img = oneunits(RGB{Float64}, 10, 10)
        ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        @test size(ret) == size(img)
        @test eltype(ret) == eltype(img)

        @info "Test skipped for lack of support for `ARGB` in adjust_histogram"
        # img = oneunits(ARGB{N0f8}, 10, 10)
        # ret = adjust_histogram(img, AdaptiveEqualization(nbins=100))
        # @test size(ret) == size(img)
        # @test eltype(ret) == eltype(img)


        # TODO: the "boundary regions" img[4:5, :] and img[:, 4:5] are not
        # "symmetric," is this a problem?

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred adjust_histogram(img, AdaptiveEqualization())
        @inferred adjust_histogram!(img, AdaptiveEqualization())
    end

    @testset "Other" begin

        # Issue #282
        rawimg = [1 0; 0 1]
        img = Gray{N0f8}.(rawimg)
        imgs = adjust_histogram(img, ContrastStretching(t=0.3, slope=0.4, ϵ=eps(Float32)))
        @test imgs ≈ N0f8.(1 ./ (1 .+ (0.3 ./ (rawimg .+ eps(float(N0f8)))).^0.4))

        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test adjust_histogram(img, LinearStretching(; src_minval=0.0103761, src_maxval=0.0252166))[2,1] == 0.0
        @test eltype(adjust_histogram(img, LinearStretching())) == Gray{N0f16}

        hist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Verify exported ImageContrastAdjustment symbol
        img = Gray{N0f8}.(rand(10, 10));
        @inferred adjust_histogram(img, ContrastStretching())
    end

end

nothing
