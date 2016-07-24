using FactCheck, Base.Test, Images, Colors, FixedPointNumbers

facts("Exposure") do

	context("Histogram") do
    
        # Many of these test integer values, but of course most images
        # should have pixel values between 0 and 1.  Still, it doesn't
        # hurt to get the integer case right too.
        img = 1:10
        bins, hist = imhist(img, 10)
        @fact length(hist) --> length(bins)+1
        @fact bins --> 1.0:1.0:11.0
        @fact hist --> [0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0]
        bins, hist = imhist(img, 5, 2, 6)
        @fact length(hist) --> length(bins)+1
        @fact bins --> 2.0:1.0:7.0
        @fact hist --> [1, 1, 1, 1, 1, 1, 4]

        img = reshape(0:99, 10, 10)
        bins, hist = imhist(img, 10)
        @fact length(hist) --> length(bins)+1
        @fact bins --> 0.0:10.0:100.0
        @fact hist --> [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0]
        bins, hist = imhist(img, 7, 25, 59)
        @fact length(hist) --> length(bins)+1
        @fact bins --> 25.0:5.0:60.0
        @fact hist --> [25, 5, 5, 5, 5, 5, 5, 5, 40]

        # Test the more typical case
        img = reinterpret(Gray{U8}, [0x20,0x40,0x80,0xd0])
        @fact imhist(img, 5) --> (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{U8}, [0x00,0x40,0x80,0xd0])
        @fact imhist(img, 5) --> (0.0:0.2:1.0,[0,1,1,1,0,1,0])
        img = reinterpret(Gray{U8}, [0x00,0x40,0x80,0xff])
        @fact imhist(img, 6) --> (0.0:0.2:1.2,[0,1,1,1,0,0,1,0])
    
    end

	context("Histogram Equalisation") do
        
        #DataTypes
        img = ones(Images.Gray{Float64}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U8}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U16}, 10, 10)
        ret = histeq(img, 100)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.AGray{Images.U8}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U8}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U16}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Float64}, 10, 10)
        ret = histeq(img, 100)
        @fact all(map((i, r) -> isapprox(i, r), img, ret)) --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.ARGB{Images.U8}, 10, 10)
        ret = histeq(img, 100)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        #Working

        img = zeros(10, 10)
        for i in 1:10
            img[i, :] = 10 * (i - 1)
        end
        @fact img == Images.histeq(img, 10, 0, 90) --> true
        
        ret = Images.histeq(img, 2, 0, 90)
        @fact all(ret[1:5, :] .== 0) --> true
        @fact all(ret[6:10, :] .== 90) --> true
        
        ret = Images.histeq(img, 5, 0, 90)
        for i in 1:2:10
            @fact all(ret[i:i+1, :] .== 22.5 * floor(i / 2)) --> true
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

        ret = Images.histeq(img, 10, 0, 99)
        cdf = cumsum(imhist(img, 10)[2][2:end-1])
        @fact all(ret[1:cdf[1]] .== 0.0) --> true
        for i in 1:(size(cdf)[1]-1)
            @fact all(ret[cdf[i] + 1 : cdf[i + 1]] .== (cdf[i + 1] - cdf[1]) * 99.0 / (cdf[end] - cdf[1])) --> true
        end

    end
    
	context("Gamma Correction") do

        img = ones(Images.Gray{Float64}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U16}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.AGray{Images.U8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U16}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Float64}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact all(map((i, r) -> isapprox(i, r), img, ret)) --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.ARGB{Images.U8}, 10, 10)
        ret = adjust_gamma(img, 1)
        @fact img == ret --> true

        #Working

        img = reshape(1:1:100, 10, 10)
        ret = Images.adjust_gamma(img, 2)
        @fact ret == img .^ 2 --> true
        ret = Images.adjust_gamma(img, 0.3)
        @fact ret == img .^ 0.3 --> true
        ret = Images.adjust_gamma(img, 1, 1, 100)
        @fact ret == img --> true

        img = zeros(10, 10)
        ret = Images.adjust_gamma(img, 2)
        @fact all(ret .== 0) --> true

        a = ARGB(0.2, 0.3, 0.4, 0.5)    
        r = Images._gamma_pixel_rescale(a, 1)
        @fact isapprox(a, r, rtol = 0.0001) --> true
        r = Images._gamma_pixel_rescale(a, 2)
        @fact alpha(r) --> alpha(a)

        b = AGray(0.2, 0.6)    
        r = Images._gamma_pixel_rescale(b, 1)
        @fact b --> r
        r = Images._gamma_pixel_rescale(b, 2)
        @fact alpha(r) --> alpha(b)
        @fact isapprox(r.val, b.val ^ 2) --> true

    end

	context("Histogram Matching") do

        #DataTypes
        img = ones(Images.Gray{Float64}, 10, 10)
        ret = histmatch(img, img)
        @fact all(ret .== zero(eltype(img))) --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U8}, 10, 10)
        ret = histmatch(img, img)
        @fact all(ret .== zero(eltype(img))) --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.Gray{Images.U16}, 10, 10)
        ret = histmatch(img, img)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.AGray{Images.U8}, 10, 10)
        ret = histmatch(img, img)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U8}, 10, 10)
        ret = histmatch(img, img)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Images.U16}, 10, 10)
        ret = histmatch(img, img)
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.RGB{Float64}, 10, 10)
        ret = histmatch(img, img)
        @fact all(map((i, r) -> isapprox(zero(RGB), r, atol = 0.001), img, ret)) --> true
        @fact eltype(ret) == eltype(img) --> true

        img = ones(Images.ARGB{Images.U8}, 10, 10)
        ret = histmatch(img, img)
        @fact eltype(ret) == eltype(img) --> true

        img = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges, hist = imhist(img, 2)
        himg = Images._histmatch(img, edges, hist)
        @fact himg == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5] --> true
        edges, hist = imhist(img, 5)
        himg = Images._histmatch(img, edges, hist)
        @fact himg == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8] --> true
    end

    context("CLAHE") do

    end

    context("Other") do 

        # Issue #282
        img = convert(Images.Image{Gray{UFixed8}}, eye(2,2))
        imgs = Images.imstretch(img, 0.3, 0.4)
        @fact data(imgs) --> roughly(1./(1 + (0.3./(eye(2,2) + eps())).^0.4))

        img = convert(Images.Image{Gray{UFixed16}}, [0.01164 0.01118; 0.01036 0.01187])
        @fact data(imadjustintensity(img,[0.0103761, 0.0252166]))[2,1] --> 0.0
        @fact eltype(imadjustintensity(img)) --> Gray{UFixed16}

    end

end