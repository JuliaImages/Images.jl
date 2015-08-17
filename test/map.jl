using FactCheck, Images, Colors, FixedPointNumbers

macro chk(a, b)
    :(@fact ($a == $b && typeof($a) == typeof($b)) --> true)
end

facts("Map") do
    context("MapNone") do
        mapi = MapNone{Int}()
        @chk map(mapi, 7) 7
        @chk map(mapi, 0x07) 7
        a7 = Int[7]
        @chk map(mapi, [0x07]) a7
        @fact map(mapi, a7) --> exactly(a7)
        mapi = MapNone{RGB24}()
        g = Ufixed8(0.1)
        @chk Images.map1(mapi, 0.1) g
        @chk map(mapi, 0.1) RGB24(g,g,g)
        @chk map(mapi, Gray(0.1)) RGB24(g,g,g)
        @chk map(mapi, g) RGB24(g,g,g)
        @chk map(mapi, true) RGB24(1,1,1)
        @chk map(mapi, false) RGB24(0x00,0x00,0x00)
        mapi = MapNone{RGB{Float32}}()
        g = 0.1f0
        @chk Images.map1(mapi, 0.1) g
        @chk map(mapi, 0.1) RGB(g,g,g)

        c = RGB(0.1,0.2,0.3)
        mapi = MapNone{HSV{Float64}}()
        @chk map(mapi, c) convert(HSV, c)

        # issue #200
        c = RGBA{Ufixed8}(1,0.5,0.25,0.8)
        mapi = MapNone{Images.ColorTypes.BGRA{Ufixed8}}()
        @chk map(mapi, c) convert(Images.ColorTypes.BGRA{Ufixed8}, c)
    end

    context("BitShift") do
        mapi = BitShift{Uint8,7}()
        @chk map(mapi, 0xff) 0x01
        @chk map(mapi, 0xf0ff) 0xff
        @chk map(mapi, [0xff]) Uint8[0x01]
        mapi = BitShift{Ufixed8,7}()
        @chk map(mapi, 0xffuf8) 0x01uf8
        @chk map(mapi, 0xf0ffuf16) 0xffuf8
        mapi = BitShift{ARGB32,4}()
        @chk map(mapi, 0xffuf8) ARGB32(0xff0f0f0f)
        mapi = BitShift{RGB24,2}()
        @chk map(mapi, Gray(0xffuf8)) RGB24(0x003f3f3f)
        mapi = BitShift{ARGB32,2}()
        @chk map(mapi, Gray(0xffuf8)) ARGB32(0xff3f3f3f)
        @chk map(mapi, GrayA{Ufixed8}(Gray(0xffuf8),0x3fuf8)) ARGB32(0x0f3f3f3f)
        mapi = BitShift{RGB{Ufixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) RGB(0x3fuf8, 0x3fuf8, 0x3fuf8)
        mapi = BitShift{ARGB{Ufixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) ARGB{Ufixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0xffuf8)
        @chk map(mapi, GrayA{Ufixed8}(Gray(0xffuf8),0x3fuf8)) ARGB{Ufixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0x0fuf8)
        mapi = BitShift{RGBA{Ufixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) RGBA{Ufixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0xffuf8)
        @chk map(mapi, GrayA{Ufixed8}(Gray(0xffuf8),0x3fuf8)) RGBA{Ufixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0x0fuf8)
        mapi = BitShift(ARGB{Ufixed8}, 8)
        @chk map(mapi, RGB{Ufixed16}(1.0,0.8,0.6)) ARGB{Ufixed8}(1.0,0.8,0.6,1.0)
        mapi = BitShift(RGBA{Ufixed8}, 8)
        @chk map(mapi, RGB{Ufixed16}(1.0,0.8,0.6)) RGBA{Ufixed8}(1.0,0.8,0.6,1.0)
        # Issue, #269, IJulia issue #294
        bs = BitShift(Gray{Ufixed8}, 8)
        v = Gray(ufixed16(0.8))
        @chk map(bs, v) Gray{Ufixed8}(0.8)
    end

    context("Clamp") do
        mapi = ClampMin(Float32, 0.0)
        @chk map(mapi,  1.2) 1.2f0
        @chk map(mapi, -1.2) 0.0f0
        mapi = ClampMin(RGB24, 0.0f0)
        @chk map(mapi, RGB{Float32}(-5.3,0.4,0.8)) RGB24(0x000066cc)
        mapi = ClampMax(Float32, 1.0)
        @chk map(mapi,  1.2)  1.0f0
        @chk map(mapi, -1.2) -1.2f0
        mapi = ClampMax(RGB24, 1.0f0)
        @chk map(mapi, RGB{Float32}(0.2,1.3,0.8)) RGB24(0x0033ffcc)
        mapi = ClampMinMax(Float32, 0.0, 1.0)
        @chk map(mapi,  1.2) 1.0f0
        @chk map(mapi, -1.2) 0.0f0
        mapi = ClampMinMax(ARGB32, 0.0f0, 1.0f0)
        @chk map(mapi, RGB{Float32}(-0.2,1.3,0.8)) ARGB32(0xff00ffcc)
        @chk map(mapi, ARGB{Float32}(-0.2,1.3,0.8,0.6)) ARGB32(0x9900ffcc)
        mapi = Clamp(Float32)
        @chk map(mapi,  1.2) 1.0f0
        @chk map(mapi, -1.2) 0.0f0
        mapi = Clamp(Ufixed12)
        @chk map(mapi, Ufixed12(1.2)) one(Ufixed12)
        mapi = Clamp(Gray{Ufixed12})
        @chk map(mapi, Gray(Ufixed12(1.2))) Gray(one(Ufixed12))
        mapi = ClampMinMax(RGB24, 0.0, 1.0)
        @chk map(mapi, 1.2) RGB24(0x00ffffff)
        @chk map(mapi, 0.5) RGB24(0x00808080)
        @chk map(mapi, -.3) RGB24(0x00000000)
        mapi = ClampMinMax(RGB{Ufixed8}, 0.0, 1.0)
        @chk map(mapi, 1.2) RGB{Ufixed8}(1,1,1)
        @chk map(mapi, 0.5) RGB{Ufixed8}(0.5,0.5,0.5)
        @chk map(mapi, -.3) RGB{Ufixed8}(0,0,0)
        mapi = Clamp(RGB{Ufixed8})
        @chk map(mapi, RGB(1.2,0.5,-.3)) RGB{Ufixed8}(1,0.5,0)
        mapi = Clamp(ARGB{Ufixed8})
        @chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{Ufixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) ARGB{Ufixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, 0.2) ARGB{Ufixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(0.2),1.2)) ARGB{Ufixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(-.4),0.8)) ARGB{Ufixed8}(0.0,0.0,0.0,0.8)
        mapi = Clamp(RGBA{Ufixed8})
        @chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) RGBA{Ufixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{Ufixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, 0.2) RGBA{Ufixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(0.2),1.2)) RGBA{Ufixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(-.4),0.8)) RGBA{Ufixed8}(0.0,0.0,0.0,0.8)
        # Issue #253
        mapi = Clamp(BGRA{Ufixed8})
        @chk map(mapi, RGBA{Float32}(1.2,0.5,-.3,0.2)) BGRA{Ufixed8}(1.0,0.5,0.0,0.2)

        @chk clamp(RGB{Float32}(-0.2,0.5,1.8)) RGB{Float32}(0.0,0.5,1.0)
        @chk clamp(ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{Float64}(1.0,0.5,0.0,0.2)
        @chk clamp(RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{Float64}(1.0,0.5,0.0,0.2)
    end

    context("Issue #285") do
        a = [Gray(0xd0uf8)]
        a1 = 10*a
        mapi = mapinfo(Gray{Ufixed8}, a1)
        @chk map(mapi, a1[1]) Gray(0xffuf8)
    end

    context("ScaleMinMax") do
        mapi = ScaleMinMax(Ufixed8, 100, 1000)
        @chk map(mapi, 100) Ufixed8(0.0)
        @chk map(mapi, 1000) Ufixed8(1.0)
        @chk map(mapi, 10) Ufixed8(0.0)
        @chk map(mapi, 2000) Ufixed8(1.0)
        @chk map(mapi, 550) Ufixed8(0.5)
        mapinew = ScaleMinMax(Ufixed8, [100,500,1000])
        @fact mapinew --> mapi
        mapinew = ScaleMinMax(Ufixed8, [0,500,2000], convert(Uint16, 100), convert(Uint16, 1000))
        @fact mapinew --> mapi
        mapi = ScaleMinMax(ARGB32, 100, 1000)
        @chk map(mapi, 100) ARGB32(0,0,0,1)
        @chk map(mapi, 550) ARGB32(0x80uf8,0x80uf8,0x80uf8,0xffuf8)
        @chk map(mapi,2000) ARGB32(1,1,1,1)
        mapi = ScaleMinMax(RGB{Float32}, 100, 1000)
        @chk map(mapi,  50) RGB(0.0f0, 0.0f0, 0.0f0)
        @chk map(mapi, 550) RGB{Float32}(0.5, 0.5, 0.5)
        @chk map(mapi,2000) RGB(1.0f0, 1.0f0, 1.0f0)
        A = Gray{Ufixed8}[Ufixed8(0.1), Ufixed8(0.9)]
        @fact mapinfo(RGB24, A) --> MapNone{RGB24}()
        mapi = ScaleMinMax(RGB24, A, zero(Gray{Ufixed8}), one(Gray{Ufixed8}))
        @fact map(mapi, A) --> map(mapinfo(RGB24, A), A)
        mapi = ScaleMinMax(Float32, [Gray(one(Ufixed8))], 0, 1) # issue #180
        @chk map(mapi, Gray(Ufixed8(0.6))) 0.6f0
        @fact_throws ErrorException ScaleMinMax(Float32, 0, 0, 1.0) # issue #245
        A = [Gray{Float64}(0.2)]
        mapi = ScaleMinMax(RGB{Ufixed8}, A, 0.0, 0.2)
        @fact map(mapi, A) --> [RGB{Ufixed8}(1,1,1)]
    end

    context("ScaleSigned") do
        mapi = ScaleSigned(Float32, 1/5)
        @chk map(mapi, 7) 1.0f0
        @chk map(mapi, 5) 1.0f0
        @chk map(mapi, 3) convert(Float32, 3/5)
        @chk map(mapi, -3) convert(Float32, -3/5)
        @chk map(mapi, -6) -1.0f0
        mapi = ScaleSigned(RGB24, 1.0f0/10)
        @chk map(mapi, 12) RGB24(0x00ff00ff)
        @chk map(mapi, -10.0) RGB24(0x0000ff00)
        @chk map(mapi, 0) RGB24(0x00000000)
    end

    context("ScaleAutoMinMax") do
        mapi = ScaleAutoMinMax()
        A = [100,550,1000]
        @chk map(mapi, A) ufixed8([0.0,0.5,1.0])
        mapi = ScaleAutoMinMax(RGB24)
        @chk map(mapi, A) RGB24[0x00000000, 0x00808080, 0x00ffffff]

        # Issue #304
        imgr = colorim(rand(UInt16, 3, 2, 2))
        mi = ScaleAutoMinMax(RGB{Ufixed16})
        res = map(mi, imgr)
        @fact raw(res) --> raw(map(ScaleAutoMinMax(Ufixed16), raw(imgr)))
    end

    context("Scaling and ssd") do
        img = Images.grayim(fill(typemax(Uint16), 3, 3))
        mapi = Images.mapinfo(Ufixed8, img)
        img8 = map(mapi, img)
        @fact all(img8 .== typemax(Ufixed8)) --> true
        A = 0
        mnA, mxA = 1.0, -1.0
        while mnA > 0 || mxA < 0
            A = randn(3,3)
            mnA, mxA = extrema(A)
        end
        offset = 30.0
        img = convert(Images.Image, A .+ offset)
        mapi = Images.ScaleMinMax(Ufixed8, offset, offset+mxA, 1/mxA)
        imgs = map(mapi, img)
        @fact minimum(imgs) --> 0
        @fact maximum(imgs) --> 1
        @fact eltype(imgs) --> Ufixed8
        imgs = Images.imadjustintensity(img, [])
        @fact_throws ErrorException Images.imadjustintensity(img, [1])
        mnA = minimum(A)
        @fact Images.ssd(imgs, (A.-mnA)/(mxA-mnA)) --> less_than(eps())
        A = reshape(1:9, 3, 3)
        B = map(Images.ClampMin(Float32, 3), A)
        @fact (eltype(B) == Float32 && B == [3 4 7; 3 5 8; 3 6 9]) --> true
        B = map(Images.ClampMax(Uint8, 7), A)
        @fact (eltype(B) == Uint8 && B == [1 4 7; 2 5 7; 3 6 7]) --> true

        A = reinterpret(Ufixed8, [convert(UInt8,1):convert(UInt8,24);], (3, 2, 4))
        img = reinterpret(RGB{Ufixed8}, A, (2,4))
        @fact separate(img) --> permutedims(A, (2,3,1))
    end

    context("sc") do
        arr = zeros(4,4)
        arr[2,2] = 0.5
        @fact sc(arr)[2,2] --> 0xffuf8
        @fact sc(arr, 0.0, 0.75)[2,2] --> 0xaauf8
    end

    context("Color conversion") do
        gray = collect(linspace(0.0,1.0,5)) # a 1-dimensional image
        gray8 = round(Uint8, 255*gray)
        gray32 = Uint32[convert(UInt32, g)<<16 | convert(UInt32, g)<<8 | convert(UInt32, g) for g in gray8]
        imgray = Images.Image(gray, Dict{ASCIIString,Any}([("colordim",0), ("colorspace","Gray")]))
        buf = map(Images.mapinfo(Uint32, imgray), imgray) # Images.uint32color(imgray)
        @fact buf --> gray32
        rgb = RGB{Float64}[RGB(g, g, g) for g in gray]
        buf = map(Images.mapinfo(Uint32, rgb), rgb) # Images.uint32color(rgb)
        @fact buf --> gray32
        r = red(rgb)
        @fact r --> gray
        img = Images.Image(reinterpret(RGB24, gray32)) # , ["colordim"-->0, "colorspace"=>"RGB24"])
        buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        rgb = repeat(gray, outer=[1,3])
        img = Images.Image(rgb, Dict{ASCIIString,Any}([("colordim",2), ("colorspace","RGB"), ("spatialorder",["x"])]))
        buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        g = green(img)
        @fact g --> gray
        rgb = repeat(gray', outer=[3,1])
        img = Images.Image(rgb, Dict{ASCIIString,Any}([("colordim",1), ("colorspace","RGB"), ("spatialorder",["x"])]))
        buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        b = blue(img)
        @fact b --> gray
    end

    context("Map and indexed images") do
        img = Images.ImageCmap([1 2 3; 3 2 1], [RGB{Ufixed16}(1.0,0.6,0.4), RGB{Ufixed16}(0.2, 0.4, 0.6), RGB{Ufixed16}(0.5,0.5,0.5)])
        mapi = MapNone(RGB{Ufixed8})
        imgd = map(mapi, img)
        cmap = [RGB{Ufixed8}(1.0,0.6,0.4), RGB{Ufixed8}(0.2, 0.4, 0.6), RGB{Ufixed8}(0.5,0.5,0.5)]
        @fact imgd --> reshape(cmap[[1,3,2,2,3,1]], (2,3))
    end
end
