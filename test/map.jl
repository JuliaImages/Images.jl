using FactCheck, Images, Colors, FixedPointNumbers
using Compat; import Compat.String

macro chk(a, b)
    :(@fact ($a == $b && typeof($a) == typeof($b)) --> true)
end

macro chk_approx(a, b)
    :(@fact (abs($a - $b) < 2*(eps($a)+eps($b)) && typeof($a) == typeof($b)) --> true)
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
        g = UFixed8(0.1)
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
        c = RGBA{UFixed8}(1,0.5,0.25,0.8)
        mapi = MapNone{Images.ColorTypes.BGRA{UFixed8}}()
        @chk map(mapi, c) convert(Images.ColorTypes.BGRA{UFixed8}, c)
    end

    context("BitShift") do
        mapi = BitShift{UInt8,7}()
        @chk map(mapi, 0xff) 0x01
        @chk map(mapi, 0xf0ff) 0xff
        @chk map(mapi, [0xff]) UInt8[0x01]
        mapi = BitShift{UFixed8,7}()
        @chk map(mapi, 0xffuf8) 0x01uf8
        @chk map(mapi, 0xf0ffuf16) 0xffuf8
        mapi = BitShift{ARGB32,4}()
        @chk map(mapi, 0xffuf8) ARGB32(0xff0f0f0f)
        mapi = BitShift{RGB24,2}()
        @chk map(mapi, Gray(0xffuf8)) RGB24(0x003f3f3f)
        mapi = BitShift{ARGB32,2}()
        @chk map(mapi, Gray(0xffuf8)) ARGB32(0xff3f3f3f)
        @chk map(mapi, GrayA{UFixed8}(Gray(0xffuf8),0x3fuf8)) ARGB32(0x0f3f3f3f)
        mapi = BitShift{RGB{UFixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) RGB(0x3fuf8, 0x3fuf8, 0x3fuf8)
        mapi = BitShift{ARGB{UFixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) ARGB{UFixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0xffuf8)
        @chk map(mapi, GrayA{UFixed8}(Gray(0xffuf8),0x3fuf8)) ARGB{UFixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0x0fuf8)
        mapi = BitShift{RGBA{UFixed8},2}()
        @chk map(mapi, Gray(0xffuf8)) RGBA{UFixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0xffuf8)
        @chk map(mapi, GrayA{UFixed8}(Gray(0xffuf8),0x3fuf8)) RGBA{UFixed8}(0x3fuf8, 0x3fuf8, 0x3fuf8, 0x0fuf8)
        mapi = BitShift(ARGB{UFixed8}, 8)
        @chk map(mapi, RGB{UFixed16}(1.0,0.8,0.6)) ARGB{UFixed8}(1.0,0.8,0.6,1.0)
        mapi = BitShift(RGBA{UFixed8}, 8)
        @chk map(mapi, RGB{UFixed16}(1.0,0.8,0.6)) RGBA{UFixed8}(1.0,0.8,0.6,1.0)
        # Issue, #269, IJulia issue #294
        bs = BitShift(Gray{UFixed8}, 8)
        v = Gray(ufixed16(0.8))
        @chk map(bs, v) Gray{UFixed8}(0.8)
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
        mapi = Clamp(UFixed12)
        @chk map(mapi, UFixed12(1.2)) one(UFixed12)
        mapi = Clamp(Gray{UFixed12})
        @chk map(mapi, Gray(UFixed12(1.2))) Gray(one(UFixed12))
        mapi = ClampMinMax(RGB24, 0.0, 1.0)
        @chk map(mapi, 1.2) RGB24(0x00ffffff)
        @chk map(mapi, 0.5) RGB24(0x00808080)
        @chk map(mapi, -.3) RGB24(0x00000000)
        mapi = ClampMinMax(RGB{UFixed8}, 0.0, 1.0)
        @chk map(mapi, 1.2) RGB{UFixed8}(1,1,1)
        @chk map(mapi, 0.5) RGB{UFixed8}(0.5,0.5,0.5)
        @chk map(mapi, -.3) RGB{UFixed8}(0,0,0)
        mapi = Clamp(RGB{UFixed8})
        @chk map(mapi, RGB(1.2,0.5,-.3)) RGB{UFixed8}(1,0.5,0)
        mapi = Clamp(ARGB{UFixed8})
        @chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{UFixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) ARGB{UFixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, 0.2) ARGB{UFixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(0.2),1.2)) ARGB{UFixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(-.4),0.8)) ARGB{UFixed8}(0.0,0.0,0.0,0.8)
        mapi = Clamp(RGBA{UFixed8})
        @chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) RGBA{UFixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{UFixed8}(1.0,0.5,0.0,0.2)
        @chk map(mapi, 0.2) RGBA{UFixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(0.2),1.2)) RGBA{UFixed8}(0.2,0.2,0.2,1.0)
        @chk map(mapi, GrayA{Float32}(Gray(-.4),0.8)) RGBA{UFixed8}(0.0,0.0,0.0,0.8)
        # Issue #253
        mapi = Clamp(BGRA{UFixed8})
        @chk map(mapi, RGBA{Float32}(1.2,0.5,-.3,0.2)) BGRA{UFixed8}(1.0,0.5,0.0,0.2)

        @chk clamp(RGB{Float32}(-0.2,0.5,1.8)) RGB{Float32}(0.0,0.5,1.0)
        @chk clamp(ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{Float64}(1.0,0.5,0.0,0.2)
        @chk clamp(RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{Float64}(1.0,0.5,0.0,0.2)
    end

    context("Issue #285") do
        a = [Gray(0xd0uf8)]
        a1 = 10*a
        mapi = mapinfo(Gray{UFixed8}, a1)
        @chk map(mapi, a1[1]) Gray(0xffuf8)
    end

    context("ScaleMinMax") do
        mapi = ScaleMinMax(UFixed8, 100, 1000)
        @chk map(mapi, 100) UFixed8(0.0)
        @chk map(mapi, 1000) UFixed8(1.0)
        @chk map(mapi, 10) UFixed8(0.0)
        @chk map(mapi, 2000) UFixed8(1.0)
        @chk map(mapi, 550) UFixed8(0.5)
        mapinew = ScaleMinMax(UFixed8, [100,500,1000])
        @fact mapinew --> mapi
        mapinew = ScaleMinMax(UFixed8, [0,500,2000], convert(UInt16, 100), convert(UInt16, 1000))
        @fact mapinew --> mapi
        mapi = ScaleMinMax(ARGB32, 100, 1000)
        @chk map(mapi, 100) ARGB32(0,0,0,1)
        @chk map(mapi, 550) ARGB32(0x80uf8,0x80uf8,0x80uf8,0xffuf8)
        @chk map(mapi,2000) ARGB32(1,1,1,1)
        mapi = ScaleMinMax(RGB{Float32}, 100, 1000)
        @chk map(mapi,  50) RGB(0.0f0, 0.0f0, 0.0f0)
        @chk map(mapi, 550) RGB{Float32}(0.5, 0.5, 0.5)
        @chk map(mapi,2000) RGB(1.0f0, 1.0f0, 1.0f0)
        A = Gray{UFixed8}[UFixed8(0.1), UFixed8(0.9)]
        @fact mapinfo(RGB24, A) --> MapNone{RGB24}()
        mapi = ScaleMinMax(RGB24, A, zero(Gray{UFixed8}), one(Gray{UFixed8}))
        @fact map(mapi, A) --> map(mapinfo(RGB24, A), A)
        mapi = ScaleMinMax(Float32, [Gray(one(UFixed8))], 0, 1) # issue #180
        @chk map(mapi, Gray(UFixed8(0.6))) 0.6f0
        @fact_throws ErrorException ScaleMinMax(Float32, 0, 0, 1.0) # issue #245
        A = [Gray{Float64}(0.2)]
        mapi = ScaleMinMax(RGB{UFixed8}, A, 0.0, 0.2)
        @fact map(mapi, A) --> [RGB{UFixed8}(1,1,1)]
        mapi = ScaleMinMax(Gray{U8}, Gray{U8}(0.2), Gray{U8}(0.4))
        @fact Gray{U8}(0.49) <= map(mapi, Gray{U8}(0.3)) <= Gray{U8}(0.5) --> true
        @fact Gray{U8}(0.49) <= map(mapi, 0.3) <= Gray{U8}(0.5) --> true
        mapi = ScaleMinMax(Gray{U8}, 0.2, 0.4)
        @fact Gray{U8}(0.49) <= map(mapi, Gray{U8}(0.3)) <= Gray{U8}(0.5) --> true
        @fact Gray{U8}(0.49) <= map(mapi, 0.3) <= Gray{U8}(0.5) --> true

        A = [-0.5 0.5; Inf NaN]
        @fact map(Clamp01NaN(A), A) --> [0.0 0.5; 1.0 0.0]
        B = colorim(repeat(reshape(A, (1,2,2)), outer=[3,1,1]))
        @fact map(Clamp01NaN(B), B) --> [RGB(0.0,0,0) RGB(0.5,0.5,0.5); RGB(1.0,1,1) RGB(0.0,0,0)]
        # Integer-valued images are not recommended, but let's at
        # least make sure they work
        smm = ScaleMinMax(UInt8, 0.0, 1.0, 255)
        @fact map(smm, 0.0) --> exactly(0x00)
        @fact map(smm, 1.0) --> exactly(0xff)
        @fact map(smm, 0.1) --> exactly(round(UInt8, 0.1*255.0f0))
        smm = ScaleMinMax(Gray{U8}, typemin(Int8), typemax(Int8))
        @fact map(smm, 2) --> Gray{U8}(0.51)
        smm = ScaleMinMax(RGB24, typemin(Int8), typemax(Int8))
        @fact map(smm, 2) --> RGB24(0x828282)
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
        A = rand(UInt16, 3, 2, 2)
        imgr = colorim(A)
        mi1 = ScaleAutoMinMax(RGB{UFixed16})
        res1 = raw(map(mi1, imgr))
        mi2 = ScaleAutoMinMax(UFixed16)
        res2 = raw(map(mi2, raw(imgr)))
        # @fact res1 --> res2
        # Note: this fails occassionally. Reproduce it with
        #    s = 1.1269798f0
        #    val = 0xdeb5
        #    UFixed16(s*UFixed16(val,0)) == UFixed16((s/typemax(UInt16))*val)
        @fact maxabs(convert(Array{Int32}, res1) - convert(Array{Int32}, res2)) --> less_than_or_equal(1)
    end

    context("Scaling and ssd") do
        img = Images.grayim(fill(typemax(UInt16), 3, 3))
        mapi = Images.mapinfo(UFixed8, img)
        img8 = map(mapi, img)
        @fact all(img8 .== typemax(UFixed8)) --> true
        A = 0
        mnA, mxA = 1.0, -1.0
        while mnA > 0 || mxA < 0
            A = randn(3,3)
            mnA, mxA = extrema(A)
        end
        offset = 30.0
        img = convert(Images.Image, A .+ offset)
        mapi = Images.ScaleMinMax(UFixed8, offset, offset+mxA, 1/mxA)
        imgs = map(mapi, img)
        @fact minimum(imgs) --> 0
        @fact maximum(imgs) --> 1
        @fact eltype(imgs) --> UFixed8
        imgs = Images.imadjustintensity(img)
        @fact_throws MethodError Images.imadjustintensity(img, [1])
        mnA = minimum(A)
        @fact Images.ssd(imgs, (A.-mnA)/(mxA-mnA)) --> less_than(eps(UFixed16))
        A = reshape(1:9, 3, 3)
        B = map(Images.ClampMin(Float32, 3), A)
        @fact (eltype(B) == Float32 && B == [3 4 7; 3 5 8; 3 6 9]) --> true
        B = map(Images.ClampMax(UInt8, 7), A)
        @fact (eltype(B) == UInt8 && B == [1 4 7; 2 5 7; 3 6 7]) --> true

        A = reinterpret(UFixed8, [convert(UInt8,1):convert(UInt8,24);], (3, 2, 4))
        img = reinterpret(RGB{UFixed8}, A, (2,4))
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
        gray8 = round(UInt8, 255*gray)
        gray32 = UInt32[convert(UInt32, g)<<16 | convert(UInt32, g)<<8 | convert(UInt32, g) for g in gray8]
        imgray = Images.Image(gray, Dict{String,Any}([("colordim",0), ("colorspace","Gray")]))
        buf = map(Images.mapinfo(UInt32, imgray), imgray) # Images.uint32color(imgray)
        @fact buf --> gray32
        rgb = RGB{Float64}[RGB(g, g, g) for g in gray]
        buf = map(Images.mapinfo(UInt32, rgb), rgb) # Images.uint32color(rgb)
        @fact buf --> gray32
        r = red(rgb)
        @fact r --> gray
        img = Images.Image(reinterpret(RGB24, gray32)) # , ["colordim"-->0, "colorspace"=>"RGB24"])
        buf = map(Images.mapinfo(UInt32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        rgb = repeat(gray, outer=[1,3])
        img = Images.Image(rgb, Dict{String,Any}([("colordim",2), ("colorspace","RGB"), ("spatialorder",["x"])]))
        buf = map(Images.mapinfo(UInt32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        g = green(img)
        @fact g --> gray
        rgb = repeat(gray', outer=[3,1])
        img = Images.Image(rgb, Dict{String,Any}([("colordim",1), ("colorspace","RGB"), ("spatialorder",["x"])]))
        buf = map(Images.mapinfo(UInt32, img), img) # Images.uint32color(img)
        @fact buf --> gray32
        b = blue(img)
        @fact b --> gray
    end

    context("Map and indexed images") do
        img = Images.ImageCmap([1 2 3; 3 2 1], [RGB{UFixed16}(1.0,0.6,0.4), RGB{UFixed16}(0.2, 0.4, 0.6), RGB{UFixed16}(0.5,0.5,0.5)])
        mapi = MapNone(RGB{UFixed8})
        imgd = map(mapi, img)
        cmap = [RGB{UFixed8}(1.0,0.6,0.4), RGB{UFixed8}(0.2, 0.4, 0.6), RGB{UFixed8}(0.5,0.5,0.5)]
        @fact imgd --> reshape(cmap[[1,3,2,2,3,1]], (2,3))
    end
end
