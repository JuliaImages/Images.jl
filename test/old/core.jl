using FactCheck, Images, Colors, FixedPointNumbers, IndirectArrays, AxisArrays
using Compat
using Compat.view

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end

const scalar_getindex_new = VERSION >= v"0.5.0-dev+1195"

facts("Core") do
    a = rand(3,3)
    @inferred(Image(a))
    # support integer-valued types, but these are NOT recommended (use UFixed)
    B = rand(UInt16(1):UInt16(20), 3, 5)
    # img, imgd, and imgds will be used in many more tests
    # Thus, these must be defined as local if reassigned in any context() blocks
    cmap = reinterpret(RGB, repmat(reinterpret(UFixed8, round(UInt8, linspace(12, 255, 20)))', 3, 1))
    imgi = IndirectArray(copy(B), cmap)
    img = AxisArray(imgi, Axis{:y}(2*(1:size(imgi,1))), Axis{:x}(3*(1:size(imgi,2))))
    imgd0 = convert(Array, imgi)
    if testing_units
        imgd = AxisArray(imgd0,
                         Axis{:y}(2mm:2mm:2*size(imgi,1)*mm),
                         Axis{:x}(3mm:3mm:3*size(imgi,2)*mm))
    else
        imgd = AxisArray(imgd0, axes(img)...)
    end
    imgds = separate(imgd)

    context("Constructors of Image types") do
        @fact colorspace(B) --> "Gray" "test h42DSP"
        @fact colordim(B) --> 0 "test 4ZQwBT"
        let img = colorview(RGB, ufixedview(UFixed{UInt16,8}, B))
            @fact colorspace(img) --> "RGB" "test THoeTd"
            @fact colordim(img) --> 0 "test Zm2gDZ"
            img = grayim(B)
            @fact colorspace(img) --> "Gray" "test SIFbZo"
            @fact colordim(B) --> 0 "test HAdcG0"
            @fact grayim(img) --> img "test mJppbD"
            Bf = grayim(round(UInt8, B))
            @fact eltype(Bf) --> Gray{UFixed8} "test luvkCN"
            @fact colorspace(Bf) --> "Gray" "test JwE99B"
            @fact colordim(Bf) --> 0 "test DNzikz"
            Bf = grayim(B)
            @fact eltype(Bf) --> Gray{UFixed16} "test CPd4Fa"
            # colorspace encoded as a Color (enables multiple dispatch)
            BfCV = reinterpret(Gray{UFixed8}, round(UInt8, B))
            @fact colorspace(BfCV) --> "Gray" "test Tp3xdg"
            @fact colordim(BfCV) --> 0 "test aALUCW"
            Bf3 = grayim(reshape(collect(convert(UInt8,1):convert(UInt8,36)), 3,4,3))
            @fact eltype(Bf3) --> Gray{UFixed8} "test k8hBgR"
            Bf3 = grayim(reshape(collect(convert(UInt16,1):convert(UInt16,36)), 3,4,3))
            @fact eltype(Bf3) --> Gray{UFixed16} "test KASRod"
            Bf3 = grayim(reshape(collect(1.0f0:36.0f0), 3,4,3))
            @fact eltype(Bf3) --> Gray{Float32} "test Vr9iDW"
        end
    end

    context("Colorim") do
        C = colorim(rand(UInt8, 3, 5, 5))
        @fact eltype(C) --> RGB{UFixed8} "test N0iGXo"
        @fact colordim(C) --> 0 "test 6COaeI"
        @fact colorim(C) --> C "test hPgSHp"
        C = colorim(rand(UInt16, 4, 5, 5), "ARGB")
        @fact eltype(C) --> ARGB{UFixed16} "test PZ5Ld5"
        C = colorim(rand(UInt8(1):UInt8(20), 3, 5, 5))
        @fact eltype(C) --> RGB{U8} "test MyDpnz"
        @fact colordim(C) --> 0 "test vu1pXs"
        @fact colorspace(C) --> "RGB" "test moBy0x"
        @fact eltype(colorim(rand(UInt16, 3, 5, 5))) --> RGB{UFixed16} "test pzSy1a"
        @fact eltype(colorim(rand(3, 5, 5))) --> RGB{Float64} "test F3ex39"
        @fact colordim(colorim(rand(UInt8, 5, 5, 3))) --> 0 "test kElkKH"
        @fact spatialorder(colorim(rand(UInt8, 3, 5, 5))) --> (:y, :x) "test auwSni"
        @fact spatialorder(colorim(rand(UInt8, 5, 5, 3))) --> (:y, :x) "test g307S1"
        @fact eltype(colorim(rand(UInt8, 4, 5, 5), "RGBA")) --> RGBA{UFixed8} "test RcSRnh"
        @fact eltype(colorim(rand(UInt8, 4, 5, 5), "ARGB")) --> ARGB{UFixed8} "test 9hscsz"
        @fact colordim(colorim(rand(UInt8, 5, 5, 4), "RGBA")) --> 0 "test SJTuxu"
        @fact colordim(colorim(rand(UInt8, 5, 5, 4), "ARGB")) --> 0 "test lIpIVN"
        @fact spatialorder(colorim(rand(UInt8, 5, 5, 4), "ARGB")) --> (:y, :x) "test y0Vpv4"
        @fact_throws ErrorException colorim(rand(UInt8, 3, 5, 3)) "test zCDA2C"
        @fact_throws ErrorException colorim(rand(UInt8, 4, 5, 5)) "test evgCjW"
        @fact_throws ErrorException colorim(rand(UInt8, 5, 5, 4)) "test ByYXn6"
        @fact_throws ErrorException colorim(rand(UInt8, 4, 5, 4), "ARGB") "test E8cclV"
        @fact_throws ErrorException colorim(rand(UInt8, 5, 5, 5), "foo") "test JHb2sd"
        @fact_throws ErrorException colorim(rand(UInt8, 2, 2, 2), "bar") "test eVybiQ"
    end

    context("Indexed color") do
        let cmap = linspace(RGB(0x0cuf8, 0x00uf8, 0x00uf8), RGB(0xffuf8, 0x00uf8, 0x00uf8), 20)
            img_ = ImageCmap(copy(B), cmap)
            @fact colorspace(img_) --> "RGB" "test ekvCpM"
            img_ = ImageCmap(copy(B), cmap)
            @fact colorspace(img_) --> "RGB" "test F3LOhS"
            # Note: img from opening of facts() block
            # TODO: refactor whole block
            @fact eltype(img) --> RGB{UFixed8} "test 8aSK8W"
            @fact eltype(imgd) --> RGB{UFixed8} "test IhyuSH"
        end
    end


    context("Basic information") do
        @fact size(img) --> (3,5) "test SVoECb"
        @fact size(imgd) --> (3,5) "test GEziWW"
        @fact ndims(img) --> 2 "test uD5ST6"
        @fact ndims(imgd) --> 2 "test Gfuga9"
        @fact size(img,"y") --> 3 "test 6tVTzS"
        @fact size(img,"x") --> 5 "test WYg5qs"
        @fact strides(img) --> (1,3) "test lH4UyT"
        @fact strides(imgd) --> (1,3) "test Rvvapy"
        @fact strides(imgds) --> (1,3,15) "test dYffuB"
        @fact nimages(img) --> 1 "test jE9DjL"
        @fact ncolorelem(img) --> 1 "test W1Jzs4"
        @fact ncolorelem(imgd) --> 1 "test 76okNe"
        @fact ncolorelem(imgds) --> 1 "test xtbHDi"
        vimg = vec(img)
        @fact haskey(vimg, "spatialorder") --> false "test Jvo4UH"
        @fact haskey(vimg, "pixelspacing") --> false "test D37jeN"
        vimgds = vec(imgds)
        @fact haskey(vimg, "colordim") --> false "test nYSqMQ"
        @fact haskey(vimg, "colorspace") --> false "test OLYptD"
        @fact sort(vimgds)[1] --> minimum(imgds) "test gTnGfY"
    end

    context("1-dimensional images") do
        @fact colordim(1:5) --> 0 "test ZzuntI"
        @fact nimages(1:5) --> 1 "test qATU2x"
        @fact colorspace(1:5) --> "Gray" "test urhFyz"
        img1 = AxisArray(map(Gray, 0.1:0.1:0.5), :z)
        @fact colordim(img1) --> 0 "test w9LJFP"
        @fact colorspace(img1) --> "Gray" "test bTdx4f"
        @fact sdims(img1) --> 1 "test fImXGd"
        @fact convert(Vector{Gray{Float64}}, img1) --> map(Gray, 0.1:0.1:0.5) "test sx0MrI"
        @fact size(img1, "z") --> 5 "test UVd1Qc"
        @fact_throws ErrorException size(img1, "x") "test w99nLd"
    end

    context("Printing") do
        iob = IOBuffer()
        show(iob, img)
        show(iob, imgd)
    end

    context("Copy / similar") do
        A = randn(3,5,3)
        imgc = @inferred(copy(img))
        @fact imgc.data --> img.data "test bmmvpl"
        imgc = copyproperties(imgd, A)
        @fact imgc --> A "test FtSoza"
        img2 = @inferred(similar(img))
        @fact isa(img2, AxisArray) --> true "test ZQlGw2"
        @fact (img2.data == img.data) --> false "test Fis1BF"
        img2 = @inferred(similar(imgd))
        @fact isa(img2, typeof(imgd)) --> true "test kWPhBO"
        img2 = similar(img, (4,4))
        @fact size(img2) --> (4,4) "test XM2fWe"
        img2 = similar(imgd, (3,4,4))
        @fact size(img2) --> (3,4,4) "test CDWvHz"
        @fact copyproperties(B, A) --> A "test JQUH5M"
        @fact shareproperties(A, B) --> B "test xfed58"
        @fact shareproperties(img, B) --> B "test gvgFZu"
    end

    context("Properties") do
        @fact colorspace(img) --> "RGB" "test r4GQh5"
        @fact colordim(img) --> 0 "test P9R46B"
        @fact colordim(imgds) --> 3 "test 7Cmhyv"
        @fact timedim(img) --> 0 "test UxEzff"
        @fact pixelspacing(img) --> (2, 3) "test xxx4JY"
        @fact spacedirections(img) --> ((2, 0), (0, 3)) "test BMHiEE"
        if testing_units
            @fact pixelspacing(imgd) --> (2mm, 3mm) "test Io4D8M"
            @fact spacedirections(imgd) --> ((2mm, 0mm), (0mm, 3mm)) "test 1yJxFQ"
        end
    end

    context("Dims and ordering") do
        @fact sdims(img) --> sdims(imgd) "test qbjpWy"
        @fact coords_spatial(img) --> coords_spatial(imgd) "test UBaUrc"
        @fact size_spatial(img) --> size_spatial(imgd) "test Z6fpfp"
        A = randn(3,5,3)

        @fact storageorder(img) --> (Symbol.(Images.yx)...,) "test FbZZeV"
        @fact storageorder(imgds) --> (Symbol.(Images.yx)..., :color) "test PthQLh"
    end

    context("Sub / slice") do
        s = view(img, 2:2, 1:4)
        @fact ndims(s) --> 2 "test zKaJui"
        @fact sdims(s) --> 2 "test UZP9Ss"
        @fact size(s) --> (1,4) "test Fj3Elk"
        @fact data(s) --> cmap[view(B, 2:2, 1:4)] "test MEE0Kf"
        if scalar_getindex_new
            s = getindexim(img, 2, 1:4)
            @fact ndims(s) --> 1 "test oUbfko"
            @fact sdims(s) --> 1 "test gGEDEE"
            @fact size(s) --> (4,) "test iiUIN1"
            @fact data(s) --> cmap[B[2, 1:4]] "test J3Eh9N"
            s = getindexim(img, 2:2, 1:4)
            @fact ndims(s) --> 2 "test TqlvGx"
            @fact sdims(s) --> 2 "test m1UOXa"
            @fact size(s) --> (1,4) "test Y28Rxj"
            @fact data(s) --> cmap[B[2:2, 1:4]] "test w2w6dq"
        else
            s = getindexim(img, 2, 1:4)
            @fact ndims(s) --> 2 "test HucgqQ"
            @fact sdims(s) --> 2 "test OPRJZ2"
            @fact size(s) --> (1,4) "test rKuBuM"
            @fact data(s) --> B[2, 1:4] "test CBcZSh"
        end
        s = subim(img, 2, 1:4)
        @fact ndims(s) --> 2 "test u6Ga7f"
        @fact sdims(s) --> 2 "test RDsYNq"
        @fact size(s) --> (1,4) "test L390gv"
        s = subim(img, 2, [1,2,4])
        @fact ndims(s) --> 2 "test qvMWQI"
        @fact sdims(s) --> 2 "test sEei68"
        @fact size(s) --> (1,3) "test pRQ45S"
        s = sliceim(img, 2, 1:4)
        @fact ndims(s) --> 1 "test fAmwu6"
        @fact sdims(s) --> 1 "test hfxc7J"
        @fact size(s) --> (4,) "test 6xHsFS"
        s = sliceim(img, 2, [1,2,4])
        @fact ndims(s) --> 1 "test td0QjR"
        @fact sdims(s) --> 1 "test akcJpS"
        @fact size(s) --> (3,) "test AUktx0"
        s = sliceim(imgds, 2, 1:4, 1:3)
        @fact ndims(s) --> 2 "test 2tJFtL"
        @fact sdims(s) --> 1 "test SYKGkr"
        @fact colordim(s) --> 2 "test lPN4QA"
        @fact spatialorder(s) --> (:x,) "test 2OoCHQ"
        s = sliceim(imgds, 2, :, 1:3)
        @fact ndims(s) --> 2 "test rfr9cD"
        @fact sdims(s) --> 1 "test IH9aUL"
        @fact colordim(s) --> 2 "test wamZwZ"
        @fact spatialorder(s) --> (:x,) "test 6gJq01"
        s = sliceim(imgds, 2:2, 1:4, 1:3)
        @fact ndims(s) --> 3 "test Kxj25e"
        @fact sdims(s) --> 2 "test L1qDs3"
        @fact colordim(s) --> 3 "test BGZevb"
        @fact colorspace(s) --> "Gray" "test U8NVOG"
        s = getindexim(imgds, 2:2, 1:4, 2)
        @fact ndims(s) --> 2 "test nKN91R"
        @fact sdims(s) --> 2 "test SFKKhD"
        @fact colordim(s) --> 0 "test pu5AHT"
        @fact colorspace(s) --> "Gray" "test YUBU3N"
        s = sliceim(imgds, 2:2, 1:4, 2)
        @fact ndims(s) --> 2 "test 9jeHx1"
        @fact sdims(s) --> 2 "test uDCJHl"
        @fact colordim(s) --> 0 "test 30wLdG"
        @fact colorspace(s) --> "Gray" "test KTknYZ"
        s = sliceim(imgds, 2:2, 1:4, 1:2)
        @fact ndims(s) --> 3 "test LxgrvE"
        @fact sdims(s) --> 2 "test Wbsn5j"
        @fact colordim(s) --> 3 "test 0R4CXY"
        @fact colorspace(s) --> "Gray" "test 3o1uhP"
        @fact spatialorder(s) --> (:y,:x) "test 9NoVQt"
        s = view(img, "y", 2:2)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact size(s) --> (1,5)
        s = view(img, "y", 2)
        @fact ndims(s) --> 1
        @fact size(s) --> (5,)
        @fact size(getindexim(imgds, :, 1:2, :)) --> (size(imgds,1), 2, 3) "test Rfvge1"

        s = permutedims(imgds, (3,1,2))
        @fact colordim(s) --> 1 "test 8Jv9n7"
        ss = getindexim(s, 2:2, :, :)
        @fact colorspace(ss) --> "Gray" "test BEPoQi"
        @fact colordim(ss) --> 1 "test hlloiv"
        sss = squeeze(ss, 1)
        @fact colorspace(ss) --> "Gray" "test 6NNgi9"
        @fact colordim(sss) --> 0 "test pwQRnh"
        ss = getindexim(imgds, 2:2, :, :)
        @fact colordim(ss) --> 3 "test jx31ut"
        @fact spatialorder(ss) --> (:y, :x) "test u11jIc"
        sss = squeeze(ss, 1)
        @fact colordim(sss) --> 2 "test 4X0Hjv"
        @fact spatialorder(sss) --> (:x,) "test d5hRJs"
    end

    context("Named indexing") do
        @fact dimindex(imgds, "color") --> 3 "test EINbuA"
        @fact dimindex(imgds, "y") --> 1 "test F4E3tQ"
        @fact dimindex(imgds, "z") --> 0 "test 6u1u4T"
        imgdp = permutedims(imgds, [3,1,2])
        @fact dimindex(imgdp, "y") --> 2 "test U3pebL"
        @fact img["y", 2, "x", 4] --> imgi[2,4] "test oJ519u"
        @fact img["x", 4, "y", 2] --> imgi[2,4] "test oCsNbf"
        chan = imgds["color", 2]
        Blookup = reshape(green(cmap[B[:]]), size(B))
        @fact chan --> Blookup "test 2cKtWB"
    end

    context("Spatial order, width/ height, and permutations") do
        @fact widthheight(imgds) --> (5,3) "test tgY2FG"
        imgp = permutedims(imgds, ["x", "y", "color"])
        @fact imgp.data --> permutedims(imgds.data, [2,1,3]) "test MobXxa"
        imgp = permutedims(imgds, ("color", "x", "y"))
        @fact imgp.data --> permutedims(imgds.data, [3,2,1]) "test 1XMuxB"
        if testing_units
            @fact pixelspacing(imgp) --> (3mm, 2mm) "test 2XyBNB"
        end
        imgc = ImageMeta(copy(imgds))
        imgc["spacedirections"] = spacedirections(imgc)
        imgp = permutedims(imgc, ["x", "y", "color"])
        if testing_units
            @fact spacedirections(imgp) --> ((0mm, 3mm),(2mm, 0mm)) "test NUTHJC"
            @fact pixelspacing(imgp) --> (3mm, 2mm) "test 5RGPJ0"
        end
    end

    context("Reinterpret, separate, more convert") do
        a = RGB{Float64}[RGB(1,1,0)]
        af = reinterpret(Float64, a)
        @fact vec(af) --> [1.0,1.0,0.0] "test M3uyLv"
        @fact size(af) --> (3,1) "test pLunxL"
        @fact_throws DimensionMismatch reinterpret(Float32, a) "test 2GvyKZ"
        anew = reinterpret(RGB, af)
        @fact anew --> a "test ekkFD4"
        anew = reinterpret(RGB, vec(af))
        @fact anew[1] --> a[1] "test FLyINs"
        @fact ndims(anew) --> 0 "test JQCoAo"
        anew = reinterpret(RGB{Float64}, af)
        @fact anew --> a "test VU6f3n"
        @fact_throws DimensionMismatch reinterpret(RGB{Float32}, af) "test 86GKXq"
        Au8 = rand(0x00:0xff, 3, 5, 4)
        A8 = reinterpret(UFixed8, Au8)
        rawrgb8 = reinterpret(RGB, A8)
        @fact eltype(rawrgb8) --> RGB{UFixed8} "test U5YnpG"
        @fact reinterpret(UFixed8, rawrgb8) --> A8 "test VIsSBT"
        @fact reinterpret(UInt8, rawrgb8) --> Au8 "test cWVSZ5"
        rawrgb32 = convert(Array{RGB{Float32}}, rawrgb8)
        @fact eltype(rawrgb32) --> RGB{Float32} "test BPX0oN"
        @fact ufixed8(rawrgb32) --> rawrgb8 "test 10hrUB"
        @fact reinterpret(UFixed8, rawrgb8) --> A8 "test xz6V7Y"
        imrgb8 = convert(Image, rawrgb8)
        @fact spatialorder(imrgb8) --> (Symbol.(Images.yx)...,) "test up03c2"
        @fact convert(Image, imrgb8) --> exactly(imrgb8) "test CaRwd8"
        @fact convert(Image{RGB{UFixed8}}, imrgb8) --> exactly(imrgb8) "test 39AZU3"
        im8 = reinterpret(UFixed8, imrgb8)
        @fact data(im8) --> A8 "test lZSAH9"
        @fact permutedims(ufixedview(U8, separate(imrgb8)), (3, 1, 2)) --> im8 "test zDOWZM"
        @fact reinterpret(UInt8, imrgb8) --> Au8 "test HeezpR"
        @fact reinterpret(RGB, im8) --> imrgb8 "test VJUpj3"
        ims8 = separate(imrgb8)
        @fact colordim(ims8) --> 0 "test nGifan"
        @fact colorspace(ims8) --> "Gray" "test R0VFeL"
        @fact convert(Image, ims8) --> exactly(ims8) "test EGoCYN"
        @fact convert(Image{UFixed8}, ims8) --> exactly(ims8) "test Qly190"
        @fact separate(ims8) --> exactly(ims8) "test hAxqus"
        A = reinterpret(UFixed8, UInt8[1 2; 3 4])
        imgray = convert(Image{Gray{UFixed8}}, A)
        @fact spatialorder(imgray) --> (Symbol.(Images.yx)...,) "test gLMUOh"
        @fact data(imgray) --> reinterpret(Gray{UFixed8}, [0x01 0x02; 0x03 0x04]) "test UC9NtZ"
        @fact eltype(convert(Image{HSV{Float32}}, imrgb8)) --> HSV{Float32} "test cwCaVn"
        @fact eltype(convert(Image{HSV}, float32(imrgb8))) --> HSV{Float32} "test VB4EU3"

        @fact eltype(convert(Array{Gray}, imrgb8)) --> Gray{U8} "test 4UOxZh"
        @fact eltype(convert(Image{Gray}, imrgb8)) --> Gray{U8} "test 2hhcQd"
        @fact eltype(convert(Array{Gray}, data(imrgb8))) --> Gray{U8} "test SdEY95"
        @fact eltype(convert(Image{Gray}, data(imrgb8))) --> Gray{U8} "test lzipLL"
        # Issue 232
        let img = Image(reinterpret(Gray{UFixed16}, rand(UInt16, 5, 5)))
            imgs = subim(img, :, :)
            @fact isa(minfinite(imgs), Gray{UFixed16}) --> true "test PlxHep"
            # Raw
            imgdata = rand(UInt16, 5, 5)
            img = Image(reinterpret(Gray{UFixed16}, imgdata))
            @fact all(raw(img) .== imgdata) --> true "test EvOATF"
            @fact typeof(raw(img).data) --> Array{UInt16,2} "test YlySCh"
            @fact typeof(raw(Image(rawrgb8)).data) --> Array{UInt8,3}  "test uOxsmv" # check color images
            @fact size(raw(Image(rawrgb8))) --> (3,5,4) "test bM2K4C"
            @fact typeof(raw(imgdata)) --> typeof(imgdata)  "test 1Bf874" # check array fallback
            @fact all(raw(imgdata) .== imgdata) --> true "test LJLDPq"
        end
    end
end
