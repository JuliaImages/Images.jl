using FactCheck, Images, Colors, FixedPointNumbers

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end

facts("Core") do
    # support integer-valued types, but these are NOT recommended (use Ufixed)
    B = rand(convert(UInt16, 1):convert(UInt16, 20), 3, 5)
    # img, imgd, and imgds will be used in many more tests
    # Thus, these must be defined as local if reassigned in any context() blocks
    cmap = reinterpret(RGB, repmat(reinterpret(Ufixed8, round(UInt8, linspace(12, 255, 20)))', 3, 1))
    img = ImageCmap(copy(B), cmap, Dict{ASCIIString, Any}([("pixelspacing", [2.0, 3.0]), ("spatialorder", Images.yx)]))
    imgd = convert(Image, img)
    if testing_units
        imgd["pixelspacing"] = [2.0mm, 3.0mm]
    end
    imgds = separate(imgd)

    context("Constructors of Image types") do
        @fact colorspace(B) --> "Gray"
        @fact colordim(B) --> 0
        local img = Image(B, colorspace="RGB", colordim=1)  # keyword constructor
        @fact colorspace(img) --> "RGB"
        @fact colordim(img) --> 1
        img = grayim(B)
        @fact colorspace(img) --> "Gray"
        @fact colordim(B) --> 0
        @fact grayim(img) --> img
        # this is recommended for "integer-valued" images (or even better, directly as a Ufixed type)
        Bf = grayim(round(UInt8, B))
        @fact eltype(Bf) --> Ufixed8
        @fact colorspace(Bf) --> "Gray"
        @fact colordim(Bf) --> 0
        Bf = grayim(B)
        @fact eltype(Bf) --> Ufixed16
        # colorspace encoded as a ColorValue (enables multiple dispatch)
        BfCV = reinterpret(Gray{Ufixed8}, round(UInt8, B))
        @fact colorspace(BfCV) --> "Gray"
        @fact colordim(BfCV) --> 0
        Bf3 = grayim(reshape(convert(UInt8,1):convert(UInt8,36), 3,4,3))
        @fact eltype(Bf3) --> Ufixed8
        Bf3 = grayim(reshape(convert(UInt16,1):convert(UInt16,36), 3,4,3))
        @fact eltype(Bf3) --> Ufixed16
        Bf3 = grayim(reshape(1.0f0:36.0f0, 3,4,3))
        @fact eltype(Bf3) --> Float32
    end

    context("Colorim") do
        C = colorim(rand(Uint8, 3, 5, 5))
        @fact eltype(C) --> RGB{Ufixed8}
        @fact colordim(C) --> 0
        @fact colorim(C) --> C
        C = colorim(rand(Uint16, 4, 5, 5), "ARGB")
        @fact eltype(C) --> ARGB{Ufixed16}
        C = colorim(rand(1:20, 3, 5, 5))
        @fact eltype(C) --> Int
        @fact colordim(C) --> 1
        @fact colorspace(C) --> "RGB"
        @fact eltype(colorim(rand(Uint16, 3, 5, 5))) --> RGB{Ufixed16}
        @fact eltype(colorim(rand(3, 5, 5))) --> RGB{Float64}
        @fact colordim(colorim(rand(Uint8, 5, 5, 3))) --> 3
        @fact spatialorder(colorim(rand(Uint8, 3, 5, 5))) --> ["x", "y"]
        @fact spatialorder(colorim(rand(Uint8, 5, 5, 3))) --> ["y", "x"]
        @fact eltype(colorim(rand(Uint8, 4, 5, 5), "RGBA")) --> RGBA{Ufixed8}
        @fact eltype(colorim(rand(Uint8, 4, 5, 5), "ARGB")) --> ARGB{Ufixed8}
        @fact colordim(colorim(rand(Uint8, 5, 5, 4), "RGBA")) --> 3
        @fact colordim(colorim(rand(Uint8, 5, 5, 4), "ARGB")) --> 3
        @fact spatialorder(colorim(rand(Uint8, 4, 5, 5), "ARGB")) --> ["x", "y"]
        @fact spatialorder(colorim(rand(Uint8, 5, 5, 4), "ARGB")) --> ["y", "x"]
        @fact_throws ErrorException colorim(rand(Uint8, 3, 5, 3))
        @fact_throws ErrorException colorim(rand(Uint8, 4, 5, 5))
        @fact_throws ErrorException colorim(rand(Uint8, 5, 5, 4))
        @fact_throws ErrorException colorim(rand(Uint8, 4, 5, 4), "ARGB")
        @fact_throws ErrorException colorim(rand(Uint8, 5, 5, 5), "foo")
        @fact_throws ErrorException colorim(rand(Uint8, 2, 2, 2), "bar")
    end

    context("Indexed color") do
        local cmap = linspace(RGB(0x0cuf8, 0x00uf8, 0x00uf8), RGB(0xffuf8, 0x00uf8, 0x00uf8), 20)
        local img_ = ImageCmap(copy(B), cmap, Dict{ASCIIString, Any}([("spatialorder", Images.yx)]))
        @fact colorspace(img_) --> "RGB"
        img_ = ImageCmap(copy(B), cmap, spatialorder=Images.yx)
        @fact colorspace(img_) --> "RGB"
        # Note: img from opening of facts() block
        # TODO: refactor whole block
        @fact eltype(img) --> RGB{Ufixed8}
        @fact eltype(imgd) --> RGB{Ufixed8}
    end


    context("Basic information") do
        @fact size(img) --> (3,5)
        @fact size(imgd) --> (3,5)
        @fact ndims(img) --> 2
        @fact ndims(imgd) --> 2
        @fact size(img,"y") --> 3
        @fact size(img,"x") --> 5
        @fact strides(img) --> (1,3)
        @fact strides(imgd) --> (1,3)
        @fact strides(imgds) --> (1,3,15)
        @fact nimages(img) --> 1
        @fact ncolorelem(img) --> 1
        @fact ncolorelem(imgd) --> 1
        @fact ncolorelem(imgds) --> 3
    end

    context("1-dimensional images") do
        @fact colordim(1:5) --> 0
        @fact nimages(1:5) --> 1
        @fact colorspace(1:5) --> "Gray"
        img1 = Image(map(Gray, 0.1:0.1:0.5), spatialorder=["z"])
        @fact colordim(img1) --> 0
        @fact colorspace(img1) --> "Gray"
        @fact sdims(img1) --> 1
        @fact convert(Vector{Gray{Float64}}, img1) --> map(Gray, 0.1:0.1:0.5)
        @fact size(img1, "z") --> 5
        @fact_throws ErrorException size(img1, "x")
    end

    context("Printing") do
        iob = IOBuffer()
        show(iob, img)
        show(iob, imgd)
    end

    context("Copy / similar") do
        A = randn(3,5,3)
        imgc = copy(img)
        @fact imgc.data --> img.data
        imgc = copyproperties(imgd, A)
        @fact imgc.data --> A
        img2 = similar(img)
        @fact isa(img2, ImageCmap) --> true
        @fact (img2.data == img.data) --> false
        img2 = similar(imgd)
        @fact isa(img2, Image) --> true
        img2 = similar(img, (4,4))
        @fact isa(img2, ImageCmap) --> true
        @fact size(img2) --> (4,4)
        img2 = similar(imgd, (3,4,4))
        @fact isa(img2, Image) --> true
        @fact size(img2) --> (3,4,4)
        @fact copyproperties(B, A) --> A
        @fact shareproperties(A, B) --> B
        @fact shareproperties(img, B) --> B
    end

    context("Getindex / setindex!") do
        prev = img[4]
        @fact prev --> B[4]
        img[4] = prev+1
        @fact img.data[4] --> prev+1
        @fact img[4] --> prev+1
        @fact img[1,2] --> prev+1
        img[1,2] = prev
        @fact img[4] --> prev
    end

    context("Properties") do
        @fact colorspace(img) --> "RGB"
        @fact colordim(img) --> 0
        @fact colordim(imgds) --> 3
        @fact timedim(img) --> 0
        @fact pixelspacing(img) --> [2.0, 3.0]
        @fact spacedirections(img) --> Vector{Float64}[[2.0, 0], [0, 3.0]]
        if testing_units
            @fact pixelspacing(imgd) --> [2.0mm, 3.0mm]
            @fact spacedirections(imgd) -->
                Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[2.0mm, 0.0mm], [0.0mm, 3.0mm]]
        end
    end

    context("Dims and ordering") do
        @fact sdims(img) --> sdims(imgd)
        @fact coords_spatial(img) --> coords_spatial(imgd)
        @fact size_spatial(img) --> size_spatial(imgd)
        A = randn(3,5,3)
        tmp = Image(A, Dict{ASCIIString,Any}())
        copy!(tmp, imgd, "spatialorder")
        @fact properties(tmp) --> Dict{ASCIIString,Any}([("spatialorder",Images.yx)])
        copy!(tmp, imgd, "spatialorder", "pixelspacing")
        if testing_units
            @fact tmp["pixelspacing"] --> [2.0mm, 3.0mm]
        end

        @fact storageorder(img) --> Images.yx
        @fact storageorder(imgds) --> [Images.yx; "color"]

        A = rand(4,4,3)
        @fact colordim(A) --> 3
        Aimg = permutedims(convert(Image, A), [3,1,2])
        @fact colordim(Aimg) --> 1
    end

    context("Sub / slice") do
        s = sub(img, 2, 1:4)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact size(s) --> (1,4)
        @fact data(s) --> B[2, 1:4]
        s = getindexim(img, 2, 1:4)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact size(s) --> (1,4)
        @fact data(s) --> B[2, 1:4]
        s = subim(img, 2, 1:4)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact size(s) --> (1,4)
        s = sliceim(img, 2, 1:4)
        @fact ndims(s) --> 1
        @fact sdims(s) --> 1
        @fact size(s) --> (4,)
        s = sliceim(imgds, 2, 1:4, 1:3)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 1
        @fact colordim(s) --> 2
        @fact spatialorder(s) --> ["x"]
        s = sliceim(imgds, 2:2, 1:4, 1:3)
        @fact ndims(s) --> 3
        @fact sdims(s) --> 2
        @fact colordim(s) --> 3
        @fact colorspace(s) --> "RGB"
        s = getindexim(imgds, 2:2, 1:4, 2)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact colordim(s) --> 0
        @fact colorspace(s) --> "Unknown"
        s = sliceim(imgds, 2:2, 1:4, 2)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact colordim(s) --> 0
        @fact colorspace(s) --> "Unknown"
        s = sliceim(imgds, 2:2, 1:4, 1:2)
        @fact ndims(s) --> 3
        @fact sdims(s) --> 2
        @fact colordim(s) --> 3
        @fact colorspace(s) --> "Unknown"
        @fact spatialorder(s) --> ["y","x"]
        s = sub(img, "y", 2)
        @fact ndims(s) --> 2
        @fact sdims(s) --> 2
        @fact size(s) --> (1,5)
        s = slice(img, "y", 2)
        @fact ndims(s) --> 1
        @fact size(s) --> (5,)
        @fact_throws ErrorException subim(img, [1,3,2],1:4)
        @fact_throws ErrorException sliceim(img, [1,3,2],1:4)
        @fact size(getindexim(imgds, :, 1:2, :)) --> (size(imgds,1), 2, 3)

        s = permutedims(imgds, (3,1,2))
        @fact colordim(s) --> 1
        ss = getindexim(s, 2, :, :)
        @fact colorspace(ss) --> "Unknown"
        @fact colordim(ss) --> 1
        sss = squeeze(ss, 1)
        @fact colorspace(ss) --> "Unknown"
        @fact colordim(sss) --> 0
        ss = getindexim(imgds, 2, :, :)
        @fact colordim(ss) --> 3
        @fact spatialorder(ss) --> ["y", "x"]
        sss = squeeze(ss, 1)
        @fact colordim(sss) --> 2
        @fact spatialorder(sss) --> ["x"]
    end

# # reslicing
# D = randn(3,5,4)
# sd = SliceData(D, 2)
# C = slice(D, sd, 2)
# @fact C --> reshape(D[1:end, 2, 1:end], size(C))
# reslice!(C, sd, 3)
# @fact C --> reshape(D[1:end, 3, 1:end], size(C))
# sd = SliceData(D, 3)
# C = slice(D, sd, 2)
# @fact C --> reshape(D[1:end, 1:end, 2], size(C))
#
# sd = SliceData(imgds, 2)
# s = sliceim(imgds, sd, 2)
# @fact colordim(s) --> 2
# @fact colorspace(s) --> "RGB"
# @fact spatialorder(s) --> ["y"]
# @fact s.data --> reshape(imgds[:,2,:], size(s))
# sd = SliceData(imgds, 3)
# s = sliceim(imgds, sd, 2)
# @fact colordim(s) --> 0
# @fact colorspace(s) --> "Unknown"
# @fact spatialorder(s) --> Images.yx
# @fact s.data --> imgds[:,:,2]
# reslice!(s, sd, 3)
# @fact s.data --> imgds[:,:,3]

    context("Named indexing") do
        @fact dimindex(imgds, "color") --> 3
        @fact dimindex(imgds, "y") --> 1
        @fact dimindex(imgds, "z") --> 0
        imgdp = permutedims(imgds, [3,1,2])
        @fact dimindex(imgdp, "y") --> 2
        @fact coords(imgds, "x", 2:4) --> (1:3, 2:4, 1:3)
        @fact coords(imgds, x=2:4, y=2:3) --> (2:3, 2:4, 1:3)
        @fact img["y", 2, "x", 4] --> B[2,4]
        @fact img["x", 4, "y", 2] --> B[2,4]
        chan = imgds["color", 2]
        Blookup = reshape(green(cmap[B[:]]), size(B))
        @fact chan --> Blookup

        sd = SliceData(imgds, "x")
        s = sliceim(imgds, sd, 2)
        @fact spatialorder(s) --> ["y"]
        @fact s.data --> reshape(imgds[:,2,:], size(s))
        sd = SliceData(imgds, "y")
        s = sliceim(imgds, sd, 2)
        @fact spatialorder(s) --> ["x"]
        @fact s.data --> reshape(imgds[2,:,:], size(s))
        sd = SliceData(imgds, "x", "y")
        s = sliceim(imgds, sd, 2, 1)
        @fact s.data --> reshape(imgds[1,2,:], 3)
    end

    context("Spatial order, width/ height, and permutations") do
        @fact spatialpermutation(Images.yx, imgds) --> [1,2]
        @fact widthheight(imgds) --> (5,3)
        C = convert(Array, imgds)
        @fact C --> imgds.data
        imgds["spatialorder"] = ["x", "y"]
        @fact spatialpermutation(Images.xy, imgds) --> [1,2]
        @fact widthheight(imgds) --> (3,5)
        C = convert(Array, imgds)
        @fact C --> permutedims(imgds.data, [2,1,3])
        imgds.properties["spatialorder"] = ["y", "x"]
        @fact spatialpermutation(Images.xy, imgds) --> [2,1]
        imgds.properties["spatialorder"] = ["x", "L"]
        @fact spatialpermutation(Images.xy, imgds) --> [1,2]
        imgds.properties["spatialorder"] = ["L", "x"]
        @fact spatialpermutation(Images.xy, imgds) --> [2,1]
        A = randn(3,5,3)
        @fact spatialpermutation(Images.xy, A) --> [2,1]
        @fact spatialpermutation(Images.yx, A) --> [1,2]

        imgds.properties["spatialorder"] = Images.yx
        imgp = permutedims(imgds, ["x", "y", "color"])
        @fact imgp.data --> permutedims(imgds.data, [2,1,3])
        imgp = permutedims(imgds, ("color", "x", "y"))
        @fact imgp.data --> permutedims(imgds.data, [3,2,1])
        if testing_units
            @fact pixelspacing(imgp) --> [3.0mm, 2.0mm]
        end
        imgc = copy(imgds)
        imgc["spacedirections"] = spacedirections(imgc)
        delete!(imgc, "pixelspacing")
        imgp = permutedims(imgc, ["x", "y", "color"])
        if testing_units
            @fact spacedirections(imgp) --> Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[0.0mm, 3.0mm],[2.0mm, 0.0mm]]
            @fact pixelspacing(imgp) --> [3.0mm, 2.0mm]
        end
    end

    context("Reinterpret, separate, more convert") do
        a = RGB{Float64}[RGB(1,1,0)]
        af = reinterpret(Float64, a)
        @fact vec(af) --> [1.0,1.0,0.0]
        @fact size(af) --> (3,1)
        @fact_throws ErrorException reinterpret(Float32, a)
        anew = reinterpret(RGB, af)
        @fact anew --> a
        anew = reinterpret(RGB, vec(af))
        @fact anew[1] --> a[1]
        @fact ndims(anew) --> 0
        anew = reinterpret(RGB{Float64}, af)
        @fact anew --> a
        @fact_throws ErrorException reinterpret(RGB{Float32}, af)
        Au8 = rand(0x00:0xff, 3, 5, 4)
        A8 = reinterpret(Ufixed8, Au8)
        rawrgb8 = reinterpret(RGB, A8)
        @fact eltype(rawrgb8) --> RGB{Ufixed8}
        @fact reinterpret(Ufixed8, rawrgb8) --> A8
        @fact reinterpret(Uint8, rawrgb8) --> Au8
        rawrgb32 = convert(Array{RGB{Float32}}, rawrgb8)
        @fact eltype(rawrgb32) --> RGB{Float32}
        @fact ufixed8(rawrgb32) --> rawrgb8
        @fact reinterpret(Ufixed8, rawrgb8) --> A8
        imrgb8 = convert(Image, rawrgb8)
        @fact spatialorder(imrgb8) --> Images.yx
        @fact convert(Image, imrgb8) --> exactly(imrgb8)
        @fact convert(Image{RGB{Ufixed8}}, imrgb8) --> exactly(imrgb8)
        im8 = reinterpret(Ufixed8, imrgb8)
        @fact data(im8) --> A8
        @fact permutedims(reinterpret(Ufixed8, separate(imrgb8)), (3, 1, 2)) --> im8
        @fact reinterpret(Uint8, imrgb8) --> Au8
        @fact reinterpret(RGB, im8) --> imrgb8
        ims8 = separate(imrgb8)
        @fact colordim(ims8) --> 3
        @fact colorspace(ims8) --> "RGB"
        @fact convert(Image, ims8) --> exactly(ims8)
        @fact convert(Image{Ufixed8}, ims8) --> exactly(ims8)
        @fact separate(ims8) --> exactly(ims8)
        imrgb8_2 = convert(Image{RGB}, ims8)
        @fact isa(imrgb8_2, Image{RGB{Ufixed8}}) --> true
        @fact imrgb8_2 --> imrgb8
        A = reinterpret(Ufixed8, Uint8[1 2; 3 4])
        imgray = convert(Image{Gray{Ufixed8}}, A)
        @fact spatialorder(imgray) --> Images.yx
        @fact data(imgray) --> reinterpret(Gray{Ufixed8}, [0x01 0x02; 0x03 0x04])
        @fact eltype(convert(Image{HSV{Float32}}, imrgb8)) --> HSV{Float32}
        @fact eltype(convert(Image{HSV}, float32(imrgb8))) --> HSV{Float32}
        # Issue 232
        local img = Image(reinterpret(Gray{Ufixed16}, rand(Uint16, 5, 5)))
        imgs = subim(img, :, :)
        @fact isa(minfinite(imgs), Ufixed16) --> true
        # Raw
        imgdata = rand(Uint16, 5, 5)
        img = Image(reinterpret(Gray{Ufixed16}, imgdata))
        @fact all(raw(img) .== imgdata) --> true
        @fact typeof(raw(img)) --> Array{Uint16,2}
        @fact typeof(raw(Image(rawrgb8))) --> Array{Uint8,3}  # check color images
        @fact size(raw(Image(rawrgb8))) --> (3,5,4)
        @fact typeof(raw(imgdata)) --> typeof(imgdata)  # check array fallback
        @fact all(raw(imgdata) .== imgdata) --> true
    end
end
