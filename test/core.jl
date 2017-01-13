using Images, Colors, FixedPointNumbers
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
using Compat
using Compat.view

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end

const scalar_getindex_new = VERSION >= v"0.5.0-dev+1195"

@testset "Core" begin
    a = rand(3,3)
    @inferred(Image(a))
    # support integer-valued types, but these are NOT recommended (use UFixed)
    B = rand(convert(UInt16, 1):convert(UInt16, 20), 3, 5)
    # img, imgd, and imgds will be used in many more tests
    # Thus, these must be defined as local if reassigned in any context() blocks
    cmap = reinterpret(RGB, repmat(reinterpret(UFixed8, round(UInt8, linspace(12, 255, 20)))', 3, 1))
    img = ImageCmap(copy(B), cmap, Dict{Compat.ASCIIString, Any}([("pixelspacing", [2.0, 3.0]), ("spatialorder", Images.yx)]))
    imgd = convert(Image, img)
    if testing_units
        imgd["pixelspacing"] = [2.0mm, 3.0mm]
    end
    imgds = separate(imgd)

    @testset "Constructors of Image types" begin
        @test colorspace(B) == "Gray"
        @test colordim(B) == 0
        let img = Image(B, colorspace="RGB", colordim=1)  # keyword constructor
            @test colorspace(img) == "RGB"
            @test colordim(img) == 1
            img = grayim(B)
            @test colorspace(img) == "Gray"
            @test colordim(B) == 0
            @test grayim(img) == img
            # this is recommended for "integer-valued" images (or even better, directly as a UFixed type)
            Bf = grayim(round(UInt8, B))
            @test eltype(Bf) == UFixed8
            @test colorspace(Bf) == "Gray"
            @test colordim(Bf) == 0
            Bf = grayim(B)
            @test eltype(Bf) == UFixed16
            # colorspace encoded as a Color (enables multiple dispatch)
            BfCV = reinterpret(Gray{UFixed8}, round(UInt8, B))
            @test colorspace(BfCV) == "Gray"
            @test colordim(BfCV) == 0
            Bf3 = grayim(reshape(collect(convert(UInt8,1):convert(UInt8,36)), 3,4,3))
            @test eltype(Bf3) == UFixed8
            Bf3 = grayim(reshape(collect(convert(UInt16,1):convert(UInt16,36)), 3,4,3))
            @test eltype(Bf3) == UFixed16
            Bf3 = grayim(reshape(collect(1.0f0:36.0f0), 3,4,3))
            @test eltype(Bf3) == Float32
        end
    end

    @testset "Colorim" begin
        C = colorim(rand(UInt8, 3, 5, 5))
        @test eltype(C) == RGB{UFixed8}
        @test colordim(C) == 0
        @test colorim(C) == C
        C = colorim(rand(UInt16, 4, 5, 5), "ARGB")
        @test eltype(C) == ARGB{UFixed16}
        C = colorim(rand(1:20, 3, 5, 5))
        @test eltype(C) == Int
        @test colordim(C) == 1
        @test colorspace(C) == "RGB"
        @test eltype(colorim(rand(UInt16,3,5,5))) == RGB{UFixed16}
        @test eltype(colorim(rand(3,5,5))) == RGB{Float64}
        @test colordim(colorim(rand(UInt8,5,5,3))) == 3
        @test spatialorder(colorim(rand(UInt8,3,5,5))) == ["x","y"]
        @test spatialorder(colorim(rand(UInt8,5,5,3))) == ["y","x"]
        @test eltype(colorim(rand(UInt8,4,5,5),"RGBA")) == RGBA{UFixed8}
        @test eltype(colorim(rand(UInt8,4,5,5),"ARGB")) == ARGB{UFixed8}
        @test colordim(colorim(rand(UInt8,5,5,4),"RGBA")) == 3
        @test colordim(colorim(rand(UInt8,5,5,4),"ARGB")) == 3
        @test spatialorder(colorim(rand(UInt8,4,5,5),"ARGB")) == ["x","y"]
        @test spatialorder(colorim(rand(UInt8,5,5,4),"ARGB")) == ["y","x"]
        @test_throws ErrorException colorim(rand(UInt8,3,5,3))
        @test_throws ErrorException colorim(rand(UInt8,4,5,5))
        @test_throws ErrorException colorim(rand(UInt8,5,5,4))
        @test_throws ErrorException colorim(rand(UInt8,4,5,4),"ARGB")
        @test_throws ErrorException colorim(rand(UInt8,5,5,5),"foo")
        @test_throws ErrorException colorim(rand(UInt8,2,2,2),"bar")
    end

    @testset "Indexed color" begin
        let cmap = linspace(RGB(0x0cuf8, 0x00uf8, 0x00uf8), RGB(0xffuf8, 0x00uf8, 0x00uf8), 20)
            img_ = ImageCmap(copy(B), cmap, Dict{Compat.ASCIIString, Any}([("spatialorder", Images.yx)]))
            @test colorspace(img_) == "RGB"
            img_ = ImageCmap(copy(B), cmap, spatialorder=Images.yx)
            @test colorspace(img_) == "RGB"
            # Note: img from opening of facts() block
            # TODO: refactor whole block
            @test eltype(img) == RGB{UFixed8}
            @test eltype(imgd) == RGB{UFixed8}
        end
    end


    @testset "Basic information" begin
        @test size(img) == (3,5)
        @test size(imgd) == (3,5)
        @test ndims(img) == 2
        @test ndims(imgd) == 2
        @test size(img,"y") == 3
        @test size(img,"x") == 5
        @test strides(img) == (1,3)
        @test strides(imgd) == (1,3)
        @test strides(imgds) == (1,3,15)
        @test nimages(img) == 1
        @test ncolorelem(img) == 1
        @test ncolorelem(imgd) == 1
        @test ncolorelem(imgds) == 3
        vimg = vec(img)
        @test isa(vimg,ImageCmap)
        @test !(haskey(vimg,"spatialorder"))
        @test !(haskey(vimg,"pixelspacing"))
        @test (sort(vimg))[1] == minimum(img)
        vimgds = vec(imgds)
        @test isa(vimgds,Image)
        @test !(haskey(vimg,"colordim"))
        @test !(haskey(vimg,"colorspace"))
        @test (sort(vimgds))[1] == minimum(imgds)
    end

    @testset "1-dimensional images" begin
        @test colordim(1:5) == 0
        @test nimages(1:5) == 1
        @test colorspace(1:5) == "Gray"
        img1 = Image(map(Gray, 0.1:0.1:0.5), spatialorder=["z"])
        @test colordim(img1) == 0
        @test colorspace(img1) == "Gray"
        @test sdims(img1) == 1
        @test convert(Vector{Gray{Float64}},img1) == map(Gray,0.1:0.1:0.5)
        @test size(img1,"z") == 5
        @test_throws ErrorException size(img1,"x")
    end

    @testset "Printing" begin
        iob = IOBuffer()
        show(iob, img)
        show(iob, imgd)
    end

    @testset "Copy / similar" begin
        A = randn(3,5,3)
        imgc = @inferred(copy(img))
        @test imgc.data == img.data
        imgc = copyproperties(imgd, A)
        @test imgc.data == A
        img2 = @inferred(similar(img))
        @test isa(img2,ImageCmap)
        @test !(img2.data == img.data)
        img2 = @inferred(similar(imgd))
        @test isa(img2,Image)
        img2 = similar(img, (4,4))
        @test isa(img2,ImageCmap)
        @test size(img2) == (4,4)
        img2 = similar(imgd, (3,4,4))
        @test isa(img2,Image)
        @test size(img2) == (3,4,4)
        @test copyproperties(B,A) == A
        @test shareproperties(A,B) == B
        @test shareproperties(img,B) == B
    end

    @testset "Getindex / setindex!" begin
        prev = img[4]
        @test prev == B[4]
        img[4] = prev+1
        @test img.data[4] == prev + 1
        @test img[4] == prev + 1
        @test img[1,2] == prev + 1
        img[1,2] = prev
        @test img[4] == prev
    end

    @testset "Properties" begin
        @test colorspace(img) == "RGB"
        @test colordim(img) == 0
        @test colordim(imgds) == 3
        @test timedim(img) == 0
        @test pixelspacing(img) == [2.0,3.0]
        @test spacedirections(img) == Vector{Float64}[[2.0,0],[0,3.0]]
        if testing_units
            @test pixelspacing(imgd) == [2.0mm,3.0mm]
            @test spacedirections(imgd) ==
                Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[2.0mm, 0.0mm], [0.0mm, 3.0mm]]
        end
    end

    @testset "Dims and ordering" begin
        @test sdims(img) == sdims(imgd)
        @test coords_spatial(img) == coords_spatial(imgd)
        @test size_spatial(img) == size_spatial(imgd)
        A = randn(3,5,3)
        tmp = Image(A, Dict{Compat.ASCIIString,Any}())
        copy!(tmp, imgd, "spatialorder")
        @test properties(tmp) == Dict{Compat.ASCIIString,Any}([("spatialorder",Images.yx)])
        copy!(tmp, imgd, "spatialorder", "pixelspacing")
        if testing_units
            @test tmp["pixelspacing"] == [2.0mm,3.0mm]
        end

        @test storageorder(img) == Images.yx
        @test storageorder(imgds) == [Images.yx;"color"]

        A = rand(4,4,3)
        @test colordim(A) == 3
        Aimg = permutedims(convert(Image, A), [3,1,2])
        @test colordim(Aimg) == 1
    end

    @testset "Sub / slice" begin
        s = view(img, 2:2, 1:4)
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test size(s) == (1,4)
        @test data(s) == view(B,2:2,1:4)
        if scalar_getindex_new
            s = getindexim(img, 2, 1:4)
            @test ndims(s) == 1
            @test sdims(s) == 1
            @test size(s) == (4,)
            @test data(s) == B[2,1:4]
            s = getindexim(img, 2:2, 1:4)
            @test ndims(s) == 2
            @test sdims(s) == 2
            @test size(s) == (1,4)
            @test data(s) == B[2:2,1:4]
        else
            s = getindexim(img, 2, 1:4)
            @test ndims(s) == 2
            @test sdims(s) == 2
            @test size(s) == (1,4)
            @test data(s) == B[2,1:4]
        end
        s = subim(img, 2, 1:4)
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test size(s) == (1,4)
        s = subim(img, 2, [1,2,4])
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test size(s) == (1,3)
        s = sliceim(img, 2, 1:4)
        @test ndims(s) == 1
        @test sdims(s) == 1
        @test size(s) == (4,)
        s = sliceim(img, 2, [1,2,4])
        @test ndims(s) == 1
        @test sdims(s) == 1
        @test size(s) == (3,)
        s = sliceim(imgds, 2, 1:4, 1:3)
        @test ndims(s) == 2
        @test sdims(s) == 1
        @test colordim(s) == 2
        @test spatialorder(s) == ["x"]
        s = sliceim(imgds, 2, :, 1:3)
        @test ndims(s) == 2
        @test sdims(s) == 1
        @test colordim(s) == 2
        @test spatialorder(s) == ["x"]
        s = sliceim(imgds, 2:2, 1:4, 1:3)
        @test ndims(s) == 3
        @test sdims(s) == 2
        @test colordim(s) == 3
        @test colorspace(s) == "RGB"
        s = getindexim(imgds, 2:2, 1:4, 2)
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test colordim(s) == 0
        @test colorspace(s) == "Unknown"
        s = sliceim(imgds, 2:2, 1:4, 2)
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test colordim(s) == 0
        @test colorspace(s) == "Unknown"
        s = sliceim(imgds, 2:2, 1:4, 1:2)
        @test ndims(s) == 3
        @test sdims(s) == 2
        @test colordim(s) == 3
        @test colorspace(s) == "Unknown"
        @test spatialorder(s) == ["y","x"]
        s = view(img, "y", 2:2)
        @test ndims(s) == 2
        @test sdims(s) == 2
        @test size(s) == (1,5)
        s = view(img, "y", 2)
        @test ndims(s) == 1
        @test size(s) == (5,)
        @test size(getindexim(imgds,:,1:2,:)) == (size(imgds,1),2,3)

        s = permutedims(imgds, (3,1,2))
        @test colordim(s) == 1
        ss = getindexim(s, 2:2, :, :)
        @test colorspace(ss) == "Unknown"
        @test colordim(ss) == 1
        sss = squeeze(ss, 1)
        @test colorspace(ss) == "Unknown"
        @test colordim(sss) == 0
        ss = getindexim(imgds, 2:2, :, :)
        @test colordim(ss) == 3
        @test spatialorder(ss) == ["y","x"]
        sss = squeeze(ss, 1)
        @test colordim(sss) == 2
        @test spatialorder(sss) == ["x"]
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

    @testset "Named indexing" begin
        @test dimindex(imgds,"color") == 3
        @test dimindex(imgds,"y") == 1
        @test dimindex(imgds,"z") == 0
        imgdp = permutedims(imgds, [3,1,2])
        @test dimindex(imgdp,"y") == 2
        @test coords(imgds,"x",2:4) == (1:3,2:4,1:3)
        @test coords(imgds,x=2:4,y=2:3) == (2:3,2:4,1:3)
        @test img["y",2,"x",4] == B[2,4]
        @test img["x",4,"y",2] == B[2,4]
        chan = imgds["color", 2]
        Blookup = reshape(green(cmap[B[:]]), size(B))
        @test chan == Blookup

        sd = SliceData(imgds, "x")
        s = sliceim(imgds, sd, 2)
        @test spatialorder(s) == ["y"]
        @test s.data == reshape(imgds[:,2,:],size(s))
        sd = SliceData(imgds, "y")
        s = sliceim(imgds, sd, 2)
        @test spatialorder(s) == ["x"]
        @test s.data == reshape(imgds[2,:,:],size(s))
        sd = SliceData(imgds, "x", "y")
        s = sliceim(imgds, sd, 2, 1)
        @test s.data == reshape(imgds[1,2,:],3)
    end

    @testset "Spatial order, width/ height, and permutations" begin
        @test spatialpermutation(Images.yx,imgds) == [1,2]
        @test widthheight(imgds) == (5,3)
        C = convert(Array, imgds)
        @test C == imgds.data
        imgds["spatialorder"] = ["x", "y"]
        @test spatialpermutation(Images.xy,imgds) == [1,2]
        @test widthheight(imgds) == (3,5)
        C = convert(Array, imgds)
        @test C == permutedims(imgds.data,[2,1,3])
        imgds.properties["spatialorder"] = ["y", "x"]
        @test spatialpermutation(Images.xy,imgds) == [2,1]
        imgds.properties["spatialorder"] = ["x", "L"]
        @test spatialpermutation(Images.xy,imgds) == [1,2]
        imgds.properties["spatialorder"] = ["L", "x"]
        @test spatialpermutation(Images.xy,imgds) == [2,1]
        A = randn(3,5,3)
        @test spatialpermutation(Images.xy,A) == [2,1]
        @test spatialpermutation(Images.yx,A) == [1,2]

        imgds.properties["spatialorder"] = Images.yx
        imgp = permutedims(imgds, ["x", "y", "color"])
        @test imgp.data == permutedims(imgds.data,[2,1,3])
        imgp = permutedims(imgds, ("color", "x", "y"))
        @test imgp.data == permutedims(imgds.data,[3,2,1])
        if testing_units
            @test pixelspacing(imgp) == [3.0mm,2.0mm]
        end
        imgc = copy(imgds)
        imgc["spacedirections"] = spacedirections(imgc)
        delete!(imgc, "pixelspacing")
        imgp = permutedims(imgc, ["x", "y", "color"])
        if testing_units
            @test spacedirections(imgp) == Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[0.0mm,3.0mm],[2.0mm,0.0mm]]
            @test pixelspacing(imgp) == [3.0mm,2.0mm]
        end
    end

    @testset "Reinterpret, separate, more convert" begin
        a = RGB{Float64}[RGB(1,1,0)]
        af = reinterpret(Float64, a)
        @test vec(af) == [1.0,1.0,0.0]
        @test size(af) == (3,1)
        @test_throws ErrorException reinterpret(Float32,a)
        anew = reinterpret(RGB, af)
        @test anew == a
        anew = reinterpret(RGB, vec(af))
        @test anew[1] == a[1]
        @test ndims(anew) == 0
        anew = reinterpret(RGB{Float64}, af)
        @test anew == a
        @test_throws ErrorException reinterpret(RGB{Float32},af)
        Au8 = rand(0x00:0xff, 3, 5, 4)
        A8 = reinterpret(UFixed8, Au8)
        rawrgb8 = reinterpret(RGB, A8)
        @test eltype(rawrgb8) == RGB{UFixed8}
        @test reinterpret(UFixed8,rawrgb8) == A8
        @test reinterpret(UInt8,rawrgb8) == Au8
        rawrgb32 = convert(Array{RGB{Float32}}, rawrgb8)
        @test eltype(rawrgb32) == RGB{Float32}
        @test ufixed8(rawrgb32) == rawrgb8
        @test reinterpret(UFixed8,rawrgb8) == A8
        imrgb8 = convert(Image, rawrgb8)
        @test spatialorder(imrgb8) == Images.yx
        @test convert(Image,imrgb8) === imrgb8
        @test convert(Image{RGB{UFixed8}},imrgb8) === imrgb8
        im8 = reinterpret(UFixed8, imrgb8)
        @test data(im8) == A8
        @test permutedims(reinterpret(UFixed8,separate(imrgb8)),(3,1,2)) == im8
        @test reinterpret(UInt8,imrgb8) == Au8
        @test reinterpret(RGB,im8) == imrgb8
        ims8 = separate(imrgb8)
        @test colordim(ims8) == 3
        @test colorspace(ims8) == "RGB"
        @test convert(Image,ims8) === ims8
        @test convert(Image{UFixed8},ims8) === ims8
        @test separate(ims8) === ims8
        imrgb8_2 = convert(Image{RGB}, ims8)
        @test isa(imrgb8_2,Image{RGB{UFixed8}})
        @test imrgb8_2 == imrgb8
        A = reinterpret(UFixed8, UInt8[1 2; 3 4])
        imgray = convert(Image{Gray{UFixed8}}, A)
        @test spatialorder(imgray) == Images.yx
        @test data(imgray) == reinterpret(Gray{UFixed8},[0x01 0x02;0x03 0x04])
        @test eltype(convert(Image{HSV{Float32}},imrgb8)) == HSV{Float32}
        @test eltype(convert(Image{HSV},float32(imrgb8))) == HSV{Float32}

        @test eltype(convert(Array{Gray},imrgb8)) == Gray{U8}
        @test eltype(convert(Image{Gray},imrgb8)) == Gray{U8}
        @test eltype(convert(Array{Gray},data(imrgb8))) == Gray{U8}
        @test eltype(convert(Image{Gray},data(imrgb8))) == Gray{U8}
        # Issue 232
        let img = Image(reinterpret(Gray{UFixed16}, rand(UInt16, 5, 5)))
            imgs = subim(img, :, :)
            @test isa(minfinite(imgs),UFixed16)
            # Raw
            imgdata = rand(UInt16, 5, 5)
            img = Image(reinterpret(Gray{UFixed16}, imgdata))
            @test all(raw(img) .== imgdata)
            @test typeof(raw(img)) == Array{UInt16,2}
            @test typeof(raw(Image(rawrgb8))) == Array{UInt8,3} # check color images
            @test size(raw(Image(rawrgb8))) == (3,5,4)
            @test typeof(raw(imgdata)) == typeof(imgdata) # check array fallback
            @test all(raw(imgdata) .== imgdata)
        end
        # Issue #497
        let img = colorim(rand(3, 5, 5))
            img["colorspace"] = "sRGB"
            imgg = convert(Image{Gray}, img)
            @test !(haskey(imgg,"colorspace"))
        end
    end
end
