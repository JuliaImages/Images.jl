using Images, FixedPointNumbers, Color, SIUnits.ShortUnits, Base.Test

## Constructors of Image types
B = rand(1:20,3,5)    # support integer-valued types, but these are NOT recommended (use Ufixed)
@test colorspace(B) == "Gray"
@test colordim(B) == 0
img = Image(B, colorspace="RGB", colordim=1)  # keyword constructor
@test colorspace(img) == "RGB"
@test colordim(img) == 1
img = grayim(B)
@test colorspace(img) == "Gray"
@test colordim(B) == 0
Bf = grayim(uint8(B))  # this is recommended for "integer-valued" images (or even better, directly as a Ufixed type)
@test eltype(Bf) == Ufixed8
@test colorspace(Bf) == "Gray"
@test colordim(Bf) == 0
Bf = grayim(uint16(B))
@test eltype(Bf) == Ufixed16
BfCV = reinterpret(Gray{Ufixed8}, uint8(B)) # colorspace encoded as a ColorValue (enables multiple dispatch)
@test colorspace(BfCV) == "Gray"
@test colordim(BfCV) == 0

# colorim
@test colordim(colorim(rand(Uint8, 3, 5, 5))) == 1
@test colordim(colorim(rand(Uint8, 5, 5, 3))) == 3
@test spatialorder(colorim(rand(Uint8, 3, 5, 5))) == ["x", "y"]
@test spatialorder(colorim(rand(Uint8, 5, 5, 3))) == ["y", "x"]
@test colordim(colorim(rand(Uint8, 4, 5, 5), "RGBA")) == 1
@test colordim(colorim(rand(Uint8, 4, 5, 5), "ARGB")) == 1
@test colordim(colorim(rand(Uint8, 5, 5, 4), "RGBA")) == 3
@test colordim(colorim(rand(Uint8, 5, 5, 4), "ARGB")) == 3
@test spatialorder(colorim(rand(Uint8, 4, 5, 5), "ARGB")) == ["x", "y"]
@test spatialorder(colorim(rand(Uint8, 5, 5, 4), "ARGB")) == ["y", "x"]
@test_throws ErrorException colorim(rand(Uint8, 3, 5, 3))
@test_throws ErrorException colorim(rand(Uint8, 4, 5, 5))
@test_throws ErrorException colorim(rand(Uint8, 5, 5, 4))
@test_throws ErrorException colorim(rand(Uint8, 4, 5, 4), "ARGB")
@test_throws ErrorException colorim(rand(Uint8, 5, 5, 5), "foo")
@test_throws ErrorException colorim(rand(Uint8, 2, 2, 2), "bar")

# indexed color
cmap = linspace(RGB(0x0cuf8,0x00uf8,0x00uf8), RGB(0xffuf8,0x00uf8,0x00uf8),20)
img = ImageCmap(copy(B), cmap, (Any=>Any)["spatialorder" => Images.yx])
@test colorspace(img) == "RGB"
img = ImageCmap(copy(B), cmap, spatialorder=Images.yx)
@test colorspace(img) == "RGB"
cmap = reinterpret(RGB, repmat(reinterpret(Ufixed8, uint8(linspace(12,255,20)))',3,1))
img = ImageCmap(copy(B), cmap, ["pixelspacing" => [2.0, 3.0], "spatialorder" => Images.yx])
imgd = convert(Image, img)
@test eltype(img) == RGB{Ufixed8}
@test eltype(imgd) == RGB{Ufixed8}

imgd["pixelspacing"] = [2.0mm, 3.0mm]
imgds = separate(imgd)

# img, imgd, and imgds will be used in many more tests

# basic information
@test size(img) == (3,5)
@test size(imgd) == (3,5)
@test ndims(img) == 2
@test ndims(imgd) == 2
@test size(img,"y") == 3
@test size(img,"x") == 5

# printing
iob = IOBuffer()
show(iob, img)
show(iob, imgd)

# copy/similar
A = randn(3,5,3)
imgc = copy(img)
@test imgc.data == img.data
imgc = copy(imgd, A)
@test imgc.data == A
img2 = similar(img)
@test isa(img2, ImageCmap)
@test img2.data != img.data
img2 = similar(imgd)
@test isa(img2, Image)
img2 = similar(img, (4,4))
@test isa(img2, ImageCmap)
@test size(img2) == (4,4)
img2 = similar(imgd, (3,4,4))
@test isa(img2, Image)
@test size(img2) == (3,4,4)

# getindex/setindex!
prev = img[4]
@test prev == B[4]
img[4] = prev+1
@test img.data[4] == prev+1
@test img[4] == prev+1
@test img[1,2] == prev+1
img[1,2] = prev
@test img[4] == prev

# properties
@test colorspace(img) == "RGB"
@test colordim(img) == 0
@test colordim(imgds) == 3
@test timedim(img) == 0
@test pixelspacing(img) == [2.0, 3.0]
@test pixelspacing(imgd) == [2.0mm, 3.0mm]
@test spacedirections(img) == Vector{Float64}[[2.0, 0], [0, 3.0]]
@test spacedirections(imgd) == Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[2.0mm, 0.0mm], [0.0mm, 3.0mm]]

@test sdims(img) == sdims(imgd)
@test coords_spatial(img) == coords_spatial(imgd)
@test size_spatial(img) == size_spatial(imgd)

tmp = Image(A, (Any=>Any)[])
copy!(tmp, imgd, "spatialorder")
@test properties(tmp) == (Any=>Any)["spatialorder" => Images.yx]
copy!(tmp, imgd, "spatialorder", "pixelspacing")
@test tmp["pixelspacing"] == [2.0mm, 3.0mm]

@test storageorder(img) == Images.yx
@test storageorder(imgds) == [Images.yx, "color"]

A = rand(4,4,3)
@test colordim(A) == 3
Aimg = permutedims(convert(Image, A), [3,1,2])
@test colordim(Aimg) == 1

# sub/slice
s = sub(img, 2, 1:4)
@test ndims(s) == 2
@test sdims(s) == 2
@test size(s) == (1,4)
s = subim(img, 2, 1:4)
@test ndims(s) == 2
@test sdims(s) == 2
@test size(s) == (1,4)
s = sliceim(img, 2, 1:4)
@test ndims(s) == 1
@test sdims(s) == 1
@test size(s) == (4,)
s = sliceim(imgds, 2, 1:4, 1:3)
@test ndims(s) == 2
@test sdims(s) == 1
@test colordim(s) == 2
@test spatialorder(s) == ["x"]
s = sliceim(imgds, 2:2, 1:4, 1:3)
@test ndims(s) == 3
@test sdims(s) == 2
@test colordim(s) == 3
@test spatialorder(s) == ["y","x"]

# reslicing
D = randn(3,5,4)
sd = SliceData(D, 2)
C = slice(D, sd, 2)
@test C == reshape(D[1:end, 2, 1:end], size(C))
reslice!(C, sd, 3)
@test C == reshape(D[1:end, 3, 1:end], size(C))
sd = SliceData(D, 3)
C = slice(D, sd, 2)
@test C == reshape(D[1:end, 1:end, 2], size(C))

sd = SliceData(imgds, 2)
s = sliceim(imgds, sd, 2)
@test colordim(s) == 2
@test colorspace(s) == "RGB"
@test spatialorder(s) == ["y"]
@test s.data == reshape(imgds[:,2,:], size(s))
sd = SliceData(imgds, 3)
s = sliceim(imgds, sd, 2)
@test colordim(s) == 0
@test colorspace(s) == "Unknown"
@test spatialorder(s) == Images.yx
@test s.data == imgds[:,:,2]
reslice!(s, sd, 3)
@test s.data == imgds[:,:,3]

# named indexing
@test dimindex(imgds, "color") == 3
@test dimindex(imgds, "y") == 1
@test dimindex(imgds, "z") == 0
imgdp = permutedims(imgds, [3,1,2])
@test dimindex(imgdp, "y") == 2
@test coords(imgds, "x", 2:4) == (1:3, 2:4, 1:3)
@test coords(imgds, x=2:4, y=2:3) == (2:3, 2:4, 1:3)
@test img["y", 2, "x", 4] == B[2,4]
@test img["x", 4, "y", 2] == B[2,4]
chan = imgds["color", 2]
Blookup = reshape(green(cmap[B[:]]), size(B))
@test chan == Blookup

sd = SliceData(imgds, "x")
s = sliceim(imgds, sd, 2)
@test spatialorder(s) == ["y"]
@test s.data == reshape(imgds[:,2,:], size(s))
sd = SliceData(imgds, "y")
s = sliceim(imgds, sd, 2)
@test spatialorder(s) == ["x"]
@test s.data == reshape(imgds[2,:,:], size(s))
sd = SliceData(imgds, "x", "y")
s = sliceim(imgds, sd, 2, 1)
@test s.data == reshape(imgds[1,2,:], 3)

# spatial order, width/height, and permutations
@test spatialpermutation(Images.yx, imgds) == [1,2]
@test widthheight(imgds) == (5,3)
C = convert(Array, imgds)
@test C == imgds.data
imgds["spatialorder"] = ["x", "y"]
@test spatialpermutation(Images.xy, imgds) == [1,2]
@test widthheight(imgds) == (3,5)
C = convert(Array, imgds)
@test C == permutedims(imgds.data, [2,1,3])
imgds.properties["spatialorder"] = ["y", "x"]
@test spatialpermutation(Images.xy, imgds) == [2,1]
imgds.properties["spatialorder"] = ["x", "L"]
@test spatialpermutation(Images.xy, imgds) == [1,2]
imgds.properties["spatialorder"] = ["L", "x"]
@test spatialpermutation(Images.xy, imgds) == [2,1]
@test spatialpermutation(Images.xy, A) == [2,1]
@test spatialpermutation(Images.yx, A) == [1,2]

imgds.properties["spatialorder"] = Images.yx
imgp = permutedims(imgds, ["x", "y", "color"])
@test imgp.data == permutedims(imgds.data, [2,1,3])
imgp = permutedims(imgds, ("color", "x", "y"))
@test imgp.data == permutedims(imgds.data, [3,2,1])
@test pixelspacing(imgp) == [3.0mm, 2.0mm]
imgc = copy(imgds)
imgc["spacedirections"] = spacedirections(imgc)
delete!(imgc, "pixelspacing")
imgp = permutedims(imgc, ["x", "y", "color"])
@test spacedirections(imgp) == Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[0.0mm, 3.0mm],[2.0mm, 0.0mm]]
@test pixelspacing(imgp) == [3.0mm, 2.0mm]

# reinterpret, separate, more convert
A8 = ufixed8(rand(0x00:0xff, 3, 5, 4))
rawrgb8 = reinterpret(RGB, A8)
@test eltype(rawrgb8) == RGB{Ufixed8}
rawrgb32 = float32(rawrgb8)
@test eltype(rawrgb32) == RGB{Float32}
@test ufixed8(rawrgb32) == rawrgb8
@test reinterpret(Ufixed8, rawrgb8) == A8
imrgb8 = convert(Image, rawrgb8)
@test spatialorder(imrgb8) == Images.yx
im8 = reinterpret(Ufixed8, imrgb8)
@test data(im8) == A8
@test reinterpret(RGB, im8) == imrgb8
ims8 = separate(imrgb8)
@test colordim(ims8) == 3
@test colorspace(ims8) == "RGB"
imrgb8_2 = convert(Image{RGB}, ims8)
@test isa(imrgb8_2, Image{RGB{Ufixed8}})
@test imrgb8_2 == imrgb8

@test eltype(convert(Image{HSV{Float32}}, imrgb8)) == HSV{Float32}
@test eltype(convert(Image{HSV}, float32(imrgb8))) == HSV{Float32}
