using Images, SIUnits.ShortUnits
using Base.Test

B = rand(1:20,3,5)
cmap = uint8(repmat(linspace(12,255,20),1,3))
img = ImageCmap(copy(B),cmap,["colorspace" => "RGB", "pixelspacing" => [2.0, 3.0], "spatialorder" => Images.yx])
imgd = convert(Image, img)
imgd["pixelspacing"] = [2.0mm, 3.0mm]
@test eltype(img) == Uint8
@test eltype(imgd) == Uint8

# basic information
@test size(img) == (3,5)
@test size(imgd) == (3,5,3)
@test ndims(img) == 2
@test ndims(imgd) == 3

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

# ref/assign
prev = img[4]
@test prev == B[4]
img[4] = prev+1
@test img.data[4] == prev+1
@test img[4] == prev+1

# properties
@test colorspace(img) == "RGB"
@test colordim(img) == 0
@test colordim(imgd) == 3
@test timedim(img) == 0
@test limits(img) == (12,255)
@test limits(imgd) == (0,255)
@test pixelspacing(img) == [2.0, 3.0]
@test pixelspacing(imgd) == [2.0mm, 3.0mm]
@test spacedirections(img) == Vector{Float64}[[2.0, 0], [0, 3.0]]
@test spacedirections(imgd) == Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[2.0mm, 0.0mm], [0.0mm, 3.0mm]]

@test sdims(img) == sdims(imgd)
@test coords_spatial(img) == coords_spatial(imgd)
@test size_spatial(img) == size_spatial(imgd)

@test storageorder(img) == Images.yx
@test storageorder(imgd) == [Images.yx, "color"]

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
s = sliceim(imgd, 2, 1:4, 1:3)
@test ndims(s) == 2
@test sdims(s) == 1
@test colordim(s) == 2
@test spatialorder(s) == ["x"]
s = sliceim(imgd, 2:2, 1:4, 1:3)
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

sd = SliceData(imgd, 2)
s = sliceim(imgd, sd, 2)
@test colordim(s) == 2
@test colorspace(s) == "RGB"
@test spatialorder(s) == ["y"]
@test s.data == reshape(imgd[:,2,:], size(s))
sd = SliceData(imgd, 3)
s = sliceim(imgd, sd, 2)
@test colordim(s) == 0
@test colorspace(s) == "Gray"
@test spatialorder(s) == Images.yx
@test s.data == imgd[:,:,2]
reslice!(s, sd, 3)
@test s.data == imgd[:,:,3]

# named indexing
@test dimindex(imgd, "color") == 3
@test dimindex(imgd, "y") == 1
@test dimindex(imgd, "z") == 0
imgdp = permutedims(imgd, [3,1,2])
@test dimindex(imgdp, "y") == 2
@test img["y", 2, "x", 4] == B[2,4]
@test img["x", 4, "y", 2] == B[2,4]
chan = imgd["color", 2]
Blookup = reshape(cmap[B[:],1], size(B))
@test chan == Blookup

sd = SliceData(imgd, "x")
s = sliceim(imgd, sd, 2)
@test spatialorder(s) == ["y"]
@test s.data == reshape(imgd[:,2,:], size(s))
sd = SliceData(imgd, "y")
s = sliceim(imgd, sd, 2)
@test spatialorder(s) == ["x"]
@test s.data == reshape(imgd[2,:,:], size(s))
sd = SliceData(imgd, "x", "y")
s = sliceim(imgd, sd, 2, 1)
@test s.data == reshape(imgd[1,2,:], 3)

# spatial order, width/height, and permutations
@test spatialpermutation(Images.yx, imgd) == [1,2]
@test widthheight(imgd) == (5,3)
C = convert(Array, imgd)
@test C == imgd.data
imgd["spatialorder"] = ["x", "y"]
@test spatialpermutation(Images.xy, imgd) == [1,2]
@test widthheight(imgd) == (3,5)
C = convert(Array, imgd)
@test C == permutedims(imgd.data, [2,1,3])
imgd.properties["spatialorder"] = ["y", "x"]
@test spatialpermutation(Images.xy, imgd) == [2,1]
imgd.properties["spatialorder"] = ["x", "L"]
@test spatialpermutation(Images.xy, imgd) == [1,2]
imgd.properties["spatialorder"] = ["L", "x"]
@test spatialpermutation(Images.xy, imgd) == [2,1]
@test spatialpermutation(Images.xy, A) == [2,1]
@test spatialpermutation(Images.yx, A) == [1,2]

imgd.properties["spatialorder"] = Images.yx
imgp = permutedims(imgd, ["x", "y", "color"])
@test imgp.data == permutedims(imgd.data, [2,1,3])
imgp = permutedims(imgd, ("color", "x", "y"))
@test imgp.data == permutedims(imgd.data, [3,2,1])
@test pixelspacing(imgp) == [3.0mm, 2.0mm]
imgc = copy(imgd)
imgc["spacedirections"] = spacedirections(imgc)
delete!(imgc, "pixelspacing")
imgp = permutedims(imgc, ["x", "y", "color"])
@test spacedirections(imgp) == Vector{SIUnits.SIQuantity{Float64,1,0,0,0,0,0,0}}[[0.0mm, 3.0mm],[2.0mm, 0.0mm]]
@test pixelspacing(imgp) == [3.0mm, 2.0mm]

## Convenience constructors
# colorim
@test colordim(colorim(rand(Uint8, 3, 5, 5))) == 1
@test colordim(colorim(rand(Uint8, 5, 5, 3))) == 3
@test spatialorder(colorim(rand(Uint8, 3, 5, 5))) == ["x", "y"]
@test spatialorder(colorim(rand(Uint8, 5, 5, 3))) == ["y", "x"]
@test_throws ErrorException colorim(rand(Uint8, 3, 5, 3))

@test_throws ErrorException colorim(rand(Uint8, 4, 5, 5))
@test_throws ErrorException colorim(rand(Uint8, 5, 5, 4))
@test_throws ErrorException colorim(rand(Uint8, 4, 5, 4), "ARGB")
@test colordim(colorim(rand(Uint8, 4, 5, 5), "RGBA")) == 1
@test colordim(colorim(rand(Uint8, 4, 5, 5), "ARGB")) == 1
@test colordim(colorim(rand(Uint8, 5, 5, 4), "RGBA")) == 3
@test colordim(colorim(rand(Uint8, 5, 5, 4), "ARGB")) == 3
@test spatialorder(colorim(rand(Uint8, 4, 5, 5), "ARGB")) == ["x", "y"]
@test spatialorder(colorim(rand(Uint8, 5, 5, 4), "ARGB")) == ["y", "x"]

@test_throws ErrorException colorim(rand(Uint8, 5, 5, 5), "foo")
@test_throws ErrorException colorim(rand(Uint8, 2, 2, 2), "bar")
