B = rand(1:20,3,5)
cmap = uint8(repmat(linspace(12,255,20),1,3))
img = Images.ImageCmap(copy(B),cmap,["colorspace" => "RGB", "pixelspacing" => [2.0, 3.0], "spatialorder" => Images.yx])
imgd = convert(Images.Image, img)

# basic information
@assert size(img) == (3,5)
@assert size(imgd) == (3,5,3)
@assert ndims(img) == 2
@assert ndims(imgd) == 3

# copy/similar
A = randn(3,5,3)
imgc = copy(img)
@assert imgc.data == img.data
imgc = copy(imgd, A)
@assert imgc.data == A
img2 = similar(img)
@assert isa(img2, Images.ImageCmap)
@assert img2.data != img.data
img2 = similar(imgd)
@assert isa(img2, Images.Image)

# ref/assign
prev = img[4]
@assert prev == B[4]
img[4] = prev+1
@assert img.data[4] == prev+1
@assert img[4] == prev+1

# properties
@assert Images.colorspace(img) == "RGB"
@assert Images.colordim(img) == 0
@assert Images.colordim(imgd) == 3
@assert Images.timedim(img) == 0
@assert Images.limits(img) == (12,255)
@assert Images.limits(imgd) == (0,255)
@assert Images.pixelspacing(img) == [2.0, 3.0]
@assert Images.pixelspacing(imgd) == [2.0, 3.0]

@assert Images.sdims(img) == Images.sdims(imgd)
@assert Images.coords_spatial(img) == Images.coords_spatial(imgd)
@assert Images.size_spatial(img) == Images.size_spatial(imgd)

@assert Images.storageorder(img) == Images.yx
@assert Images.storageorder(imgd) == [Images.yx, "color"]

# sub/slice
s = sub(img, 2, 1:4)
@assert ndims(s) == 2
@assert Images.sdims(s) == 2
@assert size(s) == (1,4)
s = Images.subim(img, 2, 1:4)
@assert ndims(s) == 2
@assert Images.sdims(s) == 2
@assert size(s) == (1,4)
s = Images.sliceim(img, 2, 1:4)
@assert ndims(s) == 1
@assert Images.sdims(s) == 1
@assert size(s) == (4,)
s = Images.sliceim(imgd, 2, 1:4, 1:3)
@assert ndims(s) == 2
@assert Images.sdims(s) == 1
@assert Images.colordim(s) == 2

# named indexing
@assert img["y", 2, "x", 4] == B[2,4]
@assert img["x", 4, "y", 2] == B[2,4]
chan = imgd["color", 2]
Blookup = reshape(cmap[B[:],1], size(B))
@assert chan == Blookup

# spatial order, width/height, and permutations
@assert Images.spatialpermutation(Images.yx, imgd) == [1,2]
@assert Images.widthheight(imgd) == (5,3)
C = convert(Array{Uint8,3}, imgd)
@assert C == imgd.data
imgd.properties["spatialorder"] = ["x", "y"]
@assert Images.spatialpermutation(Images.xy, imgd) == [1,2]
@assert Images.widthheight(imgd) == (3,5)
C = convert(Array{Uint8,3}, imgd)
@assert C == permutedims(imgd.data, [2,1,3])
imgd.properties["spatialorder"] = ["y", "x"]
@assert Images.spatialpermutation(Images.xy, imgd) == [2,1]
imgd.properties["spatialorder"] = ["x", "L"]
@assert Images.spatialpermutation(Images.xy, imgd) == [1,2]
imgd.properties["spatialorder"] = ["L", "x"]
@assert Images.spatialpermutation(Images.xy, imgd) == [2,1]
@assert Images.spatialpermutation(Images.xy, A) == [2,1]
@assert Images.spatialpermutation(Images.yx, A) == [1,2]
