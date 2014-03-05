using Images

B = rand(1:20,3,5)
cmap = uint8(repmat(linspace(12,255,20),1,3))
img = ImageCmap(copy(B),cmap,["colorspace" => "RGB", "pixelspacing" => [2.0, 3.0], "spatialorder" => Images.yx])
imgd = convert(Image, img)

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
@assert isa(img2, ImageCmap)
@assert img2.data != img.data
img2 = similar(imgd)
@assert isa(img2, Image)

# ref/assign
prev = img[4]
@assert prev == B[4]
img[4] = prev+1
@assert img.data[4] == prev+1
@assert img[4] == prev+1

# properties
@assert colorspace(img) == "RGB"
@assert colordim(img) == 0
@assert colordim(imgd) == 3
@assert timedim(img) == 0
@assert limits(img) == (12,255)
@assert limits(imgd) == (0,255)
@assert pixelspacing(img) == [2.0, 3.0]
@assert pixelspacing(imgd) == [2.0, 3.0]

@assert sdims(img) == sdims(imgd)
@assert coords_spatial(img) == coords_spatial(imgd)
@assert size_spatial(img) == size_spatial(imgd)

@assert storageorder(img) == Images.yx
@assert storageorder(imgd) == [Images.yx, "color"]

A = rand(4,4,3)
@assert colordim(A) == 3
Aimg = permutedims(convert(Image, A), [3,1,2])
@assert colordim(Aimg) == 1

# sub/slice
s = sub(img, 2, 1:4)
@assert ndims(s) == 2
@assert sdims(s) == 2
@assert size(s) == (1,4)
s = subim(img, 2, 1:4)
@assert ndims(s) == 2
@assert sdims(s) == 2
@assert size(s) == (1,4)
s = sliceim(img, 2, 1:4)
@assert ndims(s) == 1
@assert sdims(s) == 1
@assert size(s) == (4,)
s = sliceim(imgd, 2, 1:4, 1:3)
@assert ndims(s) == 2
@assert sdims(s) == 1
@assert colordim(s) == 2
@assert spatialorder(s) == ["x"]
s = sliceim(imgd, 2:2, 1:4, 1:3)
@assert ndims(s) == 3
@assert sdims(s) == 2
@assert colordim(s) == 3
@assert spatialorder(s) == ["y","x"]

# reslicing
D = randn(3,5,4)
sd = SliceData(D, 2)
C = slice(D, sd, 2)
@assert C == reshape(D[1:end, 2, 1:end], size(C))
reslice!(C, sd, 3)
@assert C == reshape(D[1:end, 3, 1:end], size(C))
sd = SliceData(D, 3)
C = slice(D, sd, 2)
@assert C == reshape(D[1:end, 1:end, 2], size(C))

sd = SliceData(imgd, 2)
s = sliceim(imgd, sd, 2)
@assert colordim(s) == 2
@assert colorspace(s) == "RGB"
@assert spatialorder(s) == ["y"]
@assert s.data == reshape(imgd[:,2,:], size(s))
sd = SliceData(imgd, 3)
s = sliceim(imgd, sd, 2)
@assert colordim(s) == 0
@assert colorspace(s) == "Gray"
@assert spatialorder(s) == Images.yx
@assert s.data == imgd[:,:,2]
reslice!(s, sd, 3)
@assert s.data == imgd[:,:,3]

# named indexing
@assert dimindex(imgd, "color") == 3
@assert dimindex(imgd, "y") == 1
@assert dimindex(imgd, "z") == 0
imgdp = permutedims(imgd, [3,1,2])
@assert dimindex(imgdp, "y") == 2
@assert img["y", 2, "x", 4] == B[2,4]
@assert img["x", 4, "y", 2] == B[2,4]
chan = imgd["color", 2]
Blookup = reshape(cmap[B[:],1], size(B))
@assert chan == Blookup

sd = SliceData(imgd, "x")
s = sliceim(imgd, sd, 2)
@assert spatialorder(s) == ["y"]
@assert s.data == reshape(imgd[:,2,:], size(s))
sd = SliceData(imgd, "y")
s = sliceim(imgd, sd, 2)
@assert spatialorder(s) == ["x"]
@assert s.data == reshape(imgd[2,:,:], size(s))
sd = SliceData(imgd, "x", "y")
s = sliceim(imgd, sd, 2, 1)
@assert s.data == reshape(imgd[1,2,:], 3)

# spatial order, width/height, and permutations
@assert spatialpermutation(Images.yx, imgd) == [1,2]
@assert widthheight(imgd) == (5,3)
C = convert(Array, imgd)
@assert C == imgd.data
imgd.properties["spatialorder"] = ["x", "y"]
@assert spatialpermutation(Images.xy, imgd) == [1,2]
@assert widthheight(imgd) == (3,5)
C = convert(Array, imgd)
@assert C == permutedims(imgd.data, [2,1,3])
imgd.properties["spatialorder"] = ["y", "x"]
@assert spatialpermutation(Images.xy, imgd) == [2,1]
imgd.properties["spatialorder"] = ["x", "L"]
@assert spatialpermutation(Images.xy, imgd) == [1,2]
imgd.properties["spatialorder"] = ["L", "x"]
@assert spatialpermutation(Images.xy, imgd) == [2,1]
@assert spatialpermutation(Images.xy, A) == [2,1]
@assert spatialpermutation(Images.yx, A) == [1,2]

imgd.properties["spatialorder"] = Images.yx
imgp = permutedims(imgd, ["x", "y", "color"])
@assert imgp.data == permutedims(imgd.data, [2,1,3])
imgp = permutedims(imgd, ("color", "x", "y"))
@assert imgp.data == permutedims(imgd.data, [3,2,1])
