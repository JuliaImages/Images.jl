B = rand(1:20,3,5)
cmap = uint8(repmat(linspace(12,255,20),1,3))
img = Images.ImageCmap(B,cmap,["colorspace" => "RGB", "pixelspacing" => [2.0, 3.0]])
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
@assert Images.seqdim(img) == 0
@assert Images.limits(img) == (12,255)
@assert Images.limits(imgd) == (0,255)

