import Images
using Color

all_close(ar::AbstractImage, v) = all(abs(data(ar)-v) .< sqrt(eps(v)))
all_close(ar, v) = all(abs(ar-v) .< sqrt(eps(v)))

# arithmetic
img = convert(Image, zeros(3,3))
img2 = (img + 3)/2
@assert all(img2 .== 1.5)
img3 = 2img2
@assert all(img3 .== 3)
img3 = copy(img2)
img3[img2 .< 4] = -1
@assert all(img3 .== -1)

# scaling, ssd
img = convert(Image, fill(typemax(Uint16), 3, 3))
scalei = scaleinfo(Uint8, img)
img8 = scale(scalei, img)
@assert all(img8 .== typemax(Uint8))
A = randn(3,3)
mxA = max(A)
offset = 30.0
img = convert(Image, A + offset)
scalei = Images.ScaleMinMax{Uint8, Float64}(offset, offset+mxA, 100/mxA)
imgs = scale(scalei, img)
@assert min(imgs) == 0
@assert max(imgs) == 100
@assert eltype(imgs) == Uint8
imgs = imadjustintensity(img, [])
mnA = min(A)
@assert ssd(imgs, (A-mnA)/(mxA-mnA)) < eps()

# SubVector
A = reshape(1:48, 8, 6)
s = Images.SubVector(A, 1:3:8, 2)
@assert s[2] == A[4,2]
@assert s == A[1:3:8, 2]
s = Images.SubVector(A, 2:3:8, 2)
@assert s == A[2:3:8, 2]
s = Images.SubVector(A, 4, 3:6)
@assert s == vec(A[4, 3:6])
s[1] = 0
@assert A[4,3] == 0
ss = Images.SubVector(s, 4:-2:2)
@assert ss == [44,28]

# filtering
@assert all_close(imfilter(ones(4,4), ones(3,3)), 9.0)
@assert all_close(imfilter(ones(3,3), ones(3,3)), 9.0)
@assert all_close(imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)

img = convert(Image, ones(4,4))
@assert all_close(imfilter(img, ones(3,3)), 9.0)

# color conversion
gray = linspace(0.0,1.0,5) # a 1-dimensional image
gray8 = iround(Uint8, 255*gray)
gray32 = [uint32(g)<<16 | uint32(g)<<8 | uint32(g) for g in gray8]
imgray = Images.Image(gray, ["colordim"=>0, "colorspace"=>"Gray"])
buf = Images.uint32color(imgray)
@assert buf == gray32
rgb = [RGB(g, g, g) for g in gray]
buf = Images.uint32color(rgb)
@assert buf == gray32
img = Images.Image(gray32, ["colordim"=>0, "colorspace"=>"RGB24"])
buf = Images.uint32color(img)
@assert buf == gray32
rgb = repeat(gray, outer=[1,3])
img = Images.Image(rgb, ["colordim"=>2, "colorspace"=>"RGB"])
buf = Images.uint32color(img)
@assert buf == gray32
rgb = repeat(gray', outer=[3,1])
img = Images.Image(rgb, ["colordim"=>1, "colorspace"=>"RGB"])
buf = Images.uint32color(img)
@assert buf == gray32
ovr = Images.Overlay((gray, 0*gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
buf = Images.uint32color(ovr)
nogreen = [uint32(g)<<16 | uint32(g) for g in gray8]
@assert buf == nogreen
ovr = Images.Overlay((gray, gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
ovr.visible[2] = false
buf = Images.uint32color(ovr)
@assert buf == nogreen
