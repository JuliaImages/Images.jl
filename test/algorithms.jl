import Images, Color, Base.Test

# Comparison of each element in arrays with scalars
approx_equal(ar::Images.AbstractImage, v) = all(abs(Images.data(ar)-v) .< sqrt(eps(v)))
approx_equal(ar, v) = all(abs(ar-v) .< sqrt(eps(v)))

# arithmetic
img = convert(Images.Image, zeros(3,3))
img2 = (img + 3)/2
@assert all(img2 .== 1.5)
img3 = 2img2
@assert all(img3 .== 3)
img3 = copy(img2)
img3[img2 .< 4] = -1
@assert all(img3 .== -1)

# scaling, ssd
img = convert(Images.Image, fill(typemax(Uint16), 3, 3))
scalei = Images.scaleinfo(Uint8, img)
img8 = scale(scalei, img)
@assert all(img8 .== typemax(Uint8))
A = randn(3,3)
mxA = max(A)
offset = 30.0
img = convert(Images.Image, A + offset)
scalei = Images.ScaleMinMax{Uint8, Float64}(offset, offset+mxA, 100/mxA)
imgs = scale(scalei, img)
@assert min(imgs) == 0
@assert max(imgs) == 100
@assert eltype(imgs) == Uint8
imgs = Images.imadjustintensity(img, [])
mnA = min(A)
@assert Images.ssd(imgs, (A-mnA)/(mxA-mnA)) < eps()

# filtering
@assert approx_equal(Images.imfilter(ones(4,4), ones(3,3)), 9.0)
@assert approx_equal(Images.imfilter(ones(3,3), ones(3,3)), 9.0)
@assert approx_equal(Images.imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)
img = convert(Images.Image, ones(4,4))
@assert approx_equal(Images.imfilter(img, ones(3,3)), 9.0)
A = zeros(5,5,3); A[3,3,[1,3]] = 1
@assert colordim(A) == 3
h = [0   0.5 0;
     0.2 1.0 0.2;
     0   0.5 0]
hpad = zeros(5,5); hpad[2:4,2:4] = h
Af = Images.imfilter(A, h)
@test_approx_eq Af cat(3, hpad, zeros(5,5), hpad)
Aimg = permutedims(convert(Images.Image, A), [3,1,2])
@test_approx_eq Images.imfilter(Aimg, h) permutedims(Af, [3,1,2])

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
