# scaling, ssd
img = Images.grayim(fill(typemax(Uint16), 3, 3))
mapi = Images.mapinfo(Ufixed8, img)
img8 = map(mapi, img)
@assert all(img8 .== typemax(Ufixed8))
mnA, mxA = 1.0, -1.0
while mnA > 0 || mxA < 0
    A = randn(3,3)
    mnA, mxA = extrema(A)
end
offset = 30.0
img = convert(Images.Image, A .+ offset)
mapi = Images.ScaleMinMax(Ufixed8, offset, offset+mxA, 1/mxA)
imgs = map(mapi, img)
@assert minimum(imgs) == 0
@assert maximum(imgs) == 1
@assert eltype(imgs) == Ufixed8
imgs = Images.imadjustintensity(img, [])
mnA = minimum(A)
@assert Images.ssd(imgs, (A.-mnA)/(mxA-mnA)) < eps()
A = reshape(1:9, 3, 3)
B = map(Images.ClampMin(Float32, 3), A)
@assert eltype(B) == Float32 && B == [3 4 7; 3 5 8; 3 6 9]
B = map(Images.ClampMax(Uint8, 7), A)
@assert eltype(B) == Uint8 && B == [1 4 7; 2 5 7; 3 6 7]

A = reinterpret(Ufixed8, [uint8(1:24)], (3, 2, 4))
img = reinterpret(RGB{Ufixed8}, A, (2,4))
@test separate(img) == permutedims(A, (2,3,1))

# color conversion
gray = linspace(0.0,1.0,5) # a 1-dimensional image
gray8 = iround(Uint8, 255*gray)
gray32 = Uint32[uint32(g)<<16 | uint32(g)<<8 | uint32(g) for g in gray8]
imgray = Images.Image(gray, ["colordim"=>0, "colorspace"=>"Gray"])
buf = map(Images.mapinfo(Uint32, imgray), imgray) # Images.uint32color(imgray)
@test buf == gray32
rgb = RGB{Float64}[RGB(g, g, g) for g in gray]
buf = map(Images.mapinfo(Uint32, rgb), rgb) # Images.uint32color(rgb)
@test buf == gray32
r = red(rgb)
@test r == gray
img = Images.Image(reinterpret(RGB24, gray32)) # , ["colordim"=>0, "colorspace"=>"RGB24"])
buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
rgb = repeat(gray, outer=[1,3])
img = Images.Image(rgb, ["colordim"=>2, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
g = green(img)
@test g == gray
rgb = repeat(gray', outer=[3,1])
img = Images.Image(rgb, ["colordim"=>1, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = map(Images.mapinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
b = blue(img)
@test b == gray
