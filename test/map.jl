using Images, Base.Test, Color, FixedPointNumbers

macro chk(a, b)
    :(@test $a == $b && typeof($a) == typeof($b))
end

# MapNone
mapi = MapNone{Int}()
@chk map(mapi, 7) 7
@chk map(mapi, 0x07) 7
a7 = Int[7]
@chk map(mapi, [0x07]) a7
@test map(mapi, a7) === a7
mapi = MapNone{RGB24}()
g = Ufixed8(0.1)
@chk Images.map1(mapi, 0.1) g
@chk map(mapi, 0.1) RGB24(g,g,g)
mapi = MapNone{RGB{Float32}}()
g = 0.1f0
@chk Images.map1(mapi, 0.1) g
@chk map(mapi, 0.1) RGB(g,g,g)

c = RGB(0.1,0.2,0.3)
mapi = MapNone{HSV{Float64}}()
@chk map(mapi, c) convert(HSV, c)

# BitShift
mapi = BitShift{Uint8,7}()
@chk map(mapi, 0xff) 0x01
@chk map(mapi, 0xf0ff) 0xff
@chk map(mapi, [0xff]) Uint8[0x01]
mapi = BitShift{Ufixed8,7}()
@chk map(mapi, 0xffuf8) 0x01uf8
@chk map(mapi, 0xf0ffuf16) 0xffuf8
mapi = BitShift{ARGB32,4}()
@chk map(mapi, 0xffuf8) ARGB32(0xff0f0f0f)

# Clamp
mapi = ClampMin(Float32, 0.0)
@chk map(mapi,  1.2) 1.2f0
@chk map(mapi, -1.2) 0.0f0
mapi = ClampMax(Float32, 1.0)
@chk map(mapi,  1.2)  1.0f0
@chk map(mapi, -1.2) -1.2f0
mapi = ClampMinMax(Float32, 0.0, 1.0)
@chk map(mapi,  1.2) 1.0f0
@chk map(mapi, -1.2) 0.0f0
mapi = Clamp(Float32)
@chk map(mapi,  1.2) 1.0f0
@chk map(mapi, -1.2) 0.0f0
mapi = ClampMinMax(RGB24, 0.0, 1.0)
@chk map(mapi, 1.2) RGB24(0x00ffffff)
@chk map(mapi, 0.5) RGB24(0x00808080)
@chk map(mapi, -.3) RGB24(0x00000000)
mapi = ClampMinMax(RGB{Ufixed8}, 0.0, 1.0)
@chk map(mapi, 1.2) RGB{Ufixed8}(1,1,1)
@chk map(mapi, 0.5) RGB{Ufixed8}(0.5,0.5,0.5)
@chk map(mapi, -.3) RGB{Ufixed8}(0,0,0)
mapi = Clamp(RGB{Ufixed8})
@chk map(mapi, RGB(1.2,0.5,-.3)) RGB{Ufixed8}(1,0.5,0)
mapi = Clamp(ARGB{Ufixed8})
@chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{Ufixed8}(1.0,0.5,0.0,0.2)
@chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) ARGB{Ufixed8}(1.0,0.5,0.0,0.2)
@chk map(mapi, 0.2) ARGB{Ufixed8}(0.2,0.2,0.2,1.0)
@chk map(mapi, GrayAlpha{Float32}(Gray(0.2),1.2)) ARGB{Ufixed8}(0.2,0.2,0.2,1.0)
@chk map(mapi, GrayAlpha{Float32}(Gray(-.4),0.8)) ARGB{Ufixed8}(0.0,0.0,0.0,0.8)
mapi = Clamp(RGBA{Ufixed8})
@chk map(mapi, ARGB{Float64}(1.2,0.5,-.3,0.2)) RGBA{Ufixed8}(1.0,0.5,0.0,0.2)
@chk map(mapi, RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{Ufixed8}(1.0,0.5,0.0,0.2)
@chk map(mapi, 0.2) RGBA{Ufixed8}(0.2,0.2,0.2,1.0)
@chk map(mapi, GrayAlpha{Float32}(Gray(0.2),1.2)) RGBA{Ufixed8}(0.2,0.2,0.2,1.0)
@chk map(mapi, GrayAlpha{Float32}(Gray(-.4),0.8)) RGBA{Ufixed8}(0.0,0.0,0.0,0.8)
@chk clamp(ARGB{Float64}(1.2,0.5,-.3,0.2)) ARGB{Float64}(1.0,0.5,0.0,0.2)
@chk clamp(RGBA{Float64}(1.2,0.5,-.3,0.2)) RGBA{Float64}(1.0,0.5,0.0,0.2)

# ScaleMinMax
mapi = ScaleMinMax(Ufixed8, 100, 1000)
@chk map(mapi, 100) Ufixed8(0.0)
@chk map(mapi, 1000) Ufixed8(1.0)
@chk map(mapi, 10) Ufixed8(0.0)
@chk map(mapi, 2000) Ufixed8(1.0)
@chk map(mapi, 550) Ufixed8(0.5)
mapinew = ScaleMinMax(Ufixed8, [100,500,1000])
@test mapinew == mapi

# ScaleSigned
mapi = ScaleSigned(Float32, 1/5)
@chk map(mapi, 7) 1.0f0
@chk map(mapi, 5) 1.0f0
@chk map(mapi, 3) convert(Float32, 3/5)
@chk map(mapi, -3) convert(Float32, -3/5)
@chk map(mapi, -6) -1.0f0
mapi = ScaleSigned(RGB24, 1.0f0/10)
@chk map(mapi, 12) RGB24(0x00ff00ff)
@chk map(mapi, -10.0) RGB24(0x0000ff00)
@chk map(mapi, 0) RGB24(0x00000000)

# ScaleAutoMinMax
mapi = ScaleAutoMinMax()
A = [100,550,1000]
@chk map(mapi, A) ufixed8([0.0,0.5,1.0])

# scaling, ssd
img = Images.grayim(fill(typemax(Uint16), 3, 3))
mapi = Images.mapinfo(Ufixed8, img)
img8 = map(mapi, img)
@assert all(img8 .== typemax(Ufixed8))
A = 0
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
