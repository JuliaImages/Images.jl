A = reinterpret(Ufixed8, [uint8(1:24)], (3, 2, 4))
img = reinterpret(RGB{Ufixed8}, A, (2,4))
@test separate(img) == permutedims(A, (2,3,1))

# color conversion
gray = linspace(0.0,1.0,5) # a 1-dimensional image
gray8 = iround(Uint8, 255*gray)
gray32 = Uint32[uint32(g)<<16 | uint32(g)<<8 | uint32(g) for g in gray8]
imgray = Images.Image(gray, ["colordim"=>0, "colorspace"=>"Gray"])
buf = scale(Images.scaleinfo(Uint32, imgray), imgray) # Images.uint32color(imgray)
@test buf == gray32
rgb = RGB{Float64}[RGB(g, g, g) for g in gray]
buf = scale(Images.scaleinfo(Uint32, rgb), rgb) # Images.uint32color(rgb)
@test buf == gray32
r = red(rgb)
@test r == gray
img = Images.Image(reinterpret(RGB24, gray32)) # , ["colordim"=>0, "colorspace"=>"RGB24"])
buf = scale(Images.scaleinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
rgb = repeat(gray, outer=[1,3])
img = Images.Image(rgb, ["colordim"=>2, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = scale(Images.scaleinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
g = green(img)
@test g == gray
rgb = repeat(gray', outer=[3,1])
img = Images.Image(rgb, ["colordim"=>1, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = scale(Images.scaleinfo(Uint32, img), img) # Images.uint32color(img)
@test buf == gray32
b = blue(img)
@test b == gray
