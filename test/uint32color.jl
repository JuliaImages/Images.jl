# color conversion
gray = linspace(0.0,1.0,5) # a 1-dimensional image
gray8 = iround(Uint8, 255*gray)
gray32 = [uint32(g)<<16 | uint32(g)<<8 | uint32(g) for g in gray8]
imgray = Images.Image(gray, ["colordim"=>0, "colorspace"=>"Gray"])
buf = Images.uint32color(imgray)
@assert buf == gray32
rgb = RGB{Float64}[RGB(g, g, g) for g in gray]
buf = Images.uint32color(rgb)
@assert buf == gray32
r = red(rgb)
@assert r == gray
img = Images.Image(gray32, ["colordim"=>0, "colorspace"=>"RGB24"])
buf = Images.uint32color(img)
@assert buf == gray32
rgb = repeat(gray, outer=[1,3])
img = Images.Image(rgb, ["colordim"=>2, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = Images.uint32color(img)
@assert buf == gray32
g = green(img)
@assert g == gray
rgb = repeat(gray', outer=[3,1])
img = Images.Image(rgb, ["colordim"=>1, "colorspace"=>"RGB", "spatialorder"=>["x"]])
buf = Images.uint32color(img)
@assert buf == gray32
b = blue(img)
@assert b == gray

ovr = Images.Overlay((gray, 0*gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
buf = Images.uint32color(ovr)
nogreen = [uint32(g)<<16 | uint32(g) for g in gray8]
@assert buf == nogreen
ovr = Images.Overlay((gray, gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
ovr.visible[2] = false
buf = Images.uint32color(ovr)
@assert buf == nogreen
