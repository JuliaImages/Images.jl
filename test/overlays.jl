import Images
using Base.Test, Color, FixedPointNumbers

gray = linspace(0.0,1.0,5)
ovr = Images.Overlay((2gray, 2gray), (RGB(1,0,0), RGB(0,0,1)), (Images.Clamp{Float64}(), Images.Clamp{Float64}()))
@test ovr[1] == RGB(0,0,0)
@test ovr[2] == RGB(0.5,0,0.5)
@test ovr[3] == ovr[4] == ovr[5] === RGB(1,0,1)
@test eltype(ovr) == RGB{Float64}
@test length(ovr) == 5
@test size(ovr) == (5,)
@test size(ovr,1) == 5
@test size(ovr,2) == 1
@test Images.nchannels(ovr) == 2
iob = IOBuffer()
show(iob, ovr)

ovr = Images.Overlay((gray, 0*gray), (RGB{Ufixed8}(1,0,1), RGB{Ufixed8}(0,1,0)), ((0,1),(0,1)))
@test eltype(ovr) == RGB{Ufixed8}
s = similar(ovr)
@test isa(s, Vector{RGB{Ufixed8}}) && length(s) == 5
s = similar(ovr, RGB{Float32})
@test isa(s, Vector{RGB{Float32}}) && length(s) == 5
s = similar(ovr, RGB{Float32}, (3,2))
@test isa(s, Matrix{RGB{Float32}}) && size(s) == (3,2)
buf = Images.uint32color(ovr) #scale(Images.scaleinfo(Uint32, ovr), ovr) # Images.uint32color(ovr)
gray8 = uint8(255*gray)
nogreen = [uint32(g)<<16 | uint32(g) for g in gray8]
@test buf == nogreen

ovr = Images.Overlay((gray, 0*gray), [RGB(1,0,1), RGB(0,1,0)], ([0,1],[0,1]))
@test_throws ErrorException Images.Overlay((gray, 0*gray), (RGB(1,0,1), RGB(0,1,0)), ((0,),(0,1)))
@test_throws ErrorException ovr[1] = RGB(0.2,0.4,0.6)

img1 = Images.Image(gray, ["colorspace" => "Gray", "spatialorder" => ["x"]])
ovr = Images.OverlayImage((2gray,img1), (RGB{Ufixed8}(1,0,1), RGB{Ufixed8}(0,1,0)), ((0,1),(0,1)))
@test isa(ovr, Images.Image)
@test !haskey(ovr, "colorspace")
@test Images.colorspace(ovr) == "RGB"
@test ovr[2] == RGB{Float32}(0.5,0.25,0.5)
a = float32(rand(3,2))
b = float32(rand(3,2))
ovr = Images.OverlayImage((a,b), (RGB{Ufixed8}(1,0,1), RGB{Ufixed8}(0,1,0)), ((0,1),(0,1)))
@test isa(ovr, Images.Image)
@test ovr[1,2] == RGB{Float32}(a[1,2],b[1,2],a[1,2])
