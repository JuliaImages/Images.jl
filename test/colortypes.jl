import Images, Images.ColorTypes
using Base.Test, Color, FixedPointNumbers

@test length(RGB) == 3
@test length(ColorTypes.BGR) == 3
@test length(AlphaColorValue{ColorTypes.BGR{Float32},Float32}) == 4  # TypeConstructor mess, would be length(BGRA) otherwise
@test length(ColorTypes.AlphaColor{RGB, Ufixed8}) == 4  # length(ARGB)
@test length(ColorTypes.RGB1) == 4
@test length(ColorTypes.RGB4) == 4
@test length(RGB24) == 4
@test length(ColorTypes.Gray) == 1
@test length(AlphaColorValue{ColorTypes.Gray,Ufixed8}) == 2
@test length(ColorTypes.Gray24) == 4
# @test length(ColorTypes.AGray32) == 4
@test length(ColorTypes.ARGB{Float32}) == 4
@test length(ColorTypes.YIQ) == 3
c = ColorTypes.AlphaColor(RGB{Ufixed8}(0.8,0.2,0.4), Ufixed8(0.5))
@test length(c) == 4

cf = RGB{Float32}(0.1,0.2,0.3)
ccmp = RGB{Float32}(0.2,0.4,0.6)
@test 2*cf == ccmp
@test cf*2 == ccmp
@test 2.0f0*cf == ccmp
cu = RGB{Ufixed8}(0.1,0.2,0.3)
@test 2*cu == RGB(2*cu.r, 2*cu.g, 2*cu.b)
@test 2.0f0*cu == RGB(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b)
f = Ufixed8(0.5)
@test_approx_eq (f*cu).r f*cu.r
@test 2.*cf == ccmp
@test cf.*2 == ccmp
@test cf/2.0f0 == RGB{Float32}(0.05,0.1,0.15)
@test cu/2 == RGB(cu.r/2,cu.g/2,cu.b/2)
@test cu/0.5f0 == RGB(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0)
@test cf+cf == ccmp

acu = RGB{Ufixed8}[cu]
acf = RGB{Float32}[cf]
@test typeof(acu+acf) == Vector{RGB{Float32}}
@test typeof(2*acf) == Vector{RGB{Float32}}
@test typeof(uint8(2)*acu) == Vector{RGB{Float32}}
@test typeof(acu/2) == Vector{RGB{typeof(Ufixed8(0.5)/2)}}

c = ColorTypes.Gray{Ufixed16}(0.8)
@test convert(RGB, c) == RGB{Ufixed16}(0.8,0.8,0.8)
@test convert(RGB{Float32}, c) == RGB{Float32}(0.8,0.8,0.8)

iob = IOBuffer()
c = ColorTypes.BGR{Ufixed8}(0.1,0.2,0.3)
show(iob, c)
@test takebuf_string(iob) == "BGR{Ufixed8}(0.102,0.2,0.302)"
c = ColorTypes.Gray{Ufixed16}(0.8)
show(iob, c)
@test takebuf_string(iob) == "Gray{Ufixed16}(0.8)"
ca = AlphaColorValue(c, Ufixed16(0.2))
show(iob, ca)
@test takebuf_string(iob) == "GrayAlpha{Ufixed16}(0.8,0.2)"

# YIQ
@test convert(ColorTypes.YIQ, RGB(1,0,0)) == ColorTypes.YIQ(0.299, 0.595716, 0.211456)
@test convert(ColorTypes.YIQ, RGB(0,1,0)) == ColorTypes.YIQ(0.587, -0.274453, -0.522591)
@test convert(ColorTypes.YIQ, RGB(0,0,1)) == ColorTypes.YIQ(0.114, -0.321263, 0.311135)
@test convert(RGB, ColorTypes.YIQ(1.0,0.0,0.0)) == RGB(1,1,1)
v = 0.5957
@test convert(RGB, ColorTypes.YIQ(0.0,1.0,0.0)) == RGB(0.9563*v,-0.2721*v,-1.1070*v)  # will be clamped to v
v = -0.5226
@test convert(RGB, ColorTypes.YIQ(0.0,0.0,-1.0)) == RGB(0.6210*v,-0.6474*v,1.7046*v)  # will be clamped to v
