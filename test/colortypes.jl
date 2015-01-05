import Images, Images.ColorTypes
using Base.Test, Color, FixedPointNumbers
import Images.Gray, Images.GrayAlpha

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
@test length(zero(GrayAlpha{Float32})) == 2
@test length(one(RGB{Float32})) == 3

# Arithmetic with Gray
cf = Gray{Float32}(0.1)
ccmp = Gray{Float32}(0.2)
@test 2*cf == ccmp
@test cf*2 == ccmp
@test ccmp/2 == cf
@test 2.0f0*cf == ccmp
@test eltype(2.0*cf) == Float64
cu = Gray{Ufixed8}(0.1)
@test 2*cu == Gray(2*cu.val)
@test 2.0f0*cu == Gray(2.0f0*cu.val)
f = Ufixed8(0.5)
@test_approx_eq (f*cu).val f*cu.val
@test 2.*cf == ccmp
@test cf.*2 == ccmp
@test cf/2.0f0 == Gray{Float32}(0.05)
@test cu/2 == Gray(cu.val/2)
@test cu/0.5f0 == Gray(cu.val/0.5f0)
@test cf+cf == ccmp
@test isfinite(cf)
@test !isinf(cf)
@test !isnan(cf)
@test isfinite(Gray(NaN)) == false
@test isinf(Gray(NaN)) == false
@test isnan(Gray(NaN)) == true
@test isfinite(Gray(Inf)) == false
@test isinf(Gray(Inf)) == true
@test isnan(Gray(Inf)) == false
@test_approx_eq abs(Gray(0.1)) 0.1

acu = Gray{Ufixed8}[cu]
acf = Gray{Float32}[cf]
@test typeof(acu+acf) == Vector{Gray{Float32}}
@test typeof(2*acf) == Vector{Gray{Float32}}
@test typeof(uint8(2)*acu) == Vector{Gray{Float32}}
@test typeof(acu/2) == Vector{Gray{typeof(Ufixed8(0.5)/2)}}

# Arithemtic with RGB
cf = RGB{Float32}(0.1,0.2,0.3)
ccmp = RGB{Float32}(0.2,0.4,0.6)
@test 2*cf == ccmp
@test cf*2 == ccmp
@test ccmp/2 == cf
@test 2.0f0*cf == ccmp
@test eltype(2.0*cf) == Float64
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
@test isfinite(cf)
@test !isinf(cf)
@test !isnan(cf)
@test isfinite(RGB(NaN, 1, 0.5)) == false
@test isinf(RGB(NaN, 1, 0.5)) == false
@test isnan(RGB(NaN, 1, 0.5)) == true
@test isfinite(RGB(1, Inf, 0.5)) == false
@test isinf(RGB(1, Inf, 0.5)) == true
@test isnan(RGB(1, Inf, 0.5)) == false
@test_approx_eq abs(RGB(0.1,0.2,0.3)) 0.6

acu = RGB{Ufixed8}[cu]
acf = RGB{Float32}[cf]
@test typeof(acu+acf) == Vector{RGB{Float32}}
@test typeof(2*acf) == Vector{RGB{Float32}}
@test typeof(uint8(2)*acu) == Vector{RGB{Float32}}
@test typeof(acu/2) == Vector{RGB{typeof(Ufixed8(0.5)/2)}}

c = ColorTypes.Gray{Ufixed16}(0.8)
@test convert(RGB, c) == RGB{Ufixed16}(0.8,0.8,0.8)
@test convert(RGB{Float32}, c) == RGB{Float32}(0.8,0.8,0.8)
r4 = ColorTypes.RGB4(1,0,0)
@test convert(RGB, r4) == RGB(1,0,0)
@test convert(RGB{Ufixed8}, r4) == RGB{Ufixed8}(1,0,0)
@test convert(ColorTypes.RGB4{Ufixed8}, r4) == ColorTypes.RGB4{Ufixed8}(1,0,0)
@test convert(ColorTypes.RGB4{Float32}, r4) == ColorTypes.RGB4{Float32}(1,0,0)
@test convert(ColorTypes.BGR{Float32}, r4) == ColorTypes.BGR{Float32}(1,0,0)

iob = IOBuffer()
c = ColorTypes.BGR{Ufixed8}(0.1,0.2,0.301)
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
