using FactCheck, Images, Color, FixedPointNumbers
import Images.ColorTypes, Images.Gray, Images.GrayAlpha

macro test_colortype_approx_eq(a, b)
    :(test_colortype_approx_eq($(esc(a)), $(esc(b)), $(string(a)), $(string(b))))
end

facts("Colortypes") do
    function test_colortype_approx_eq(a::ColorValue, b::ColorValue, astr, bstr)
        @fact typeof(a) --> typeof(b)
        n = length(fieldnames(typeof(a)))
        for i = 1:n
            @fact getfield(a, i) --> roughly(getfield(b,i))
        end
    end

    function test_colortype_approx_eq(a::AbstractAlphaColorValue, b::AbstractAlphaColorValue, astr, bstr)
        @fact typeof(a) --> typeof(b)
        n = length(fieldnames(typeof(a.c)))
        for i = 1:n
            @fact getfield(a.c, i) --> roughly(getfield(b.c, i))
        end
        @fact a.alpha --> roughly(b.alpha)
    end

    context("General") do
        @fact length(RGB) --> 3
        @fact length(ColorTypes.BGR) --> 3
        @fact length(AlphaColorValue{ColorTypes.BGR{Float32},Float32}) --> 4  # TypeConstructor mess, would be length(BGRA) otherwise
        @fact length(ColorTypes.AlphaColor{RGB, Ufixed8}) --> 4  # length(ARGB)
        @fact length(ColorTypes.RGB1) --> 4
        @fact length(ColorTypes.RGB4) --> 4
        @fact length(RGB24) --> 4
        @fact length(ColorTypes.Gray) --> 1
        @fact length(AlphaColorValue{ColorTypes.Gray,Ufixed8}) --> 2
        @fact length(ColorTypes.Gray24) --> 4
        # @fact length(ColorTypes.AGray32) --> 4
        @fact length(ColorTypes.ARGB{Float32}) --> 4
        @fact length(ColorTypes.YIQ) --> 3
        c = ColorTypes.AlphaColor(RGB{Ufixed8}(0.8,0.2,0.4), Ufixed8(0.5))
        @fact length(c) --> 4
        @fact length(zero(GrayAlpha{Float32})) --> 2
        @fact length(one(RGB{Float32})) --> 3
    end

    context("Arithmetic with Gray") do
        cf = Gray{Float32}(0.1)
        ccmp = Gray{Float32}(0.2)
        @fact 2*cf --> ccmp
        @fact cf*2 --> ccmp
        @fact ccmp/2 --> cf
        @fact 2.0f0*cf --> ccmp
        @test_colortype_approx_eq cf*cf Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^2 Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^3.0f0 Gray{Float32}(0.001)
        @fact eltype(2.0*cf) --> Float64
        cu = Gray{Ufixed8}(0.1)
        @fact 2*cu --> Gray(2*cu.val)
        @fact 2.0f0*cu --> Gray(2.0f0*cu.val)
        f = Ufixed8(0.5)
        @fact (f*cu).val --> roughly(f*cu.val)
        @fact 2.*cf --> ccmp
        @fact cf.*2 --> ccmp
        @fact cf/2.0f0 --> Gray{Float32}(0.05)
        @fact cu/2 --> Gray(cu.val/2)
        @fact cu/0.5f0 --> Gray(cu.val/0.5f0)
        @fact cf+cf --> ccmp
        @fact cf --> isfinite
        @fact cf --> not(isinf)
        @fact cf --> not(isnan)
        @fact Gray(NaN) --> not(isfinite)
        @fact Gray(NaN) --> not(isinf)
        @fact Gray(NaN) --> isnan
        @fact Gray(Inf) --> not(isfinite)
        @fact Gray(Inf) --> isinf
        @fact Gray(Inf) --> not(isnan)
        @fact abs(Gray(0.1)) --> roughly(0.1)
        @fact eps(Gray{Ufixed8}) --> Gray(eps(Ufixed8))  # #282
    
        acu = Gray{Ufixed8}[cu]
        acf = Gray{Float32}[cf]
        @fact typeof(acu+acf) --> Vector{Gray{Float32}}
        @fact typeof(2*acf) --> Vector{Gray{Float32}}
        @fact typeof(0x02*acu) --> Vector{Gray{Float32}}
        @fact typeof(acu/2) --> Vector{Gray{typeof(Ufixed8(0.5)/2)}}
        @fact typeof(acf.^2) --> Vector{Gray{Float32}}
    end
    
    context("Arithmetic with GrayAlpha") do
        p1 = GrayAlpha{Float32}(Gray(0.8), 0.2)
        p2 = GrayAlpha{Float32}(Gray(0.6), 0.3)
        @test_colortype_approx_eq p1+p2 GrayAlpha{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq (p1+p2)/2 GrayAlpha{Float32}(Gray(0.7),0.25)
        @test_colortype_approx_eq 0.4f0*p1+0.6f0*p2 GrayAlpha{Float32}(Gray(0.68),0.26)
        @test_colortype_approx_eq ([p1]+[p2])[1] GrayAlpha{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1].+[p2])[1] GrayAlpha{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1]/2)[1] GrayAlpha{Float32}(Gray(0.4),0.1)
        @test_colortype_approx_eq (0.4f0*[p1]+0.6f0*[p2])[1] GrayAlpha{Float32}(Gray(0.68),0.26)
    end
    
    context("Arithemtic with RGB") do
        cf = RGB{Float32}(0.1,0.2,0.3)
        ccmp = RGB{Float32}(0.2,0.4,0.6)
        @fact 2*cf --> ccmp
        @fact cf*2 --> ccmp
        @fact ccmp/2 --> cf
        @fact 2.0f0*cf --> ccmp
        @fact eltype(2.0*cf) --> Float64
        cu = RGB{Ufixed8}(0.1,0.2,0.3)
        @fact 2*cu --> RGB(2*cu.r, 2*cu.g, 2*cu.b)
        @fact 2.0f0*cu --> RGB(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b)
        f = Ufixed8(0.5)
        @fact (f*cu).r --> roughly(f*cu.r)
        @fact 2.*cf --> ccmp
        @fact cf.*2 --> ccmp
        @fact cf/2.0f0 --> RGB{Float32}(0.05,0.1,0.15)
        @fact cu/2 --> RGB(cu.r/2,cu.g/2,cu.b/2)
        @fact cu/0.5f0 --> RGB(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0)
        @fact cf+cf --> ccmp
        @test_colortype_approx_eq (cf*[0.8f0])[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq ([0.8f0]*cf)[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq (cf.*[0.8f0])[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq ([0.8f0].*cf)[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @fact cf --> isfinite
        @fact cf --> not(isinf)
        @fact cf --> not(isnan)
        @fact RGB(NaN, 1, 0.5) --> not(isfinite)
        @fact RGB(NaN, 1, 0.5) --> not(isinf)
        @fact RGB(NaN, 1, 0.5) --> isnan
        @fact RGB(1, Inf, 0.5) --> not(isfinite)
        @fact RGB(1, Inf, 0.5) --> isinf
        @fact RGB(1, Inf, 0.5) --> not(isnan)
        @fact abs(RGB(0.1,0.2,0.3)) --> roughly(0.6)

        acu = RGB{Ufixed8}[cu]
        acf = RGB{Float32}[cf]
        @fact typeof(acu+acf) --> Vector{RGB{Float32}}
        @fact typeof(2*acf) --> Vector{RGB{Float32}}
        @fact typeof(convert(UInt8, 2)*acu) --> Vector{RGB{Float32}}
        @fact typeof(acu/2) --> Vector{RGB{typeof(Ufixed8(0.5)/2)}}

        c = ColorTypes.Gray{Ufixed16}(0.8)
        @fact convert(RGB, c) --> RGB{Ufixed16}(0.8,0.8,0.8)
        @fact convert(RGB{Float32}, c) --> RGB{Float32}(0.8,0.8,0.8)
        r4 = ColorTypes.RGB4(1,0,0)
        @fact convert(RGB, r4) --> RGB(1,0,0)
        @fact convert(RGB{Ufixed8}, r4) --> RGB{Ufixed8}(1,0,0)
        @fact convert(ColorTypes.RGB4{Ufixed8}, r4) --> ColorTypes.RGB4{Ufixed8}(1,0,0)
        @fact convert(ColorTypes.RGB4{Float32}, r4) --> ColorTypes.RGB4{Float32}(1,0,0)
        @fact convert(ColorTypes.BGR{Float32}, r4) --> ColorTypes.BGR{Float32}(1,0,0)

        iob = IOBuffer()
        c = ColorTypes.BGR{Ufixed8}(0.1,0.2,0.301)
        show(iob, c)
        @fact takebuf_string(iob) --> "BGR{Ufixed8}(0.102,0.2,0.302)"
        c = ColorTypes.Gray{Ufixed16}(0.8)
        show(iob, c)
        @fact takebuf_string(iob) --> "Gray{Ufixed16}(0.8)"
        ca = AlphaColorValue(c, Ufixed16(0.2))
        show(iob, ca)
        @fact takebuf_string(iob) --> "GrayAlpha{Ufixed16}(0.8,0.2)"
    end
    
    context("YIQ") do
        @fact convert(ColorTypes.YIQ, RGB(1,0,0)) --> ColorTypes.YIQ(0.299, 0.595716, 0.211456)
        @fact convert(ColorTypes.YIQ, RGB(0,1,0)) --> ColorTypes.YIQ(0.587, -0.274453, -0.522591)
        @fact convert(ColorTypes.YIQ, RGB(0,0,1)) --> ColorTypes.YIQ(0.114, -0.321263, 0.311135)
        @fact convert(RGB, ColorTypes.YIQ(1.0,0.0,0.0)) --> RGB(1,1,1)
        v = 0.5957
        @fact convert(RGB, ColorTypes.YIQ(0.0,1.0,0.0)) --> RGB(0.9563*v,-0.2721*v,-1.1070*v)  # will be clamped to v
        v = -0.5226
        @fact convert(RGB, ColorTypes.YIQ(0.0,0.0,-1.0)) --> RGB(0.6210*v,-0.6474*v,1.7046*v)  # will be clamped to v
    end

end
