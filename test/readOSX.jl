using FactCheck, Images, Colors, FixedPointNumbers, TestImages

facts("OS X reader") do
    context("Autumn leaves") do
        img = testimage("autumn_leaves")
        @fact colorspace(img) --> "RGBA"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> AlphaColorValue{RGB{Ufixed16}, Ufixed16}
    end
    context("Camerman") do
        img = testimage("cameraman")
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> Gray{Ufixed8}
    end
    context("Earth Apollo") do
        img = testimage("earth_apollo17")
        @fact colorspace(img) --> "RGB4"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> RGB4{Ufixed8}
    end
    context("Fabio") do
    img = testimage("fabio")
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> Gray{Ufixed8}
    end
    context("House") do
        img = testimage("house")
        @fact colorspace(img) --> "GrayAlpha"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> AlphaColorValue{Gray{Ufixed8}, Ufixed8}
    end
    context("Jetplane") do
        img = testimage("jetplane")
        @fact colorspace(img) --> "GrayAlpha"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> AlphaColorValue{Gray{Ufixed8}, Ufixed8}
    end
    context("Lighthouse") do
        img = testimage("lighthouse")
        @fact colorspace(img) --> "RGB4"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> RGB4{Ufixed8}
    end
    context("Mandrill") do
        img = testimage("mandrill")
        @fact colorspace(img) --> "RGB"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> RGB{Ufixed8}
    end
    context("Moonsurface") do
        img = testimage("moonsurface")
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> Gray{Ufixed8}
    end
    context("Mountainstream") do
        img = testimage("mountainstream")
        @fact colorspace(img) --> "RGB4"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> RGB4{Ufixed8}
    end
end
