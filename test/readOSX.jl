using Images, TestImages, Color, FixedPointNumbers
using Base.Test

img = testimage("autumn_leaves")
@assert colorspace(img) == "RGBA"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == AlphaColorValue{RGB{Ufixed16}, Ufixed16}

img = testimage("cameraman")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Gray{Ufixed8}

img = testimage("earth_apollo17")
@assert colorspace(img) == "RGB4"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == RGB4{Ufixed8}

img = testimage("fabio")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Gray{Ufixed8}

img = testimage("house")
@assert colorspace(img) == "GrayAlpha"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == AlphaColorValue{Gray{Ufixed8}, Ufixed8}

img = testimage("jetplane")
@assert colorspace(img) == "GrayAlpha"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == AlphaColorValue{Gray{Ufixed8}, Ufixed8}

img = testimage("lighthouse")
@assert colorspace(img) == "RGB4"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == RGB4{Ufixed8}

img = testimage("mandrill")
@assert colorspace(img) == "RGB"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == RGB{Ufixed8}

img = testimage("moonsurface")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Gray{Ufixed8}

img = testimage("mountainstream")
@assert colorspace(img) == "RGB4"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == RGB4{Ufixed8}
