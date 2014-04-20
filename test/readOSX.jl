using Images, TestImages
using Base.Test

img = testimage("autumn_leaves")
#@assert colorspace(img) == "Gray"
#@assert ndims(img) == 2
#@assert colordim(img) == 0
#@assert eltype(img) == Uint16

img = testimage("cameraman")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("earth_apollo17")
#@assert colorspace(img) == "RGB"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("fabio")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("house")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("jetplane")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("lighthouse")
#@assert colorspace(img) == "RGB"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("mandrill")
#@assert colorspace(img) == "RGB"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("moonsurface")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8

img = testimage("mountainstream")
#@assert colorspace(img) == "RGB"
@assert ndims(img) == 2
@assert colordim(img) == 0
@assert eltype(img) == Uint8
