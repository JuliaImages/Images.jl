using Images
using Statistics
using Test
using ImageBase
using ImageBase.OffsetArrays
using Suppressor

@testset "Images" begin
include("arrays.jl")
include("algorithms.jl")
@suppress_err include("exposure.jl") # deprecated
include("edge.jl")
include("corner.jl")
include("writemime.jl")

@suppress_err include("legacy.jl")
@suppress_err include("deprecated.jl")

end
