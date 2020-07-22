using Test
using Suppressor

@testset "Images" begin
include("arrays.jl")
include("algorithms.jl")
@suppress_err include("exposure.jl") # deprecated
include("edge.jl")
include("corner.jl")
include("writemime.jl")
# include("deprecated.jl")

end
