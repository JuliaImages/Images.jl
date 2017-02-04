include("arrays.jl")
include("algorithms.jl")
include("exposure.jl")
include("edge.jl")
include("corner.jl")

info("\n\nBeginning of tests with deprecation warnings\n\n")
include("old/runtests.jl")
