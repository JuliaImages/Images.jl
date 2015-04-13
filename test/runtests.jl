testing_units = Int == Int64

include("colortypes.jl")
include("core.jl")
include("map.jl")
include("overlays.jl")
include("io.jl")
if testing_units
    include("readnrrd.jl")
end
include("readremote.jl")
include("readstream.jl")
@osx_only include("readOSX.jl")
include("algorithms.jl")
include("edge.jl")
