module ImagesTests

using Images, Colors, FixedPointNumbers
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
using Graphics

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end


include("core.jl")
include("map.jl")
include("overlays.jl")
include("algorithms.jl")
include("exposure.jl")
include("edge.jl")
include("writemime.jl")
include("corner.jl")
include("distances.jl")

end

include("parallel.jl")
