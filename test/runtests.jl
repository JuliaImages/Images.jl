module ImagesTests

using FactCheck, Base.Test, Images, Colors, FixedPointNumbers
using Graphics
using Compat

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end


#=include("core.jl")
include("map.jl")
include("overlays.jl")
include("algorithms.jl")
include("exposure.jl")
include("edge.jl")
include("writemime.jl")
include("corner.jl")
include("distances.jl")=#
include("bwdist.jl")

isinteractive() || FactCheck.exitstatus()

end

include("parallel.jl")
