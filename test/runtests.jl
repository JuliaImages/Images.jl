module ImagesTests

using Compat, FactCheck, Base.Test, Images, Colors, FixedPointNumbers
if VERSION < v"0.4.0-dev+3275"
    using Base.Graphics
else
    using Graphics
end

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end

FactCheck.setstyle(:compact)

include("core.jl")
include("map.jl")
include("overlays.jl")
include("algorithms.jl")
include("edge.jl")

isinteractive() || FactCheck.exitstatus()

end

include("parallel.jl")
