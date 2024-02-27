using Images
using Statistics
using Test
using ImageBase
using ImageBase.OffsetArrays
using Suppressor
using Aqua

@testset "Aqua" begin
    Aqua.test_all(Images;
                  ambiguities=false,         # TODO? fix ambiguities? (may not be easy/possible) Currently 14 ambiguities
                  undefined_exports=false,   # TODO: remove `data` export from ImageMetadata
                  stale_deps=false,          # ImageMagick and ImageIO are not loaded
                  piracies=false,            # TODO: fix piracy of `zero(Point)` in edges.jl
                  )
end

@testset "Images" begin
    include("arrays.jl")
    include("algorithms.jl")
    include("exposure.jl") # deprecated
    include("edge.jl")
    include("corner.jl")
    include("writemime.jl")

    # @suppress_err include("legacy.jl")
    if Base.JLOptions().depwarn < 2
        @suppress_err include("deprecated.jl")
    else
        @info "Skipping deprecated tests because of --depwarn=error"
    end
end
