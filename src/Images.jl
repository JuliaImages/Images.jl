module Images

export HomogeneousPoint

using StaticArrays

using Reexport
@reexport using ImageCore
using ImageCore: NumberLike
using ImageCore.OffsetArrays
@reexport using ImageBase

@reexport using FileIO: load, save
import Graphics # TODO: eliminate this direct dependency
import StatsBase  # TODO: eliminate this dependency
using IndirectArrays, ImageCore.MappedArrays

const is_little_endian = ENDIAN_BOM == 0x04030201 # CHECKME(johnnychen94): is this still used?

@reexport using ImageTransformations
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering
@reexport using ImageMorphology
@reexport using ImageDistances
@reexport using ImageContrastAdjustment
@reexport using ImageQualityIndexes
@reexport using IntegralArrays
@reexport using ImageSegmentation

# Non-exported symbol bindings to ImageShow so that we can use, e.g., `Images.gif`
import ImageShow: play, explore, gif

# While we are bridging the old API and the new API in ImageContrastAdjustment
# we need to import these functions because we make new definitions for them
# in deprecations.jl
import ImageContrastAdjustment: build_histogram, adjust_histogram, adjust_histogram!

using TiledIteration: EdgeIterator

# TODO(johnnychen94): (v1.0.0) remove these entry points
# Entry points that isn't used by JuliaImages at all (except for deprecations)
# They used to be accessible by, e.g., `Images.metadata`
import .Colors: Fractional
import FileIO: metadata
import Graphics: Point

include("compat.jl")
include("misc.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("deprecations.jl")
include("corner.jl")
include("edge.jl")

export
    # types
    ColorizedArray,
    Percentile,

    # macros
    @test_approx_eq_sigma_eps,

    # algorithms
    imcorner,
    imcorner_subpixel,
    corner2subpixel,
    harris,
    shi_tomasi,
    kitchen_rosenfeld,
    fastcorners,
    meancovs,
    gammacovs,
    imedge,  # TODO: deprecate?
    imgaussiannoise,
    otsu_threshold,
    yen_threshold,

    #Exposure
    imhist,
    histeq,
    adjust_gamma,
    histmatch,
    clahe,
    imadjustintensity,
    imstretch,
    cliphist,

    magnitude,
    magnitude_phase,
    orientation,
    phase,
    thin_edges,
    thin_edges_subpix,
    thin_edges_nonmaxsup,
    thin_edges_nonmaxsup_subpix,
    canny,
    gaussian_pyramid,

    # phantoms
    shepp_logan

"""
Images.jl is an "umbrella package" that exports a set of packages which are useful for
common image processing tasks. Most of these packages are hosted at JuliaImages,
JuliaArrays, JuliaIO, JuliaGraphics, and JuliaMath.

Images provides an out-of-box toolkit for image processing. This means when you do `using
Images`, you load a lot of packages that might require multiple smaller packages, e.g.,
`using ImageCore, ImageShow, ImageTransformations, FileIO`. However, if you want to reduce
package, you should probably use those small packages for a narrower toolbox customized to
your specific needs.

The documentation for the JuliaImages ecosystem can be found at https://juliaimages.org.
Some dependencies have their own documentation. For instance, the documentation for
Colors.jl is hosted in https://juliagraphics.github.io/Colors.jl even though it is included
and exported by Images.jl.
"""
Images

end
