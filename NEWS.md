# Images Release Notes

For the full changes, please check the git history and the [release page](https://github.com/JuliaImages/Images.jl/releases)

# v0.25

This release introduces a few major changes that everyone should be aware of:

- drops compatibility to Julia 1.0. Julia at least 1.3 is required.
- drops compatibility to ImageCore 0.8. ImageCore at least 0.9.3 is required.
- revisited RGB-related operations to provide non-ambiguious implementation. See also the "abs and
  abs2" section in the [ColorVectorSpace README][ColorVectorSpace-v09-readme-abs].
- revisits and moves a lot of legacy codes in `src/algorthms.jl` of Images to sub-packages, e.g.,
  ImageBase, ImageFiltering, and ImageMorphology.
- for a large number of legacy functions, positional arguments are deprecated in favor of their
  keyword alternatives.

Because there are a lot of deprecations introduced in this release, we recommend people to run under
`julia --depwarn=yes` mode and fixes the deprecations.

There are also a lot of compatibility changes and probably would make this version incompatible with
other ecosystem. Check the result of `git diff v0.24.1 v0.25.0 -- Project.toml` for more
information.

The following list summarizes some note-worthy changes for things that used to live in Images:

- ![BREAKING][badge-breaking] for `RGB` input, `maximum_finite` and the deprecated `maxabsfinite`
  now returns RGB instead of numerical scalar value. ([Images#971])
- ![Deprecation][badge-deprecation] deprecate `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy` in favor of the
  generic and GPU-ready `fdiff` from `ImageBase.FiniteDiff`. ([ImageBase#11], [Images#971])
- ![Deprecation][badge-deprecation] deprecate non-exported `div` in favor of `fdiv` from `ImageBase` ([Images#971])
- ![Deprecation][badge-deprecation] deprecate `minfinite`/`maxfinite`/`maxabsfinite` in favor of
  `minimum_finite` and `maximum_finite`. ([Images#971])
- ![Deprecation][badge-deprecation] For RGB types `std` and `var` are deprecated in favor of
  `stdmult` and `varmult`. For other colorful types (e.g., `HSV`), `std` and `var` support for
  them will be removed in future releases with no substitutes. ([Images#971])
- ![Deprecation][badge-deprecation] `bilinear_interpolation` is deprecated in favor of `imresize`
  from `ImageTransformations`. ([Images#971])
- ![Deprecation][badge-deprecation] `imROF` is deprecated in favor of the generic and GPU-ready
  `solve_ROF_PD` from `ImageFiltering.Models` ([ImageFiltering#233], [Images#971])
- ![Deprecation][badge-deprecation] deprecate `ColorizedArray` in favor of `mappedarray` from MappedArrays. ([Images#927])
- ![Deprecation][badge-deprecation] deprecate `imaverage` in favor of `Kernel.box` from ImageFiltering. ([Images#971])
- ![Deprecation][badge-deprecation] deprecate `imlaplacian` in favor of `imlaplacian2D` from ImageFiltering. ([Images#971])
- ![Deprecation][badge-deprecation] deprecate `integral_image` and `boxdiff` in favor of IntegralArrays. ([Images#971])

[Images#927]: https://github.com/JuliaImages/Images.jl/pull/927
[Images#971]: https://github.com/JuliaImages/Images.jl/pull/971
[ImageBase#11]: https://github.com/JuliaImages/ImageBase.jl/pull/11
[ImageBase#22]: https://github.com/JuliaImages/ImageBase.jl/pull/22
[ImageFiltering#233]: https://github.com/JuliaImages/ImageFiltering.jl/pull/233
[ColorVectorSpace-v09-readme-abs]: https://github.com/JuliaGraphics/ColorVectorSpace.jl/blob/fc53c5504c6917ea02bb05308c372451068dcada/README.md#abs-and-abs2

# v0.14

- add `OffsetArray` to `REQUIRE`

- add `OffsetArray` support to `gaussian_pyramid`.

# v0.10

New features:

- added `Percentile` to disambiguate the interpretation of
  thresholds. A raw number `x` will now be interpreted as an absolute
  threshold, whereas `Percentile(x)` (with `0 <= x <= 100`) will
  choose an absolute threshold based on the distribution of values in
  the input array.

API changes:

- `canny` passes the threshold as a 2-tuple and uses `Percentile`
  rather than a `percentile` keyword. It now issues a deprecation
  warning when used with default arguments.

- `imcorner` now uses `Percentile`. The old syntax issues a deprecation warning.

# v0.9

Breaking changes:
- Return type of `canny` is now an `Array{Bool}`

Feature additions:
- `convexhull`
- MIME"text/html" output for arrays-of-images

# v0.7

Add `feature_transform` and `distance_transform`

# v0.6

Images has been rewritten essentially from scratch for this
release. The major goals of the release are:

- More consistent treatment of spatial orientation
- Preserve key properties upon indexing
- Intuitive handling of indexed (colormap) images
- Better support for a wider range of array types
- Improvements in type-stability of many operations
- Improvements in the user experience through easier interfaces, more
  informative error messages, and improved printing/display
- Improvements in documentation
- For users of former releases of Images, as smooth an upgrade path as
  can be practically provided, through deprecations or informative
  error messages

Key changes (of which many are breaking):

- Many properties that were formerly in the dictionary (colorspace,
  spatial orientation, pixel spacing, and the presence/absence of a
  time dimension) are now encoded by the type system. The `ImageAxes`
  package (a small extension of `AxisArrays`) is now used for several
  of these properties. This fixes the former loss of information like
  spatial orientation when images were indexed (`img[5:20, 5:20])`.

- The `Image` type (an array + dictionary) has been renamed
  `ImageMeta`, and should be much less needed now that most properties
  can be encoded with `AxisArrays`. `ImageMeta` is still useful if you
  need to encode information like date/time at which the image was
  taken, sky coordinates, patient IDs, or experimental
  conditions. Otherwise, it's recommended to use regular `Array`s, or
  `AxisArrays` if you need to endow axes with "meaning."

- Full commitment to the use of `Colorant` types (as defined by the
  `ColorTypes` and `Colors` packages) for encoding color
  images. Arrays are no longer allowed to declare that they use one
  axis (dimension) to store color information, i.e., a `m×n×3 Float32`
  array would be displayed as a 3d grayscale image, not an RGB
  image. This choice facilitates efficient and type-stable indexing
  behavior and enhances consistency.  "Lazy" interconversion between
  arbitrary numeric arrays and color arrays are provided by two new
  view types, `colorview` and `channelview`, defined in the
  `ImageCore` package.  These types hopefully remove any awkwardness
  from the new requirement.

- For an indexed (colormap) image `imgi`, indexing with `imgi[i,j]`
  used to return the *index*, not the *pixel value*. This operation now
  returns the *pixel value*, with the consequence that indexed images
  largely act like the array they represent. Indexed images are defined
  and handled by the `IndirectArrays` package.

- Image filtering has been greatly improved. `imfilter_fft` and
  `imfilter_gaussian` have both been rolled into `imfilter`. FFT/FIR
  filtering is chosen automatically (though a choice can be specified)
  depending on kernel size, aiming for the best performance in all
  cases and a consistent interface for specifying defaults. The main
  filtering algorithms have been considerably improved, particularly
  for separable kernels, and feature cache-efficient tiling and
  multithreading. The performance enhancement is as large as 10-fold
  in some cases, particularly when starting Julia with multiple
  threads. Certain constructs for specifying boundary conditions have
  been deprecated and replaced with dispatch-leveraging
  alternatives. Specification of standard kernels has been changed
  considerably, and has been split out into two modules, `Kernel` and
  `KernelFactors`, both defined in the `ImageFiltering` package. In
  particular note the `IIRGaussian` types which contain the
  functionality that was formerly in `imfilter_gaussian`.

- Nonlinear filtering operations have been added with
  `mapwindow`. Among the supported functions is `median!`, thus
  providing an implementation of median-filtering.

- Previous versions of Images used `reinterpret` for several
  operations, but `reinterpret` fails for most `AbstractArray`s other
  than `Array`. This release implements alternative mechanisms (e.g.,
  based on the `MappedArrays` package) that work for any
  `AbstractArray` type. (When it would help performance, `reinterpret`
  is still used when applicable.) Consequently, this release features
  better support for a wider range of array types.

- Several improvements have been made to the handling of fixed-point
  numbers, which permit the use of 8- and 16-bit types that act
  similarly to floating-point numbers and which permit a consistent
  criterion for "black" (0.0) and "white" (1.0) independent of storage
  type. Specifically:

  + Trying to convert out-of-bounds values now gives an informative
    error message rather than just `InexactError`
  + Several bugs in FixedPointNumber operations have been fixed, and
    such operations are more consistent about return types
  + FixedPointNumbers are now printed more compactly

- A new package, `ImageTransformations`, is underway for rotation,
  resizing, and other geometric operations on images.

- Many deprecation warnings were designed to help users of the current
  Images package transition to the new framework.

Other changes (all of which are breaking):

- The gradient components returned by `imgradients` match the
  dimensions of the input; in `g1, g2, ... = imgradients(img,
  ...)`, `g1` corresponds to the gradient along the first dimension,
  `g2` along the second, and so on.

- `sobel` and other filters have been normalized so that the returned
  "gradient components" are scaled to estimate the actual
  derivatives. For example, for `sobel` the normalization factor is
  1/8 compared to earlier releases.

- `extrema_filter` has been deprecated in favor of
  `mapwindow(extrema, A, window)`. However, this returns an array of
  `(min,max)` tuples rather than separate `min`, `max` arrays. This is
  intended to transition towards a future API where one can pass `min`
  or `max` in place of `extrema` to obtain just one of
  these. Currently, you can retrieve the `min` array with `first.(mm)`
  and the `max` array with `last.(mm)`.

- The old `extrema_filter` discards the edges of the image, whereas
  the new one (based on `mapwindow`) returns an array of the same size as the input.

- The output of `blob_LoG` is now a `Vector{BlobLoG}`, a new exported
  immutable, rather than the old tuple format.

- `findlocalextrema` now returns a `Vector{CartesianIndex{N}}` rather
  than a `Vector{NTuple{N,Int}}`. This makes it ready for use in efficient
  indexing.

Changes in related packages:

- NRRD.jl has been extensively revamped. The NRRD format lacks an
  official test suite, and hence it was always uncertain how well the
  package supported "the standard" (to the extent that there is
  one). However, it was discovered that `unu make` can generate files
  that can serve as a test suite, and using this strategy several
  incompatibilities in our former version were noted and fixed. It is
  possible that old .nrrd files written by julia might not be readable
  without making manual edits to the header.

# Older versions

For earlier history, please see the git revision history.

<!-- common URLs -->

[ImageBase]: https://github.com/JuliaImages/ImageBase.jl

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Security][badge-security]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
-->
