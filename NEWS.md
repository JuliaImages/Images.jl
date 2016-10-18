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
  + A compact printing scheme is being tested in the
    `teh/compact_printing` branch; check out the `fixed-renaming`
    branch of many other packages to account for deprecations

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

# Older versions

For earlier history, please see the git revision history.
