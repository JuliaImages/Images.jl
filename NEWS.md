# v0.6

Images has been rewritten essentially from scratch for this
release. The major goals of the release are:

- More consistent treatment of spatial orientation
- Stop losing key properties upon indexing
- More sensible treatment of indexed (colormap) images
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
  taken, sky coordinates, patient IDs, or experimental conditions.

- Full commitment to the use of `Colorant` types (as defined by the
  `ColorTypes` and `Colors` packages) for encoding color
  images. Arrays are no longer allowed to declare that they use one
  axis (dimension) to store color information, i.e., a `m×n×3 Float32`
  array would be a 3d grayscale image, not an RGB image. This choice
  facilitates efficient and type-stable indexing behavior and enhances
  consistency.  "Lazy" interconversion between arbitrary numeric
  arrays and color arrays are provided by two new view types,
  `colorview` and `channelview`, defined in the `ImageCore` package.
  These types hopefully remove any awkwardness from the new requirement.

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
  `KernelFactors`, both defined in the `ImageFiltering` package.

- Previous versions of Images used `reinterpret` for several
  operations, but `reinterpret` fails for most `AbstractArray`s other
  than `Array`. This release implements alternative mechanisms (e.g.,
  based on the `MappedArrays` package) that work for any
  `AbstractArray` type. (When it would help performance, `reinterpret`
  is still used when applicable.) Consequently, this release features
  better support for a wider range of array types.

# Older versions

For earlier history, please see the git revision history.
