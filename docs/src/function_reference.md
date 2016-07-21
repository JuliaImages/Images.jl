# Function Reference

Below, `[]` in an argument list means an optional argument.

# Image construction

```@docs
Image
ImageCmap
```

```
convert(Image, A)
convert(Array, img)
convert(Image{HSV}, img)
```

The first creates a 2d image from an array, setting up default properties. The
data array is assumed to be in "vertical-major" order, and an m-by-n-by-3 array
will be assumed to encode color along its third dimension.

`convert(Array, img)` works in the opposite direction, permuting dimensions (if
needed) to put it in Matlab-standard storage order.

The third syntax allows you to convert from one colorspace to another.

```@docs
grayim
colorim
copyproperties
shareproperties
similar
Overlay
OverlayImage
```

# Accessing image data

```@docs
data
raw
separate
```

img[] (indexing)
```julia
img[i, j, k,...]
img["x", 100:200, "y", 400:600]
```

return image data as an array. The latter syntax allows you to address
dimensions by name, irrespective of the storage order. The returned values have
the same storage order as the parent.

```@docs
getindexim
```

sub and slice
```julia
sub(img, i, j, k, ...)
sub(img, "x", 100:200, "y", 400:600)
slice(img, i, j, k, ...)
slice(img, "x", 15, "y", 400:600)
```

returns a `SubArray` of image data, with the ordinary meanings of `sub` and
`slice`.

subim and sliceim
```julia
subim(img, i, j, k, ...)
subim(img, "x", 100:200, "y", 400:600)
sliceim(img, i, j, k, ...)
subim(img, "x", 15, "y", 400:600)
```

returns an `Image` with `SubArray` data.

# Properties dictionary-like interface

Unless specified, these functions work on both plain arrays (when properties can
be inferred), and on `Image` types.

```julia
val = img["propertyname"]
img["propertyname"] = val
```

get and set, respectively, the value of a property. These work only for `Image`
types.

haskey
```julia
haskey(img, "propertyname")
```

Tests whether the named property exists. Returns false for `Array`s.

get
```julia
get(img, "propertyname", defaultvalue)
```

Gets the named property, or returns the default if not present. For `Array`, the
default is always returned.

# Properties accessor-function interface

Unless specified, these functions work on both plain arrays (when properties can
be inferred), and on `Image` types.

```@docs
assert2d
assert_scalar_color
assert_timedim_last
assert_xfirst
colordim
colorspace
coords_spatial
height
isdirect
isxfirst
isyfirst
pixelspacing
spacedirections
nimages
sdims
```

size
```julia
size(img, 2)
size(img, "t")
```

Obtains the size of the specified dimension, even for dimensions specified by
name. See also `nimages`, `size_spatial`, `width`, `height`, and `widthheight`.

```@docs
size_spatial
spatialorder
spatialpermutation
spatialproperties
storageorder
timedim
width
widthheight
```

# Element transformation and intensity scaling

Many images require some type of transformation before you can use or view
them. For example, visualization libraries work in terms of 8-bit data, so if
you're using a 16-bit scientific camera, your image values will need to be
scaled before display.

One can directly rescale the pixel intensities in the image array.  In general,
element-wise transformations are handled by `map` or `map!`, where the latter is
used when you want to provide a pre-allocated output.  You can use an anonymous
function of your own design, or, if speed is paramount, the "anonymous
functions" of the [FastAnonymous](https://github.com/timholy/FastAnonymous.jl)
package.

Images also supports "lazy transformations." When loading a very large image,
(e.g., loaded by memory-mapping) you may use or view just a small portion of
it. In such cases, it would be quite wasteful to force transformation of the
entire image, and indeed on might exhaust available memory or need to write a
new file on disk.  `Images` supports lazy-evaluation scaling through the
`MapInfo` abstract type.  The basic syntax is

```julia
valout = map(mapi::MapInfo, valin)
```

Here `val` can refer to a single pixel's data, or to the entire image array.
The `mapi` input is a type that determines how the input value is scale and
converted to a new type.

## MapInfo

Here is how to directly construct the major concrete `MapInfo` types:

- `MapNone(T)`, indicating that the only form of scaling is conversion
  to type T.  This can throw an error if a value `x` cannot be
  represented as an object of type `T`, e.g., `map(MapNone{U8}, 1.2)`.

- `ClampMin(T, minvalue)`, `ClampMax(T, maxvalue)`, and
  `ClampMinMax(T, minvalue, maxvalue)` create `MapInfo` objects that
  clamp pixel values at the specified min, max, and min/max values,
  respectively, before converting to type `T`. Clamping is equivalent
  to `clampedval = min(max(val, minvalue), maxvalue)`.

- `BitShift(T, N)` or `BitShift{T,N}()`, for scaling by bit-shift operators.
  `N` specifies the number of bits to right-shift by.  For example you could
  convert a 14-bit image to 8-bits using `BitShift(Uint8, 6)`.  In general this
  will be faster than using multiplication.

- `ScaleMinMax(T, min, max, [scalefactor])` clamps the image at the specified
  min/max values, subtracts the min value, scales the result by multiplying by
  `scalefactor`, and finally converts the type.  If `scalefactor` is not
  specified, it defaults to scaling the range `[min,max]` to `[0,1]`.

- `ScaleAutoMinMax(T)` will cause images to be dynamically scaled to their
  specific min/max values, using the same algorithm for `ScaleMinMax`. When
  displaying a movie, the min/max will be recalculated for each frame, so this
  can result in inconsistent contrast scaling.

- `ScaleSigned(T, scalefactor)` multiplies the image by the scalefactor, then
  clamps to the range `[-1,1]`. If `T` is a floating-point type, it stays in
  this representation.  If `T` is `RGB24` or `RGB{UFixed8}`, then it is encoded
  as a magenta (positive)/green (negative) image.

There are also convenience functions:

```@docs
imstretch
sc
MapInfo
mapinfo
```

# Color conversion

convert
```julia
convert(Image{Color}, img)
```

as described above. Use `convert(Image{Gray}, img)` to calculate
a grayscale representation of a color image using the
[Rec 601 luma](http://en.wikipedia.org/wiki/Luma_%28video%29#Rec._601_luma_versus_Rec._709_luma_coefficients).

map
```julia
map(mapi, img)
map!(mapi, dest, img)
```

can be used to specify both the form of the result and the algorithm used.

# Image I/O

Image loading and saving is handled by the [FileIO](https://github.com/JuliaIO/FileIO.jl) package.

# Image algorithms

You can perform arithmetic with `Image`s and `Color`s. Algorithms also
include the following functions:

# Linear filtering and padding

```@docs
imfilter
imfilter!
imfilter_fft
imfilter_gaussian
imfilter_LoG
imgradients
magnitude
phase
orientation
magnitude_phase
imedge
thin_edges
canny
forwarddiffx
forwarddiffy
backdiffx
backdiffy
padarray
```

# Feature Extraction

```@docs
blob_LoG
findlocalmaxima
findlocalminima
```

# Exposure

```@docs
imhist
histeq
adjust_gamma
```

# Filtering kernels

```@docs
gaussian2d
imaverage
imdog
imlaplacian
imlog
sobel
prewitt
ando3
ando4
ando5
```

# Nonlinear filtering and transformation

```@docs
imROF
imcorner
```

# Resizing

```@docs
restrict
```

# Image statistics

```@docs
minfinite
maxfinite
maxabsfinite
meanfinite
ssd
ssdn
sad
sadn
```

# Morphological operations

```@docs
dilate
erode
opening
closing
tophat
bothat
morphogradient
morpholaplace
label_components
component_boxes
component_lengths
component_indices
component_subscripts
component_centroids
```

# Phantoms

```@docs
shepp_logan
```
