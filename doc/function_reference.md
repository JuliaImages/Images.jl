## Function reference

Below, `[]` in an argument list means an optional argument.

### Image construction

```
Image(data, [properties])
```
creates a new direct image.

<br />
```
ImageCmap(data, cmap, [properties])
```
creates an indexed (colormap) image.

<br />
```
convert(Image, A)
convert(Image{HSV}, img)
convert(Array, img)
```
The first creates a 2d image from an array, setting up default properties. The
data array is assumed to be in "vertical-major" order, and if color is encoded
as a dimension of the array it must be the 3rd dimension. The second syntax
allows you to convert from one colorspace to another.

`convert(Array, img)` works in the opposite direction, permuting dimensions (if
needed) to put it in standard storage order.

<br />
```
grayim(A)
```
creates a 2d or 3d _spatial_ grayscale image, assumed to be in
"horizontal-major" order (and without permuting any dimensions).

<br />
```
copy(img, data)
```
Creates a new image from the data array `data`, copying the properties from
image `img`.

<br />
```
share(img, data)
```
Creates a new image from the data array `data`, _sharing_ the properties of
image `img`. Any modifications made to the properties of one will affect the
other.

<br />
```
similar(img, [type], [dims])
```
Like the standard Julia command, this will create an `Image` of similar type,
copying the properties.

<br />
```
Overlay(channels, colors, clim)
```
Create an `Overlay` array from grayscale channels.
`channels = (channel1, channel2, ...)`,
`colors` is a vector or tuple of `ColorValue`s, and `clim` is a
vector or tuple of min/max values, e.g., `clim = ((min1,max1),(min2,max2),...)`.

<br />
```
OverlayImage(channels, colors, clim)
```
Like `Overlay`, except it creates an Image (not just an array).


### Accessing image data

```
data(img)
```
returns a reference to the array data in the image. It allows you to use an
algorithm written for `Array`s or `AbstractArray`s on `Image` types.
This works for both `AbstractImage`s and `AbstractArray`s (for the latter it
just returns the input), so is a "safe" component of any algorithm. 

<br />
```
img[i, j, k,...]
img["x", 100:200, "y", 400:600]
```
return image data as an array. The latter syntax allows you to address
dimensions by name, irrespective of the storage order. The returned values have
the same storage order as the parent.

<br />
```
getindexim(img, i, j, k,...)
getindexim(img, "x", 100:200, "y", 400:600)
```
return the image data as an `Image`, copying (and where necessary modifying) the
properties of `img`.

<br />
```
sub(img, i, j, k, ...)
sub(img, "x", 100:200, "y", 400:600)
slice(img, i, j, k, ...)
sub(img, "x", 15, "y", 400:600)
```
returns a `SubArray` of image data, with the ordinary meanings of `sub` and
`slice`.

<br />
```
subim(img, i, j, k, ...)
subim(img, "x", 100:200, "y", 400:600)
sliceim(img, i, j, k, ...)
subim(img, "x", 15, "y", 400:600)
```
returns an `Image` with `SubArray` data.

### Image properties

Unless specified, these functions work on both plain arrays (when properties can
be inferred), and on `Image` types.

#### Dictionary-like interface
```
val = img["propertyname"]
img["propertyname"] = val
```
get and set, respectively, the value of a property. These work only for `Image`
types.

<br />
```
haskey(img, "propertyname")
```
Tests whether the named property exists. Returns false for `Array`s.

<br />
```
get(img, "propertyname", defaultvalue)
```
Gets the named property, or returns the default if not present. For `Array`, the
default is always returned.

#### Accessor-function interface

```
assert2d(img)
```
Triggers an error if the image has a time dimension or more than two spatial
dimensions.

<br />
```
assert_scalar_color(img)
```
Triggers an error if the image uses an array dimension to encode color.

<br />
```
assert_xfirst(img)
assert_yfirst(img)
```
Triggers an error if the first spatial dimension is not as specified.

<br />
```
colordim(img)
```
Returns the dimension used to encode color, or 0 if no dimension of the array is
used for color. For example, an `Array` of size `(m, n, 3)` would result in 3,
whereas an `Array` of `RGB` colorvalues would yield 0.

<br />
```
colorspace(img)
```
Returns a string specifying the colorspace representation of the image.

<br />
```
coords_spatial(img)
```
Returns a vector listing the spatial dimensions of the image. For example, an
`Array` of size `(m,n,3)` would return `[1,2]`.

<br />
```
height(img)
```
Returns the vertical size of the image, regardless of storage order. By default
horizontal corresponds to dimension `"y"`, but see `spatialpermutation` for
other options.

<br />
```
isdirect(img)
```
True if `img` encodes its values directly, rather than via an indexed colormap.

<br />
```
isxfirst(img)
isyfirst(img)
```
Tests whether the first spatial dimension is `"x"` or `"y"`, respectively.

<br />
```
limits(img)
```
Returns the value of the `"limits"` property, or infers the limits from the data
type (e.g., `Uint8`) when the property does not exist. Setting a default value
for FloatingPoint types presents a bit of a challenge; for `Image` FloatingPoint
types, the default is `(-Inf, Inf)` (consistent with using `typemin/typemax` for
integer types), but for plain `Array`s the convention is `(0,1)`. See also
`climdefault` which always returns a finite value, defaulting to `(0,1)` for a
FloatingPoint image type for which `"limits"` has not been explicitly set.

For example, if you're using a 14-bit camera and encoding the values with
`Uint16`, you can set the limits property to `(0x0000, 0x3fff)` to indicate that
not all 16 bits are meaningful. 

Note that there is no guarantee that the image values
fall within the stated limits; this is intended as a "hint"
and is used particularly for setting default contrast when images are displayed.

Arithmetic on images updates the setting of the `"limits"` property. For
example, `imgdiff = img1-img2`, where both are `Image`s with `"limits"` set to
`(0.0,1.0)`, will result in `limits(imgdiff) = (-1.0,1.0)`.

<br />
```
pixelspacing(img)
```
Returns a vector containing the spacing between adjacent pixels along each
dimension. Defaults to 1. If desired, you can set this property in terms of
physical [units](https://github.com/timholy/Units.jl).

<br />
```
nimages(img)
```
The number of time-points in the image array. This is safer than
`size(img, "t")` because it also works for plain `AbstractArray` types.

<br />
```
sdims(img)
```
Similar to `ndims`, but it returns just the number of _spatial_ dimensions in
the image array (excluding color and time).

<br />
```
size(img, 2)
size(img, "t")
```
Obtains the size of the specified dimension, even for dimensions specified by
name. See also `nimages`, `size_spatial`, `width`, `height`, and `widthheight`.

<br />
```
size_spatial(img)
```
Returns a tuple listing the sizes of the spatial dimensions of the image. For
example, an `Array` of size `(m,n,3)` would return `(m,n)`.

<br />
```
spatialorder(img)
spatialorder(ImageType)
```
Returns the storage order of the _spatial_ coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`. See
`storageorder`, `timedim` and `colordim` for related properties.

<br />
```
spatialpermutation(to, img)
```
Calculates the _spatial_ permutation needed to convert the spatial dimension to
a given format. This is probably easiest to understand by examples: for an
`Array` `A` of size `(m,n,3)`, `spatialorder(A)` would yield `["y", "x"]`, so
`spatialpermutation(["y", "x"], A) = [1,2]` and
`spatialpermutation(["x", "y"], A) = [2,1]`.
For an image type, here's a demonstration:
```
julia> Aimg = convert(Image, A)
RGB Image with:
  data: 4x5x3 Array{Float64,3}
  properties:
    limits: (0.0,1.0)
    colordim: 3
    spatialorder:  y x
    colorspace: RGB

julia> Ap = permutedims(Aimg, [3, 1, 2])
RGB Image with:
  data: 3x4x5 Array{Float64,3}
  properties:
    limits: (0.0,1.0)
    colordim: 1
    spatialorder:  y x
    colorspace: RGB

julia> spatialpermutation(["x","y"], Ap)
2-element Array{Int64,1}:
 2
 1
```
 
<br />
```
spatialproperties(img)
```
Returns all properties whose values are of the form of an array or tuple, with
one entry per spatial dimension. If you have a custom type with additional
spatial properties, you can set `img["spatialproperties"] = ["property1",
"property2", ...]`. An advantage is that functions that change spatial
dimensions, like `permutedims` and `slice`, will also adjust the properties. The
default is `["spatialorder", "pixelspacing"]`; however, if you override the
setting then these are not included automatically (you'll want to do so
manually, if applicable).

<br />
```
storageorder(img)
```
Returns the complete storage order of the image array, including `"t"` for time
and `"color"` for color.

<br />
```
timedim(img)
```
Returns the dimension used to represent time, or 0 if this is a single image.

<br />
```
width(img)
```
Returns the horizontal size of the image, regardless of storage order. By
default horizontal corresponds to dimension `"x"`, but see `spatialpermutation`
for other options.

<br />
```
widthheight(img)
```
Returns the `(w,h)` tuple. See `width` and `height`.

### Intensity scaling

One can directly rescale the pixel intensities in the image array. One such function is

```
imstretch(img, m, slope)
```
which, for an image with intensities between 0 and 1, enhances or reduces (for
slope > 1 or < 1, respectively) the contrast near saturation (0 and 1). This is
essentially a symmetric gamma-correction. For a pixel of brightness `p`, the new
intensity is `1/(1+m/(p+eps)^slope)`.

However, for very large images (e.g., loaded by memory-mapping) this may be inconvenient
because one may exhaust available memory or need to write a new file on disk.
`Images` supports lazy-evaluation scaling through the `ScaleInfo` abstract type.
The basic syntax is

```
valout = scale(scalei::ScaleInfo, valin)
```
Here `val` can refer to a single pixel's data, or to the entire image array. The
`scale` function converts the input value(s), and is (for example) used in
`ImageView`s display of many image types. The `scalei` input is a type that
determines how the input value is scale and converted to a new type.

Here is how to directly construct the main concrete `ScaleInfo` types:

- `ScaleNone{T}()`, indicating that the only form of scaling is conversion of
types (if `T` is an integer, floating-point types are rounded first). This is
not very safe, as values "wrap around": for example, converting `258` to a
`Uint8` results in `0x02`, which would look dimmer than `255 = 0xff`.

- `ClipMin{To,From}(minvalue)`, `ClipMax{To,From}(maxvalue)`, and
`ClipMinMax{To,From}(minvalue, maxvalue)` create `ScaleInfo` objects that clip
the image at the specified min, max, and min/max values, respectively, before
converting. This is much safer than `ScaleNone`.

- `BitShift{T,N}()`, for scaling by bit-shift operators. `N` specifies the
number of bits to right-shift by. For example you could convert a 14-bit image
to 8-bits using `BitShift{Uint8, 6}()`.

- `ScaleMinMax{To,From}(min, max, scalefactor)` clips the image at the specified
min/max values, subtracts the min value, scales the result by multiplying by
`scalefactor`, and finally converts the type.

- `ScaleAutoMinMax{To}()` will cause images to be dynamically scaled to their
specific min/max values, using the same algorithm for `ScaleMinMax`. When
displaying a movie, the min/max will be recalculated for each frame, so this can
result in inconsistent contrast scaling.

- `ScaleSigned(scalefactor)` multiplies the image by the scalefactor, then clips
to the range `[-1,1]`, leaving it in `Float64` format. This type interacts
with `uint32color` (below) to create magenta (positive)/green (negative) images.

There are also convenience functions:

```
climdefault(img)
```
Returns default contrast limits for images, e.g., `(0x00, 0xff)` for a `Uint8`
image. These are taken from the `"limits"` property if supplied, from
`typemin/typemax` for integer types, and `(0.0, 1.0)` for floating-point
types.

<br />
```
sc(img)
sc(img, min, max)
```
Applies default or specified ScaleMinMax scaling to the image.

<br />
```
scaleinfo(img)
```
returns the default scaling, if not supplied. For `Uint8` and `Int8` images this
is `ScaleNone`; for any other integer type, it is `BitShift{Uint8,N}` or
`BitShift{Int8,N}` where `N` is the number of right-shifts needed to reduce the
largest value to 8-bit (taken from `limits()`). For floating-point images it is
`scaleminmax(img, min, max)`, where `min` and `max` come from `climdefault`
(below).

<br />
```
scaleminmax(To, mn, max)
scaleminmax([To], img, [mn, max])
scaleminmax([To], img, tindex)
```
Creates a `ScaleMinMax` object, converting to the data type specified by `To`
(default `Uint8`), using the specified min/max values (or calculating from
`img`) if not supplied. The `tindex` syntax is useful when processing a movie,
causing the calculation of min/max values to occur just for the specified time
slice. (Otherwise the min/max calculation might take a long time.)

<br />
```
scalesigned(img)
scalesigned(img, tindex)
```
Creates a `ScaleSigned` object, using `1/max(abs(img))` as the scale factor. The
`tindex` syntax is useful when processing a movie,
causing the calculation of the `max(abs(img))` value to occur just for the
specified time slice. (Otherwise the calculation might take a long time.)


<br />
```
truncround(T, val)
```
Performs safe integer conversion, first clipping to the allowed range of integer
type `T`, and rounding if necessary.

<br />
```
float32sc(img)
float64sc(img)
uint8sc(img)
uint16sc(img)
uint32sc(img)
```
Scale and convert an image to the specified element type. The `float` variants
will scale to the range `(0.0,1.0)`, and the integer variants will scale with
the default scaling provided by `scaleinfo`.


### Color conversion

```
convert(Image{ColorValue}, img)
```
as described above.

<br />
```
uint32color(img, [scaleinfo])
uint32color!(buf, img, [scaleinfo])
```
converts to 24-bit RGB or 32-bit RGBA, primarily for use in display. The second
version uses a pre-allocated output buffer, an `Array{Uint32}`.

### Image I/O

See also the more [thorough description](extendingIO.md).

```
add_image_file_format(extension, [magicbytes,] FileType, srcfilename)
```
"Register" a source file that can handle reading/writing image files of the
given format. `FileType` must be a subtype of `ImageFileType`. For those
(unfortunate) formats without magic bytes, you can still register them, but
reading will be based solely on extension name and/or can be forced by
specifying the file type in `imread`.

<br />
```
imread(filename)
imread(filename, FileType)
imread(filename, [FileType,] ColorValue)
imread(stream, FileType)
```
Reads an image, inferring the format from (1) the magic bytes where possible
(even if this doesn't agree with the extension name), and (2) otherwise by the
extension name. The format can also be directly controlled by `FileType`.
Colorspace conversion upon read is supported.

<br />
```
imwrite(img, filename, [FileType,] args...)
```
Write an image, specifying the type by the extension name or (optionally)
directly. Some writers take additional arguments, which you can pass on.

<br />
```
loadformat(FileType)
```
Image file format code (in the `src/io` directory) is not loaded until needed,
usually triggered by reading or writing a file of the given format. You can use
this command to force the format to load, for example to gain access to specific
functions in the corresponding module. See the end of `io.jl` for a list of
`FileType`s.


### Image algorithms

You can perform arithmetic with `Image`s and `ColorValue`s. Algorithms also
include the following functions:

### Linear filtering and padding

```
imfilter(img, kernel, [border, value])
```
filters the image with the given (array) kernel, using boundary conditions
specified by `border` and `value`. See `padarray` below for an explanation of
the boundary conditions. Default is to use `"replicate"` boundary conditions.
Currently 2d only.

<br />
```
imfilter_gaussian(img, sigma)
```
filters the image with a gaussian of the specified width. `sigma` should have
one value per array dimension (any number of dimensions are supported), 0
indicating that no filtering is to occur along that dimension. Uses the Young,
van Vliet, and van Ginkel IIR-based algorithm to provide fast gaussian filtering
even with large `sigma`. Edges are handled by "NA" conditions, meaning the
result is normalized by the number and weighting of available pixels, and
missing data (NaNs) are handled likewise.

<br />
```
imedge(img, [method], [border])
```
Edge-detection filtering. `method` is either `"sobel"` or `"prewitt"`. `border` is any of the boundary conditions specified in `padarray`.

<br />
```
forwarddiffx(img)
backdiffx(img)
forwarddiffy(img)
backdiffy(img)
```
Forward- and backward finite differencing along the x- and y- axes. The size of
the image is preserved, so the first (for `backdiff`) or last (for
`forwarddiff`) row/column will be zero. These currently operate only on matrices
(2d with scalar color).

<br />
```
padarray(img, prepad, postpad, border, value)
```
For an `N`-dimensional array, apply padding on both edges. `prepad` and
`postpad` are vectors of length `N` specifying the number of pixels used to pad
each dimension. `border` is a string, one of `"value"` (to pad with a specific
pixel value), `"replicate"` (to repeat the edge value), `"circular"` (periodic
boundary conditions), `"reflect"` (reflecting boundary conditions, where the
reflection is centered on edge), and `"symmetric"` (reflecting boundary
conditions, where the reflection is centered a half-pixel spacing beyond the
edge, so the edge value gets repeated). Arrays are automatically padded before 


### Filtering kernels

```
gaussian2d(sigma, filtersize)
```
returns a kernel for FIR-based Gaussian filtering. See also `imfilter_gaussian`.

<br />
```
imaverage(filtersize)
```
constructs a boxcar-filter of the specified size.

<br />
```
imdog(sigma)
```
creates a difference-of-gaussians kernel (`sigma`s differing by a factor of
`sqrt(2)`)

<br />
```
imlaplacian(filtersize)
```
returns a kernel for laplacian filtering.

<br />
```
imlog(sigma)
```
returns a laplacian-of-gaussian kernel.

<br />
```
sobel()
prewitt()
```
Return x- and y- Sobel and Prewitt derivative filters.

### Nonlinear filtering and transformation

```
imROF(img, lambda, iterations)
```
Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total
Variation (TV) denoising or TV regularization. `lambda` is the regularization
coefficient for the derivative, and `iterations` is the number of relaxation
iterations taken. 2d only.

### Image statistics

```
minfinite(img)
maxfinite(img)
maxabsfinite(img)
```
Return the minimum and maximum value in the image, respectively, ignoring any values that are not finite (Inf or NaN).

<br />
```
ssd(img1, img2)
ssdn(img1, img2)
```
Sum-of-squared-differences (pixelwise). `ssdn` is the
mean-of-squared-differences (i.e., normalized by `n`, the number of pixels).

<br />
```
sad(img1, img2)
sadn(img1, img2)
```
sum and mean of `abs(img1-img2)`.
