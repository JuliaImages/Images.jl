## Function reference

Below, `[]` in an argument list means an optional argument.

### Image construction

```
Image(data, [properties])
```
creates a new direct image.

```
ImageCmap(data, cmap, [properties])
```
creates an indexed (colormap) image.

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

```
grayim(A)
```
creates a 2d or 3d _spatial_ grayscale image, assumed to be in
"horizontal-major" order (and without permuting any dimensions).

```
copy(img, data)
```
Creates a new image from the data array `data`, copying the properties from
image `img`.

```
share(img, data)
```
Creates a new image from the data array `data`, _sharing_ the properties of
image `img`. Any modifications made to the properties of one will affect the
other.

```
similar(img, [type], [dims])
```
Like the standard Julia command, this will create an `Image` of similar type,
copying the properties.

```
Overlay(channels, colors, clim)
```
Create an `Overlay` array from grayscale channels. `channels = (channel1,
channel2, ...)`,
`colors` is a vector or tuple of `ColorValue`s, and `clim` is a
vector or tuple of min/max values, e.g., `clim = ((min1,max1),(min2,max2),...)`.

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

```
img[i, j, k,...]
img["x", 100:200, "y", 400:600]
```
return image data as an array. The latter syntax allows you to address
dimensions by name, irrespective of the storage order. The returned values have
the same storage order as the parent.

```
getindexim(img, i, j, k,...)
getindexim(img, "x", 100:200, "y", 400:600)
```
return the image data as an `Image`, copying (and where necessary modifying) the
properties of `img`.

```
sub(img, i, j, k, ...)
sub(img, "x", 100:200, "y", 400:600)
slice(img, i, j, k, ...)
sub(img, "x", 15, "y", 400:600)
```
returns a `SubArray` of image data, with the ordinary meanings of `sub` and
`slice`.

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

```
haskey(img, "propertyname")
```
Tests whether the named property exists. Returns false for `Array`s.

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

```
assert_scalar_color(img)
```
Triggers an error if the image uses an array dimension to encode color.

```
assert_xfirst(img)
assert_yfirst(img)
```
Triggers an error if the first spatial dimension is not as specified.

```
coords_spatial(img)
```
Returns a vector listing the spatial dimensions of the image. For example, an
`Array` of size `(m,n,3)` would return `[1,2]`.

```
colordim(img)
```
Returns the dimension used to encode color, or 0 if no dimension of the array is
used for color. For example, an `Array` of size `(m, n, 3)` would result in 3,
whereas an `Array` of `RGB` colorvalues would yield 0.

```
colorspace(img)
```
Returns a string specifying the colorspace representation of the image.

```
height(img)
```
Returns the vertical size of the image, regardless of storage order. By default
horizontal corresponds to dimension `"y"`, but see `spatialpermutation` for
other options.

```
isdirect(img)
```
True if `img` encodes its values directly, rather than via an indexed colormap.

```
isxfirst(img)
isyfirst(img)
```
Tests whether the first spatial dimension is `"x"` or `"y"`, respectively.

```
limits(img)
```
Returns the value of the `limits` property, or infers the limits from the data
type (e.g., `Uint8`) where necessary. For example, if you're using a 14-bit
camera and encoding the values with `Uint16`, you can set the limits property to
`(0x0000, 0x3fff)`.

```
pixelspacing(img)
```
Returns a vector containing the spacing between adjacent pixels along each
dimension. Defaults to 1. If desired, you can set this property in terms of
physical [units](https://github.com/timholy/Units.jl).

```
nimages(img)
```
The number of time-points in the image array. This is safer than `size(img,
"t")` because it also works for plain `AbstractArray` types.

```
sdims(img)
```
Similar to `ndims`, but it returns just the number of _spatial_ dimensions in
the image array (excluding color and time).

```
size(img, 2)
size(img, "t")
```
Obtains the size of the specified dimension, even for dimensions specified by
name. See also `nimages`, `size_spatial`, `width`, `height`, and `widthheight`.

```
size_spatial(img)
```
Returns a tuple listing the sizes of the spatial dimensions of the image. For
example, an `Array` of size `(m,n,3)` would return `(m,n)`.

```
spatialorder(img)
spatialorder(ImageType)
```
Returns the storage order of the _spatial_ coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`. See
`storageorder`, `timedim` and `colordim` for related properties.

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

```
storageorder(img)
```
Returns the complete storage order of the image array, including `"t"` for time
and `"color"` for color.

```
timedim(img)
```
Returns the dimension used to represent time, or 0 if this is a single image.

```
width(img)
```
Returns the horizontal size of the image, regardless of storage order. By
default horizontal corresponds to dimension `"x"`, but see `spatialpermutation`
for other options.

```
widthheight(img)
```
Returns the `(w,h)` tuple. See `width` and `height`.

### Intensity scaling

One can directly convert the pixel intensities in the image array; however, for
very large images (e.g., loaded by memory-mapping) this may be inconvenient.
`Images` supports lazy-evaluation scaling through the `ScaleInfo` abstract type. The basic syntax is

```
valout = scale(scalei::ScaleInfo, valin)
```
Here `val` can refer to a single pixel's data, or to the entire image array.

Here are the main concrete `ScaleInfo` types:

- `ScaleNone{T}`, resulting just in conversion of types (if `T` is an
integer, floating-point types are rounded first). This is not very safe, as
values "wrap around" (e.g., `258` gets converted to `0x02`). 

- `ClipMin{To,From}(minvalue)`, `ClipMax{To,From}(maxvalue)`, and
`ClipMinMax{To,From}(minvalue, maxvalue)` create objects that clip the image at
min, max, and min/max values, respectively, before converting. This is much
safer than `ScaleNone`.

- `BitShift{T,N}`, for scaling by bit-shift operators. `N` specifies the number
of bits to right-shift by. For example you could convert a 14-bit image to
8-bits by `BitShift{Uint8, 6}}`.

- `ScaleMinMax{To,From}(min, max, scalefactor)` clips the image at min/max
values, then scales it by multiplying by `scalefactor` before converting the
type.

- `ScaleSigned(scalefactor)` multiplies the image by the scalefactor, then clips
to the range `[-1,1]`, leaving it in floating-point format.

There are also convenience functions:

```
scaleminmax([To], img, [mn, max])
scaleminmax(To, mn, max)
```
Creates a `ScaleMinMax` object, converting to the data type specified by `To` (default `Uint8`), using the specified min/max values (or calculating from `img`) if not supplied.

```
sc(img)
sc(img, min, max)
```
Applies default or specified ScaleMinMax scaling to the image.

```
truncround(T, val)
```
Performs safe integer conversion, first clipping to the allowed range of integer type `T`, and rounding if necessary.


### Color conversion

```
convert(Image{ColorValue}, img)
```
as described above.

```
uint32color(img)
uint32color!(buf, img)
```
converts to 24-bit RGB or 32-bit RGBA, primarily for use in display. The second
version uses a pre-allocated `Array{Uint32}`.

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

```
imwrite(img, filename, [FileType,] args...)
```
Write an image, specifying the type by the extension name or (optionally)
directly. Some writers take additional arguments, which you can pass on.

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

### Filtering

```
imfilter(img, kernel)
```
filters the image with the given (array) kernel. Currently 2d only.

```
imfilter_gaussian(img, sigma)
```
filters the image with a gaussian of the specified width. `sigma` should have
one value per array dimension (any number of dimensions are supported), 0
indicating that no filtering is to occur along that dimension. Uses the Young,
van Vliet, and van Ginkel IIR-based algorithm to provide fast gaussian filtering
even with large `sigma`. Edges are handled by "NA" conditions, meaning the
result is normalized by the number of available pixels, and NaNs are handled
gracefully.


### Filtering kernels

```
gaussian2d(sigma, filtersize)
```
returns a kernel for FIR-based Gaussian filtering. See also `imfilter_gaussian`.

```
imaverage(filtersize)
```
constructs a boxcar-filter of the specified size.

```
imdog(sigma)
```
creates a difference-of-gaussians kernel (`sigma`s differing by a factor of `sqrt(2)`)

```
imlaplacian(filtersize)
```
returns a kernel for laplacian filtering.

```
imlog(sigma)
```
returns a laplacian-of-gaussian kernel.

### Image statistics

```
ssd(img1, img2)
ssdn(img1, img2)
```
Sum-of-squared-differences (pixelwise). The second is the mean-of-squared-differences.

```
sad(img1, img2)
sadn(img1, img2)
```
sum and mean of `abs(img1-img2)`.
