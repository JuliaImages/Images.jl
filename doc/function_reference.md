## Function reference

Below, `[]` in an argument list means an optional argument.

### Image construction

```
Image(data, [properties])
Image(data, prop1=val1, prop2=val2, ...)
```
creates a new direct image. In contrast with `convert`, `grayim` and `colorim`,
this does not permute the data array or attempt to guess any
of the `properties`. If `data` encodes color information along one
of the dimensions of the array (as opposed to using a `ColorValue`
array, from the `Color.jl` package), be sure to specify the
`"colordim"` and `"colorspace"` in `properties`.

<br />
```
ImageCmap(data, cmap, [properties])
```
creates an indexed (colormap) image.

<br />
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


<br />
```
grayim(A)
```
creates a 2d or 3d _spatial_ grayscale Image from an AbstractArray, assumed to be in
"horizontal-major" order (and without permuting any dimensions). If you are working
with 3d grayscale images, usage of this function is strongly recommended. This can fix
errors like one of the following:
```
ERROR: Wrong number of spatial dimensions for plain Array, use an AbstractImage type
ERROR: Cannot infer colorspace of Array, use an AbstractImage type
ERROR: Cannot infer pixelspacing of Array, use an AbstractImage type
```
The main reason for such errors---and the reason that `grayim` is recommended---is the
Matlab-derived convention that a `m x n x 3` array is to be interpreted as RGB.
One might then say that an `m x n x k` array, for `k` different from 3, could be
interpreted as grayscale. However, this would lead to difficult-to-track-down surprises
on the day where `k` happened to be 3 for your grayscale image. Instead, the approach
taken in `Images.jl` is to throw an error, encouraging users to develop the habit of
wrapping their 3d grayscale arrays in an unambiguous `Image` type.

<br />
```
colorim(A, [colorspace])
```
Creates a 2d color image from an AbstractArray, auto-detecting which of the first or
last dimension encodes the color and choosing between "horizontal-" and "vertical-major"
accordingly. `colorspace` defaults to `"RGB"` but could also be e.g. `"Lab"` or `"HSV"`.
If the array represents a 4-channel image, the `colorspace` option is mandatory since
there is no way to automatically distinguish between `"ARGB"` and `"RGBA"`.
If both the first and last dimensions happen to be of size 3 or 4, it is impossible to
guess which one represents color and thus an error is generated.
Thus, if your code should be robust to arbitrary-sized images, you should use the `Image`
constructor directly.

<br />
```
copyproperties(img, data)
```
Creates a new image from the data array `data`, copying the properties from
image `img`.

<br />
```
shareproperties(img, data)
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
Overlay(channels, colors, mapi)
```
Create an `Overlay` array from grayscale channels.
`channels = (channel1, channel2, ...)`,
`colors` is a vector or tuple of `ColorValue`s, and `clim` is a
vector or tuple of min/max values, e.g., `clim = ((min1,max1),(min2,max2),...)`.
Alternatively, you can supply a list of `MapInfo` objects

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
slice(img, "x", 15, "y", 400:600)
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
Triggers an error if the image has more than two spatial
dimensions or has a time dimension.

<br />
```
assert_scalar_color(img)
```
Triggers an error if the image uses an array dimension to encode color.

<br />
```
assert_timedim_last(img)
```
Triggers an error if the image has a time dimension that is not the last dimension.

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
dimension. If this property is not set, it will be computed from `"spacedirections"` if present; otherwise
it defaults to `ones(sdims(img))`. If desired, you can set this property in terms of
physical [units](https://github.com/loladiro/SIUnits.jl).

<br />
```
spacedirections(img)
```
Returns a vector-of-vectors, each indicating the displacement between adjacent pixels along each spatial axis
of the image array, relative to some external coordinate system ("physical coordinates").  For example,
you could indicate that a photograph was taken with the camera tilted 30-degree relative to vertical using
```
img["spacedirections"] = [[0.866025,-0.5],[0.5,0.866025]]
```
If not specified, it will be computed from `pixelspacing(img)`, placing the spacing along the "diagonal".
If desired, you can set this property in terms of
physical [units](https://github.com/loladiro/SIUnits.jl).

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
`storageorder`, `timedim`, and `colordim` for related properties.

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

### Element transformation and intensity scaling

Many images require some type of transformation before you
can use or view them. For example, visualization libraries
work in terms of 8-bit data, so if you're using a 16-bit
scientific camera, your image values will need to be scaled
before display.

One can directly rescale the pixel intensities in the image array.
In general, element-wise transformations are handled by `map` or
`map!`, where the latter is used when you want to provide a pre-allocated output.
You can use an anonymous function of your own design, or,
if speed is paramount, the "anonymous functions" of the
[FastAnonymous](https://github.com/timholy/FastAnonymous.jl) package.

Images also supports "lazy transformations." When loading a very large image,
(e.g., loaded by memory-mapping) you may use or view just a small
portion of it. In such cases, it would be quite wasteful to force transformation
of the entire image, and indeed on might exhaust available memory or
need to write a new file on disk.
`Images` supports lazy-evaluation scaling through the `MapInfo` abstract type.
The basic syntax is

```
valout = map(mapi::MapInfo, valin)
```
Here `val` can refer to a single pixel's data, or to the entire image array.
The `mapi` input is a type that determines how the input value is scale and converted to a new type.

Here is how to directly construct the major concrete `MapInfo` types:

- `MapNone(T)`, indicating that the only form of scaling is conversion to type T.
This is not very safe, as values "wrap around": for example, converting `258` to a
`Uint8` results in `0x02`, which would look dimmer than `255 = 0xff`.

- `ClampMin(T, minvalue)`, `ClampMax(T, maxvalue)`, and
`ClampMinMax(T, minvalue, maxvalue)` create `MapInfo` objects that clamp
pixel values at the specified min, max, and min/max values, respectively, before
converting. Clamping is equivalent to `clampedval = min(max(val, minvalue), maxvalue)`.
This is much safer than `MapNone`.

- `BitShift(T, N)` or `BitShift{T,N}()`, for scaling by bit-shift operators.
`N` specifies the number of bits to right-shift by.
For example you could convert a 14-bit image to 8-bits using `BitShift(Uint8, 6)`.
In general this will be faster than using multiplication.

- `ScaleMinMax(T, min, max, [scalefactor])` clamps the image at the specified
min/max values, subtracts the min value, scales the result by multiplying by
`scalefactor`, and finally converts the type.
If `scalefactor` is not specified, it defaults to scaling the range `[min,max]`
to `[0,1]`.

- `ScaleAutoMinMax(T)` will cause images to be dynamically scaled to their
specific min/max values, using the same algorithm for `ScaleMinMax`. When
displaying a movie, the min/max will be recalculated for each frame, so this can
result in inconsistent contrast scaling.

- `ScaleSigned(T, scalefactor)` multiplies the image by the scalefactor, then clamps
to the range `[-1,1]`. If `T` is a floating-point type, it stays in this representation.
If `T` is `RGB24` or `RGB{Ufixed8}`, then it is encoded as a magenta (positive)/green (negative) image.

<br />

There are also convenience functions:

```
imstretch(img, m, slope)
```
which, for an image with intensities between 0 and 1, enhances or reduces (for
slope > 1 or < 1, respectively) the contrast near saturation (0 and 1). This is
essentially a symmetric gamma-correction. For a pixel of brightness `p`, the new
intensity is `1/(1+m/(p+eps)^slope)`.

<br />

```
sc(img)
sc(img, min, max)
```
Applies default or specified ScaleMinMax scaling to the image.

<br />
```
mapinfo(client, img)
```
returns the default scaling for a specified "client."
For example, clients `RGB24` and `ARGB32` are used for display,
and `Images.ImageMagick` is used when saving to disk.
You could define additional implementations for custom clients.

<br />


### Color conversion

```
convert(Image{ColorValue}, img)
```
as described above. Use `convert(Image{Gray}, img)` to calculate
a grayscale representation of a color image using the
[Rec 601 luma](http://en.wikipedia.org/wiki/Luma_%28video%29#Rec._601_luma_versus_Rec._709_luma_coefficients).


<br />
```
map(mapi, img)
map!(mapi, dest, img)
```
can be used to specify both the form of the result and the algorithm used.


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
imread(filename;[extraprop="",extrapropertynames=false])
imread(filename, FileType;[extraprop="",extrapropertynames=false])
imread(stream, FileType)
```
Reads an image, inferring the format from (1) the magic bytes where possible
(even if this doesn't agree with the extension name), and (2) otherwise by the
extension name. The format can also be directly controlled by `FileType`.

Note that imread will return images in native storage format, e.g., a 2D RGB
image will (for most file formats) be returned as a 2D `RGB` array.
Because file formats are horizontal major, you access the value of pixel
`(x,y)` by `img[x,y]` or `img["x",x,"y",y]`.

When reading with ImageMagick, arbitrary properties of the image can be transfered to the properties dictionary. If `extrapropertynames` is `true`, `imread` just returns a vector of the property names stored in the image. The `extraprop` argument takes a string or a vector of strings and adds the named properties to the dictionary.

<br />
```
imwrite(img, filename, [FileType,] args...)
```
Write an image, specifying the type by the extension name or (optionally)
directly. Some writers take additional arguments, for example
```
imwrite(img, "myimage.jpg", quality=80)
```
to control the quality setting in JPEG compression.

<br />
```
loadformat(FileType)
```
Image file format code (in the `src/io` directory) is not loaded until needed,
usually triggered by reading or writing a file of the given format. You can use
this command to force the format to load, for example to gain access to specific
functions in the corresponding module. See the end of `io.jl` for a list of
`FileType`s.

<br />
```
writemime(io, MIME("image/png"), img; mapi=Images.mapinfo_writemime(img), minpixels=10^4, maxpixels=10^6)
```
Write to stream `io` as a PNG. This is used for display front-ends such as IJulia.
The keyword arguments allow you to specify the `mapinfo` transformation to be used,
the minimum number of pixels used to display the image, and the maximum number of pixels
used, respectively. Shrinking is performed by `restrict`; enlarging is done by duplication
of adjacent pixels.

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
This uses finite-impulse-response (FIR) filtering, and is fast only for
relatively small `kernel`s.

<br />
```
imfilter!(dest, img, kernel)
```
filters the image with the given (array) kernel, storing the output
in the pre-allocated output `dest`. The size of `dest` must not be
greater than the size of the result of `imfilter` with `border = "inner"`,
and it behaves identically.
This uses finite-impulse-response (FIR) filtering, and is fast only for
relatively small `kernel`s.

<br />
```
imfilter_fft(img, kernel, [border, value])
```
filters the image with the given (array) kernel, using an FFT algorithm.
This is slower than `imfilter` for small kernels, but much faster for
large kernels. For Gaussian blur, an even better choice is
`imfilter_gaussian`.

<br />
```
imfilter_gaussian(img, sigma)
```
filters the image with a Gaussian of the specified width. `sigma` should have
one value per array dimension (any number of dimensions are supported), 0
indicating that no filtering is to occur along that dimension. Uses the Young,
van Vliet, and van Ginkel IIR-based algorithm to provide fast gaussian filtering
even with large `sigma`. Edges are handled by "NA" conditions, meaning the
result is normalized by the number and weighting of available pixels, and
missing data (NaNs) are handled likewise.

<br />
```
imfilter_LoG(img, sigma, [border])
```

filters a 2D image with a Laplacian of Gaussian of the specified width. `sigma`
may be a vector with one value per array dimension, or may be a single scalar
value for uniform filtering in both dimensions.  Uses the Huertas and Medioni
separable algorithm.

<br />
```
imgradients(img, [method], [border])
```
Edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`, `"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"` (see the functions of the same name for more information).  `border` is any of the boundary conditions specified in `padarray`.

Returns a tuple containing `x` (horizontal) and `y` (vertical) gradient images of the same size as `img`, calculated using the requested method and border.

<br />
```
magnitude(grad_x, grad_y)
```
Calculates the magnitude of the gradient images given by `grad_x` and `grad_y`.  Equivalent to ``sqrt(grad_x.^2 + grad_y.^2)``.

Returns a magnitude image the same size as `grad_x` and `grad_y`.

<br />
```
phase(grad_x, grad_y)
```
Calculates the rotation angle of the gradient images given by `grad_x` and `grad_y`. Equivalent to ``atan2(-grad_y, grad_x)``.
When both ``grad_x[i]`` and ``grad_y[i]`` are zero, the corresponding angle is set to zero.

Returns a phase image the same size as `grad_x` and `grad_y`, with values in [-pi,pi].

<br />
```
orientation(grad_x, grad_y)
```
Calculates the orientation angle of the strongest edge from gradient images given by `grad_x` and `grad_y`.
Equivalent to ``atan2(grad_x, grad_y)``.
When both `grad_x[i]` and `grad_y[i]` are zero, the corresponding angle is set to zero.

Returns a phase image the same size as `grad_x` and `grad_y`, with values in [-pi,pi].

<br />
```
magnitude_phase(grad_x, grad_y)
```
Convenience function for calculating the magnitude and phase of the gradient images given in `grad_x` and `grad_y`.
Returns a tuple containing the magnitude and phase images.
See `magnitude` and `phase` for details.

<br />
```
imedge(img, [method], [border])
```
Edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`, `"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"` (see the functions of the same name for more information).
`border` is any of the boundary conditions specified in `padarray`.

Returns a tuple `(grad_x, grad_y, mag, orient)`, which are the horizontal gradient, vertical gradient, and the magnitude and orientation of the strongest edge, respectively.

<br />
```
thin_edges(img, gradientangle, [border])
thin_edges_subpix(img, gradientangle, [border])
thin_edges_nonmaxsup(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
thin_edges_nonmaxsup_subpix(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
```
Edge thinning for 2D edge images.
Currently the only algorithm available is non-maximal suppression, which takes an edge image and its gradient angle, and checks each edge point for local maximality in the direction of the gradient.
The returned image is non-zero only at maximal edge locations.

`border` is any of the boundary conditions specified in `padarray`.

In addition to the maximal edge image, the `_subpix` versions of these functions also return an estimate of the subpixel location of each local maxima, as a 2D array or image of `Base.Graphics.Point` objects.
Additionally, each local maxima is adjusted to the estimated value at the subpixel location.

Currently, the `_nonmaxsup` functions are identical to the first two function calls, except that they also accept additional keyword arguments.
`radius` indicates the step size to use when searching in the direction of the gradient; values between 1.2 and 1.5 are suggested (default 1.35).
`theta` indicates the step size to use when discretizing angles in the `gradientangle` image, in radians (default: 1 degree in radians = pi/180).

Example:

    g = rgb2gray(rgb_image)
    gx, gy = imgradients(g)
    mag, grad_angle = magnitude_phase(gx,gy)
    mag[mag .< 0.5] = 0.0  # Threshold magnitude image    
    thinned, subpix =  thin_edges_subpix(mag, gradient)

<br />
```
thin_edges_nonmaxsup!(out, img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
thin_edges_nonmaxsup_subpix!(out, location, img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
```

For advanced usage, these versions will put results into preallocated arrays or images.  `out` must be the same size and type as `img`.  `location` must be an array of type `Graphics.Point` and must be the same size as `img`.

<br />
```
forwarddiffx(img)
backdiffx(img)
forwarddiffy(img)
backdiffy(img)
```
Forward- and backward finite differencing along the x- and y- axes.
The size of the image is preserved, so the first (for `backdiff`) or last (for
`forwarddiff`) row/column will be zero.
These currently operate only on matrices
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
filtering. Use `"inner"` to avoid padding altogether; the output array will
be smaller than the input.

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
ando3()
ando4()
ando4_sep()
ando5()
ando5_sep()
```
Return x- and y- derivative filters of the specified type:

Name          | Description
--------------|------------------------------
`"sobel"`     | Sobel filter
`"prewitt"`   | Prewitt filter
`"ando3"`     | Optimal 3x3 filter from Ando 2000
`"ando4"`     | Optimal 4x4 filter from Ando 2000
`"ando4_sep"` | Separable approximation of `"ando4"`
`"ando5"`     | Optimal 5x5 filter from Ando 2000
`"ando5_sep"` | Separable approximation of `"ando5"`

The ando filters were derived in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.
As written in the paper, the 4x4 and 5x5 papers are not separable, so the `"ando4_sep"` and `"ando5_sep"` filters are provided as separable (and therefore faster) approximations of `"ando4"` and `"ando5"`, respectively.

### Nonlinear filtering and transformation

```
imROF(img, lambda, iterations)
```
Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total
Variation (TV) denoising or TV regularization. `lambda` is the regularization
coefficient for the derivative, and `iterations` is the number of relaxation
iterations taken. 2d only.

### Resizing

```
restrict(img[, region])
```
performs two-fold reduction in size along the dimensions listed in `region`,
or all spatial coordinates if `region` is not specified.
It anti-aliases the image as it goes, so is better than a naive summation
over 2x2 blocks.

### Image statistics

```
minfinite(img)
maxfinite(img)
maxabsfinite(img)
```
Return the minimum and maximum value in the image, respectively, ignoring any values that are not finite (Inf or NaN).

```
meanfinite(img, region)
```
Calculate the mean value along the dimensions listed in `region`, ignoring any non-finite values.

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

### Morphological operations

```
dilate(img, [region])
erode(img, [region])
```
perform a max-filter and min-filter, respectively, over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., region =
[1,2] would exclude the third dimension from filtering.

```
opening(img, [region])
closing(img, [region])
```
perform the `opening` and `closing` morphology operations. `opening` does first 
`erode` the image and then `dilate` the image. `opening` applies both operations 
in the oposite way. The region parameter is passed to `erode` and `dilate` and describes
the kernel size over which these operations are performed.

<br />
```
label_components(tf, [connectivity])
label_components(tf, [region])
```
Find the connected components in a binary array `tf`. There are two forms that
`connectivity` can take:
it can be a boolean array of the same dimensionality as `tf`, of size 1 or 3
along each dimension. Each entry in the array determines whether a given
neighbor is used for connectivity analyses. For example,
```
connectivity = trues(3,3)
```
would use 8-connectivity and test all pixels that touch the current one, even the corners.
The other form is specific for
connectivity to just the nearest neighbors (e.g., 4-connectivity in 2d and
6-connectivity in 3d). You can provide a list indicating which dimensions are
used to determine connectivity. For example, `region = [1,3]` would not test
neighbors along dimension 2 for connectivity.

The default is `region = 1:ndims(A)`.

The output is an integer array, where 0 is used for background pixels, and each
connected region gets a different integer index.

### Phantoms

```
shepp_logan(N,[M]; highContrast=true)
```
output the NxM Shepp-Logan phantom, which is a standard test image usually used for comparing image reconstruction 
algorithms in the field of computed tomography (CT) and magnetic resonance imaging (MRI). If the argument M
is omitted, the phantom is of size NxN. When setting the keyword argument ``highConstrast` to false, the CT
version of the phantom is created. Otherwise, the high contrast MRI version is calculated.
