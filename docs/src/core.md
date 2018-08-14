# Julia Images Guide

## The basic types

### Plain arrays

Images can be plain arrays, which are interpreted to be in "Matlab format": the
first two dimensions are height (`h`) and width (`w`), a storage order here
called "vertical-major". This ordering is inspired by the column/row index order
of matrices and the desire to have a displayed image look like what one sees
when a matrix is written out in text.

If you're working with RGB color, your best approach is to encode color as a
`Color`, as defined in the `Color` package.  That package provides many
utility functions for analyzing and manipulating colors.  Alternatively, you can
use a third dimension of size 3, or encode your images as either `RGB24` or
`ARGB32`, which use an internal `Uint32` representation of color.

It's worth noting that these Matlab conventions are sometimes inconvenient.  For
example, the `x` coordinate (horizontal) is second and the `y` coordinate
(vertical) is first; in other words, one uses `img[y,x]` to address a pixel that
is displayed at a particular `x,y` position. This often catches newcomers (and
sometimes even old-timers) by surprise.  Moreover, most image file formats,
cameras, and graphics libraries such as Cairo use "horizontal-major" storage of
images, and have the color dimension first (fastest). The native Image
type---which allows arbitrary ordering of the data array---permits you to use
this raw representation directly, but when using plain arrays you need to
permute the dimensions of the raw data array.

The convention that a `m x n x 3` array implies RGB is also problematic for
anyone doing 3d imaging, and can result in hard-to-find bugs when the third
dimension happens to be of size 3. For 3d imaging, the use of an Image
type---perhaps converting Arrays via `grayim`---is highly recommended.

The conventions for plain arrays are "baked in" via a few simple utility
functions in the file `core.jl`; if you really need to use plain arrays but want
to work with different conventions, you can (locally) change these defaults with
just a few lines. Algorithms which have been written generically should continue
to work.

However, a more flexible approach is to use one of the self-documenting image
types.

### Image types

All image types should descend from `AbstractImage`, an abstract base type used
to indicate that an array is to be interpreted as an image. If you're writing a
custom image type, it is more likely that you'll want to derive from either
`AbstractImageDirect` or `AbstractImageIndexed`. The former is for direct images
(where intensity at a pixel is represented directly), the latter for indexed
images (where intensity is looked up in a colormap table).

In practice, it is assumed that `AbstractImages` have at least two fields,
called `data` and `properties`. (In code, you should not use these directly,
instead using the functions `data` and `properties` to extract these.)  These
are the only two fields in the first concrete image type, called `Image`:

```julia
mutable struct Image{T,N,A<:AbstractArray} <: AbstractImageDirect{T,N}
    data::A
    properties::Dict{String,Any}
end
```

`data` stores the actual image data, and is an `AbstractArray`. This fact alone
is the basis for a great deal of customizability: `data` might be a plain
`Array` stored in memory, a `SubArray`, a memory-mapped array (which is still
just an `Array`), a custom type that stores additional information about
"missing data" (like bad pixels or dropped frames), or a custom type that
seamlessly presents views of a large number of separate files.  One concrete
example in the Images codebase is the color `Overlay` [type](overlays.html).  If
you have a suitably-defined `AbstractArray` type, you can probably use `Image`
without needing to create alternative `AbstractImageDirect` types.


`properties` is a dictionary, with `String` keys, that allows you to annotate
images. More detail about this point can be found below.

The only other concrete image type is for indexed images:

```julia
mutable struct ImageCmap{T,N,A<:AbstractArray,C<:AbstractArray} <: AbstractImageIndexed{T,N}
    data::A
    cmap::C
    properties::Dict{String,Any}
end
```

The `data` array here just encodes the index used to look up the color in the
`cmap` field.

## Addressing image data

For any valid image type, `data(img)` returns the array that corresponds to the
image.  This works when `img` is a plain `Array` (in which case no operation is
performed) as well as for an `Image` (in which case it returns `img.data`).
For some image formats, Images.jl may interpret raw data with the `FixedPointNumbers`
package. The function `raw(img)` can be used to recover the buffer in its raw format
(e.g. `UInt8`). This is our first example of how to write generic algorithms.

If `img` is an `Image`, then `img[i,j]` looks up the value `img.data[i,j]`.
Assignment, `sub`, and `slice` work similarly. In other words, for indexing an
`Image` works just as if you were using plain arrays.

If you load your image data using Image's `imread`, note that the storage order
is not changed from the on-disk representation. Therefore, a 2D RGB image will
most likely be stored in color-horizontal-vertical order, meaning that a pixel
at `(x,y)` is accessed as `img[x,y]`. Note that this is quite different from
Matlab's default representation.

If you are indexing over an extended region and want to get back an `Image`,
rather than a value or an `Array`, then you will want to use `getindexim`,
`subim`, and `sliceim`. For the first two, the resulting image will share
everything but the `data` field with the original image; if you make
modifications in one, the other will also be affected. For `sliceim`, because it
can change the dimensionality some adjustments to `properties` are needed; in
this case a copy is made.

One of the properties (see below) that you can grant to images is
`spatialorder`, which provides a name for each spatial dimension in the image.
Using this feature, you can cut out regions or slices from images in the
following ways:

```julia
A = img["x", 200:400, "y", 500:700]
imgs = sliceim(img, "z", 14)      # cuts out the 14th frame in a stack
```

These routines "do the right thing" no matter what storage order is being used.

## Image properties and accessor functions

The `properties` dictionary can contain any information you want to store along
with your images. Typically, each property is also affiliated with an accessor
function of the same name.

Let's illustrate this with one of the default properties, `"colorspace"`.
The value of this property is a string, such as `"RGB"` or `"Gray"` or `"HSV"`.
You can extract the value of this field using a function:

```julia
cs = colorspace(img)
```

The reason to have a function, rather than just looking it up in the
`properties` dictionary, is that we can provide defaults. For example, images
represented as plain `Array`s don't have a `properties` dictionary; if we are to
write generic code, we don't want to have to wonder whether this information is
available. So for plain arrays, there are a number of defaults specified for the
output of the `colorspace` function, depending on the element type and size of
the array. Likewise, images stored as `Color` arrays have no need of a
`"colorspace"` property, because the colorspace is encoded in the type
parameters.

Here is a list of the properties supported in `core.jl`:

- `colorspace`: "RGB", "RGBA", "Gray", "Binary", "24bit", "Lab", "HSV", etc.  If
  your image is represented as a Color array, you cannot override that
  choice by specifying a `colorspace` property.  (Use `reinterpret` if you want
  to change the interpretation without changing the raw values.)
- `colordim`: the array dimension used to store color information, or 0 if there
  is no dimension corresponding to color
- `timedim`: the array dimension used for time (i.e., sequence), or 0 for single
  images
- `scalei`: a property that controls default contrast scaling upon display.
  This should be a
  [`MapInfo`](function_reference.html#mapinfo)
  value, to be used for setting the contrast upon display. In the absence of
  this property, the range 0 to 1 will be used.
- `pixelspacing`: the spacing between adjacent pixels along spatial dimensions
- `spacedirections`: more detailed information about the orientation of array
  axes relative to an external coordinate system (see the
  [function reference](function_reference.html)).
- `spatialorder`: a string naming each spatial dimension of the array, in the
  storage order of the data array.  Names can be arbitrary, but the choices "x"
  and "y" have special meaning (horizontal and vertical, respectively,
  irrespective of storage order).  If supplied, you must have one entry per
  spatial dimension.

If you specify their values in the `properties` dictionary, your values will be
used; if not, hopefully-reasonable defaults will be chosen.

Naturally, you can add whatever additional properties you want: you could add
the date/time at which the image was captured, the patient ID, etc. The main
point of having a properties dictionary, rather than a type with fixed fields,
is the flexibility of adding whatever metadata you find to be useful.

## Writing generic algorithms

Let's say you have an algorithm implemented for `Array`s, and you want to extend
it to work on `Image` types. Let's consider the example of a hypothetical
`imfilter`, written to perform kernel-based filtering in arbitrary dimensions.
Let's say your `imfilter` looks like this:

```julia
function imfilter{T,N}(A::Array{T,N}, kernel::Array{T,N}, options...)
```

The first step might be to simply provide a version for `AbstractImage` types:

```julia
function imfilter(img::AbstractImage{T,N}, kernel::Array{T,N}, options...) where {T,N}
    out = imfilter(data(img), kernel, options...)
    shareproperties(img, out)
end
```

Now let's say you additionally want to allow the user to filter color
images---where one dimension of the array is used to encode color---with a
filter of dimension `N-1` applied to each color channel separately. We can
implement this version simultaneously for both `Image` types and other array
types as follows:

```julia
function imfilter(img::AbstractArray{T,N}, kernel::Array{T,N1}, options...) where {T,N,N1}
    cd = colordim(img)
    if N1 != N - (cd != 0)
        error("kernel has the wrong dimensionality")
    end
    out = similar(img)
    for i = size(img, cd)
        imsl = img["color", i]
        outsl = slice(out, "color", i)
        copy!(outsl, imfilter(imsl, kernel, options...))
    end
    out
end
```

There are other ways to achieve a similar effect; if you examine the actual
implementation of `imfilter`, you'll see that the kernel is reshaped to be
commensurate with the data array.

These solutions work no matter which dimension is used to store color, a feat
that would be essentially impossible to achieve robustly in a generic algorithm
if we didn't exploit metadata. Note also that if the user supplies an `Array`,
s/he will get an `Array` back, and if using an `Image` will get an `Image` back
with properties inherited from `img`.

Naturally, you can find other examples of generic implementations throughout the
source code of `Images`.
