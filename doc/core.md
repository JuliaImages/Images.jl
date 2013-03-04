# Julia Images Guide

## The basic types

### Plain arrays

Images can be plain arrays, which are interpreted to be in "Matlab format": the first two dimensions are height (`h`) and width (`w`), a storage order here called "vertical-major". This order is inspired by the column/row order of matrices and the fact that both Julia and Matlab store matrices in column-major order.

If you're working with RGB color, you can use a third dimension, of size 3, or alternatively encode your images as either `Uint32` or `Int32` and work with 24-bit color. (If you need to use `Uint32` simply to store pixel intensities, you should not use plain arrays.)

It's worth noting that for images these conventions are sometimes inconvenient. For example, the `x` coordinate (horizontal) is second and the `y` coordinate (vertical) is first, i.e., `img[y,x]` to address a pixel that is displayed at a particular `x,y` position. Most image file formats, and display packages like Cairo, use "horizontal-major" storage of images, which therefore necessitates a transposition of the raw data before display. Moreover, usually the "fastest" dimension for these is the color channel. When Julia gets immutable types, it will become straightforward to represent color data as a type, and this will naturally address the latter issue.

These assumptions are "baked in" via a few simple utility functions in the file `core.jl`; if you really need to use plain arrays but want to work with different assumptions, you can (locally) change these defaults with just a few lines. Algorithms which have been properly written "generically" should continue to work.

### Image types

All image types should descend from `AbstractImage`, an abstract base type used to indicate that an array is to be interpreted as an image. If you're writing a custom image type, it is more likely that you'll want to derive from either `AbstractImageDirect` or `AbstractImageIndexed`. The former is for direct images (where intensity at a pixel is represented directly), the latter for indexed images (where intensity is looked up in a colormap table).

In practice, it is assumed that `AbstractImages` have at least two fields, called `data` and `properties`. These are the only two fields in the first concrete image type, called `Image`:

```
type Image{T,A<:AbstractArray} <: AbstractImageDirect{T}
    data::A
    properties::Dict
end
```

`data` stores the actual image data, and is an `AbstractArray`. This fact alone is the basis for a great deal of customizability: `data` might be a plain `Array` stored in memory, a `SubArray`, a memory-mapped array (which is still just an `Array`), a custom type that stores additional information about "missing data" (like bad pixels or dropped frames), or a custom type that seamlessly presents views of a large number of separate files. If you have a suitably-defined `AbstractArray` type, you can probably use `Image` without needing to create alternative `AbstractImageDirect` types.

`properties` is a dictionary, usually using `String` keys, that allows you to annotate images. More detail about these is below.

The only other concrete image type is for indexed images:

```
type ImageCmap{T,A<:AbstractArray,C<:AbstractArray} <: AbstractImageIndexed{T}
    data::A
    cmap::C
    properties::Dict
end
```
The `data` array here just encodes the index used to look up the color in the `cmap` field.

## Addressing image data

If `img` is an `Image`, then `img[i,j]` calls `ref`. This just provides direct access to the `data` array, and returns whatever `img.data[i,j]` would have returned. `assign`, `sub`, and `slice` work similarly. In other words, an `Image` works just as if you were using plain arrays.

TODO?: make `ref` for indexed images return the looked-up value? Not sure how one should handle `assign` in that case, however. Maybe the current behavior is better.

If you want to get back an `Image`, rather than a value or an `Array`, then you will want to use `refim`, `subim`, and `sliceim`. For the first two, the resulting image will share everything but the `data` field with the original image; if you make modifications in one, the other will also be affected. For `sliceim`, because it can change the dimensionality some adjustments to `properties` are needed; in this case a copy is made.

## Image properties and accessor functions

The `properties` dictionary can contain any information you want to store along with your images. Typically, each property is also affiliated with an accessor function of the same name.

Let's illustrate this with one of the default properties, `"colorspace"`. The value of this property is a string, such as `"RGB"` or `"Gray"` or `"HSV"`. You can extract the value of this field using a function:
```
cs = colorspace(img)
```
The reason to have a function, rather than just looking it up in the `properties` dictionary, is that we can provide defaults. For example, images represented as plain `Array`s don't have a `properties` dictionary; if we are to write generic code, we don't want to have to wonder whether this information is available. So for plain arrays, there are a number of defaults specified for the output of the `colorspace` function, depending on the element type and size of the array.

Here is a list of the properties supported in `core.jl`:

- colorspace: "RGB", "RGBA", "Gray", "Binary", "24bit", "Lab", "HSV", etc.
- colordim: the array dimension used to store color information, or 0 if there is no dimension corresponding to color
- timedim: the array dimension used for time (i.e., sequence), or 0 for single images
- limits: (minvalue,maxvalue) for this type of image (e.g., (0,255) for Uint8 images, even if pixels do not reach these values)
- pixelspacing: the spacing between adjacent pixels along spatial dimensions
- spatialorder: a string naming each spatial dimension, in the storage order of the data array. Names can be arbitrary, but the choices "x" and "y" have special meaning (horizontal and vertical, respectively, irrespective of storage order). If supplied, you must have one entry per spatial dimension.

If you specify their values in the `properties` dictionary, your values will be used; if not, hopefully-reasonable defaults will be chosen.

## Image I/O

Use the `imread` and `imwrite` functions.
