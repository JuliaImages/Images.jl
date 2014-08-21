# Getting started


## Image I/O

You likely have a number of images already at your disposal, and you can use these
to follow along.
You can install the  package,
in which case the "raw" command `imread` below should be replaced with `testimage`
supplied with the name of one of the images in TestImage.jl.
Finally, you can also retrieve a collection by running `readremote.jl` in the `test/` directory.
(This requires an internet connection.)
These will be deposited inside an `Images` directory inside your temporary directory
(e.g., `/tmp` on Linux systems).

Read an image from a file:
```
julia> img = imread("rose.png")
RGB Image with:
  data: 70x46 Array{RGB{Ufixed8},2}
  properties:
    spatialorder:  x y
```

As you can see, this is an RGB image. It is stored as a two-dimensional `Array` of `RGB{Ufixed8}`. 
To see what this pixel type is, we can do the following:
```
julia> img[1,1]
RGB{Ufixed8}(Ufixed8(0.188),Ufixed8(0.184),Ufixed8(0.176))
```
This extracts the first pixel, the one visually at the upper-left of the image. You can see that
an `RGB` (which comes from the [Color](https://github.com/JuliaLang/Color.jl) package) is a triple of values.
The `Ufixed8` type (which comes from the [FixedPointNumbers](https://github.com/JeffBezanson/FixedPointNumbers.jl) package)
represents fractional numbers, having values between 0 and 1 inclusive, using just 1 byte (8 bits).
If you've previously used other image processing libraries, you may be used to thinking of two basic
image types, floating point-valued and integer-valued. In those libraries, `1.0` commonly means "saturated"
for floating point-valued images, whereas for a `Uint8` image 255 means saturated.
`Images.jl` unifies these two types so that `1` always means saturated, making it easier to write
generic algorithms and visualization packages, while still allowing one to use efficient (or C-compatible)
raw representations.

You can see that this image has `properties`, in this case just the single property `"spatialorder"`.
`["x", "y"]` indicates that, after color, the image data are in "horizontal-major" order,
meaning that a pixel at spatial location `(x,y)` would be addressed as `img[x,y]`.
`["y", "x"]` would indicate vertical-major. Consequently, this image is 70 pixels wide and 46 pixels high.

Note that the image was loaded in "non-permuted" form, i.e., following the direct representation on disk.
If you prefer to work with plain arrays, you can convert it:
```
julia> imA = convert(Array, img);

julia> summary(imA)
"46x70 Array{RGB{Ufixed8},2}"
```
You can see that this permuted the dimensions into the vertical-major order.
If you prefer the color-last order typical of Matlab, you can use
```
julia> separate(img)
RGB Image with:
  data: 512x512x3 Array{Ufixed8,3}
  properties:
    colorspace: RGB
    colordim: 3
    spatialorder:  x y
```
You can see that two new properties were added: `"colordim"`, which specifies which dimension of the array
is used to encode color, and `"colorspace"`.


Writing works similarly (and as with all `Images` functions, it accepts both plain arrays and `Image` types):
```
julia> imwrite(img, "rose2.jpg")
```
The image file type is inferred from the extension.

## Image display

The most direct approach is the `display` command:
```
julia> ImageView.display(img)
```
For further information see the [ImageView](https://github.com/timholy/ImageView.jl) documentation.

## Working with images: some examples

Here's a short example that may help you get started:
```
using Images, ImageView, Color
img = imread("peppers.png")           # an RGB image (you can pick anything)
kern = ones(7,7)/49;                  # a boxcar smoothing filter
imgf = imfilter(img, kern)
ImageView.display(img)
ImageView.display(imgf)
clp = ClipMinMax(Uint8, 0.0, 255.0)
imgc = scale(clp,3*float64(img))      # generate an oversaturated image
ImageView.display(imgc)
imgp = permutedims(img,[2,3,1])       # so we don't have to call squeeze() next
O = Overlay((imgp["color",2],imgp["color",1]),(RGB(0,0,1),RGB(1,1,0)),((0,255),(0,255)))
ImageView.display(O,xy=["y","x"])
```

![raw](figures/peppers1.jpg)

![filtered](figures/peppers2.jpg)

![saturated](figures/peppers3.jpg)

![overlay](figures/peppers4.jpg)
