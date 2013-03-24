## Image I/O

If you need test images and are connected to the internet, you can retrieve a
collection by running `readremote.jl` in the `test/` directory. These will be
deposited inside an `Images` directory inside your temporary directory (e.g.,
`/tmp` on Linux systems).

Read an image from a file:
```
julia> img = imread("rose.png")
RGB Image with:
  data: 3x70x46 Uint8 Array
  properties: ["colordim"=>1,"spatialorder"=>["x", "y"],"colorspace"=>"RGB"]
```
Note that the image was loaded in "non-permuted" form, i.e., following the direct representation on disk. If you prefer to work with plain arrays, you can convert it:
```
julia> imA = convert(Array, img);

julia> summary(imA)
"46x70x3 Uint8 Array"
```
You can see that this permuted the dimensions into the array-canonical order.

Writing works similarly:
```
julia> imwrite(img, "rose2.jpg")
```
The image file type is inferred from the extension.

## Image display

The most direct approach is the `display` command:
```
julia> display(img)
WindowImage with buffer size 70x46
```
This shows the image in its own window, with no border, at native resolution. Alternatively, you can use an external viewer with the `imshow` command.

`display` even works with transparency:
```
julia> imgt = imread("autumn_leaves.png")
RGBA Image with:
  data: 4x140x105 Uint16 Array
  properties: ["colordim"=>1,"spatialorder"=>["x", "y"],"colorspace"=>"RGBA"]

julia> display(imgt)
WindowImage with buffer size 140x105
```
Overall, however, the `display` infrastructure is still quite crude---there is no resizing and no support for putting the image inside a container different from the full window.

## Working with images: some examples

Here's a short example that may help you get started:
```
using Images
img = float64(imread("peppers.png"))  # an RGB image (you can pick anything)
h = ones(7,7)/49;                     # a boxcar smoothing filter
imgf = imfilter(img, h)
display(img)
display(imgf)
clp = ClipMinMax(Uint8, 0.0, 255.0)
imgc = scale(clp,3img)                # generate an oversaturated image
display(imgc)
```
