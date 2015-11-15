# Images

## Exported

---

<a id="module__images.1" class="lexicon_definition"></a>
#### Images [¶](#module__images.1)
`Images` is a package for representing and processing images.

Constructors, conversions, and traits:

- Construction: `Image`, `ImageCmap`, `grayim`, `colorim`, `convert`, `copyproperties`, `shareproperties`
- Traits: `colordim`, `colorspace`, `coords_spatial`, `data`, `isdirect`, `isxfirst`, `isyfirst`, `pixelspacing`, `properties`, `sdims`, `spacedirections`, `spatialorder`, `storageorder`, `timedim`
- Size-related traits: `height`, `nchannels`, `ncolorelem`, `nimages`, `size_spatial`, `width`, `widthheight`
- Trait assertions: `assert_2d`, `assert_scalar_color`, `assert_timedim_last`, `assert_xfirst`, `assert_yfirst`
- Indexing operations: `getindexim`, `sliceim`, `subim`
- Conversions: `convert`, `raw`, `reinterpret`, `separate`

Contrast/coloration:

- `MapInfo`: `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`, `ScaleAutoMinMax`, etc.
- `imadjustintensity`, `sc`, `imstretch`, `imcomplement`


Algorithms:

- Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `sad`, `ssd`
- Resizing: `restrict`, `imresize` (not yet exported)
- Filtering: `imfilter`, `imfilter_fft`, `imfilter_gaussian`, `imfilter_LoG`, `imROF`, `ncc`, `padarray`
- Filtering kernels: `ando[345]`, `guassian2d`, `imaverage`, `imdog`, `imlaplacian`, `prewitt`, `sobel`
- Gradients: `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy`, `imgradients`
- Edge detection: `imedge`, `imgradients`, `thin_edges`, `magnitude`, `phase`, `magnitudephase`, `orientation`
- Morphological operations: `dilate`, `erode`, `closing`, `opening`
- Connected components: `label_components`

Test images and phantoms (see also TestImages.jl):

- `shepp_logan`


*source:*
[Images/src/Images.jl:296](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/Images.jl#L296)

---

<a id="method__overlayimage.1" class="lexicon_definition"></a>
#### OverlayImage(channels::Tuple{Vararg{AbstractArray{T, N}}},  colors::Tuple{Vararg{ColorTypes.Colorant{T, N}}},  arg) [¶](#method__overlayimage.1)
`OverlayImage` is identical to `Overlay`, except that it returns an Image.

*source:*
[Images/src/overlays.jl:48](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/overlays.jl#L48)

---

<a id="method__ando3.1" class="lexicon_definition"></a>
#### ando3() [¶](#method__ando3.1)
`kern1, kern2 = ando3()` returns optimal 3x3 filters for dimensions 1 and 2 of your image, as defined in
Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.

See also: `ando4`, `ando5`.


*source:*
[Images/src/edge.jl:36](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L36)

---

<a id="method__ando4.1" class="lexicon_definition"></a>
#### ando4() [¶](#method__ando4.1)
`kern1, kern2 = ando4()` returns optimal 4x4 filters for dimensions 1 and 2 of your image, as defined in
Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.

See also: `ando4_sep`, `ando3`, `ando5`.


*source:*
[Images/src/edge.jl:56](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L56)

---

<a id="method__ando5.1" class="lexicon_definition"></a>
#### ando5() [¶](#method__ando5.1)
`kern1, kern2 = ando5()` returns optimal 5x5 filters for dimensions 1 and 2 of your image, as defined in
Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.

See also: `ando5_sep`, `ando3`, `ando4`.


*source:*
[Images/src/edge.jl:86](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L86)

---

<a id="method__assert2d.1" class="lexicon_definition"></a>
#### assert2d(img::AbstractArray{T, N}) [¶](#method__assert2d.1)
`assert2d(img)` triggers an error if the image has more than two spatial
dimensions or has a time dimension.


*source:*
[Images/src/core.jl:1109](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1109)

---

<a id="method__assert_scalar_color.1" class="lexicon_definition"></a>
#### assert_scalar_color(img::AbstractArray{T, N}) [¶](#method__assert_scalar_color.1)
`assert_scalar_color(img)` triggers an error if the image uses an
array dimension to encode color.


*source:*
[Images/src/core.jl:1122](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1122)

---

<a id="method__assert_timedim_last.1" class="lexicon_definition"></a>
#### assert_timedim_last(img::AbstractArray{T, N}) [¶](#method__assert_timedim_last.1)
`assert_timedim_last(img)` triggers an error if the image has a time
dimension that is not the last dimension.


*source:*
[Images/src/core.jl:1133](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1133)

---

<a id="method__assert_xfirst.1" class="lexicon_definition"></a>
#### assert_xfirst(img::AbstractArray{T, N}) [¶](#method__assert_xfirst.1)
`assert_xfirst(img)` triggers an error if the first spatial dimension
is not `"x"`.


*source:*
[Images/src/core.jl:1165](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1165)

---

<a id="method__assert_yfirst.1" class="lexicon_definition"></a>
#### assert_yfirst(img) [¶](#method__assert_yfirst.1)
`assert_yfirst(img)` triggers an error if the first spatial dimension
is not `"y"`.


*source:*
[Images/src/core.jl:1149](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1149)

---

<a id="method__colordim.1" class="lexicon_definition"></a>
#### colordim{C<:ColorTypes.Colorant{T, N}}(img::AbstractArray{C<:ColorTypes.Colorant{T, N}, 1}) [¶](#method__colordim.1)
`dim = colordim(img)` returns the dimension used to encode color, or 0
if no dimension of the array is used for color. For example, an
`Array` of size `(m, n, 3)` would result in 3, whereas an `Array` of
`RGB` colorvalues would yield 0.

See also: `ncolorelem`, `timedim`.


*source:*
[Images/src/core.jl:909](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L909)

---

<a id="method__colorim.1" class="lexicon_definition"></a>
#### colorim(A::Images.AbstractImage{T, N}) [¶](#method__colorim.1)
```
img = colorim(A, [colorspace])
```
Creates a 2d color image from an AbstractArray `A`, auto-detecting which of the
first or last dimension encodes the color and choosing between "horizontal-" and
"vertical-major" accordingly. `colorspace` defaults to `"RGB"` but could also be
e.g. `"Lab"` or `"HSV"`.  If the array represents a 4-channel image, the
`colorspace` option is mandatory since there is no way to automatically
distinguish between `"ARGB"` and `"RGBA"`.  If both the first and last
dimensions happen to be of size 3 or 4, it is impossible to guess which one
represents color and thus an error is generated.  Thus, if your code needs to be
robust to arbitrary-sized images, you should use the `Image` constructor
directly.

See also: `grayim`, `Image`, `convert(Image{RGB}, A)`.


*source:*
[Images/src/core.jl:105](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L105)

---

<a id="method__colorspace.1" class="lexicon_definition"></a>
#### colorspace{C<:ColorTypes.Colorant{T, N}}(img::AbstractArray{C<:ColorTypes.Colorant{T, N}, 1}) [¶](#method__colorspace.1)
`cs = colorspace(img)` returns a string specifying the colorspace
representation of the image.


*source:*
[Images/src/core.jl:856](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L856)

---

<a id="method__coords_spatial.1" class="lexicon_definition"></a>
#### coords_spatial(img) [¶](#method__coords_spatial.1)
`c = coords_spatial(img)` returns a vector listing the spatial
dimensions of the image. For example, an `Array` of size `(m,n,3)`
would return `[1,2]`.

See also: `spatialorder`.


*source:*
[Images/src/core.jl:1064](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1064)

---

<a id="method__copyproperties.1" class="lexicon_definition"></a>
#### copyproperties(img::AbstractArray{T, N},  data::AbstractArray{T, N}) [¶](#method__copyproperties.1)
```
imgnew = copyproperties(img, data)
```
Creates a new image from the data array `data`, copying the properties from
Image `img`.


*source:*
[Images/src/core.jl:167](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L167)

---

<a id="method__data.1" class="lexicon_definition"></a>
#### data(img::AbstractArray{T, N}) [¶](#method__data.1)
```
A = data(img)
```
returns a reference `A` to the array data in `img`. It allows you to
use algorithms specialized for particular `AbstractArray` types on
`Image` types. This works for both `AbstractImage`s and
`AbstractArray`s (for the latter it just returns the input), so is a
"safe" component of any algorithm.

For algorithms written to accept arbitrary `AbstractArrays`, this
function is not needed.


*source:*
[Images/src/core.jl:713](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L713)

---

<a id="method__getindexim.1" class="lexicon_definition"></a>
#### getindexim(img::Images.AbstractImage{T, N},  I::Union{AbstractArray{T<:Real, N}, T<:Real}...) [¶](#method__getindexim.1)
```
imgnew = getindexim(img, i, j, k,...)
imgnew = getindexim(img, "x", 100:200, "y", 400:600)
```
return a new Image `imgnew`, copying (and where necessary modifying)
the properties of `img`.  This is in contrast with `img[i, j, k...]`,
which returns an `Array`.


*source:*
[Images/src/core.jl:488](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L488)

---

<a id="method__grayim.1" class="lexicon_definition"></a>
#### grayim(A::Images.AbstractImage{T, N}) [¶](#method__grayim.1)
```
img = grayim(A)
```
creates a 2d or 3d _spatial_ grayscale Image from an AbstractArray
`A`, assumed to be in "horizontal-major" order (and without permuting
any dimensions). If you are working with 3d grayscale images, usage of
this function is strongly recommended. This can fix errors like any of
the following:

```
ERROR: Wrong number of spatial dimensions for plain Array, use an AbstractImage type
ERROR: Cannot infer colorspace of Array, use an AbstractImage type
ERROR: Cannot infer pixelspacing of Array, use an AbstractImage type
```

The main reason for such errors---and the reason that `grayim` is
recommended---is the Matlab-derived convention that a `m x n x 3` array is to be
interpreted as RGB.  One might then say that an `m x n x k` array, for `k`
different from 3, could be interpreted as grayscale. However, this would lead to
difficult-to-track-down surprises on the day where `k` happened to be 3 for your
grayscale image.

See also: `colorim`, `Image`, `convert(Image, A)`.


*source:*
[Images/src/core.jl:80](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L80)

---

<a id="method__height.1" class="lexicon_definition"></a>
#### height(img::AbstractArray{T, N}) [¶](#method__height.1)
`h = height(img)` returns the vertical size of the image, regardless
of storage order. By default horizontal corresponds to dimension
`"y"`, but see `spatialpermutation` for other options.


*source:*
[Images/src/core.jl:1200](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1200)

---

<a id="method__imedge.1" class="lexicon_definition"></a>
#### imedge(img::AbstractArray{T, N}) [¶](#method__imedge.1)
```
grad_x, grad_y, mag, orient = imedge(img, [method], [border])
```

Edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple `(grad_x, grad_y, mag, orient)`, which are the horizontal
gradient, vertical gradient, and the magnitude and orientation of the strongest
edge, respectively.


*source:*
[Images/src/edge.jl:248](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L248)

---

<a id="method__imedge.2" class="lexicon_definition"></a>
#### imedge(img::AbstractArray{T, N},  method::AbstractString) [¶](#method__imedge.2)
```
grad_x, grad_y, mag, orient = imedge(img, [method], [border])
```

Edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple `(grad_x, grad_y, mag, orient)`, which are the horizontal
gradient, vertical gradient, and the magnitude and orientation of the strongest
edge, respectively.


*source:*
[Images/src/edge.jl:248](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L248)

---

<a id="method__imedge.3" class="lexicon_definition"></a>
#### imedge(img::AbstractArray{T, N},  method::AbstractString,  border::AbstractString) [¶](#method__imedge.3)
```
grad_x, grad_y, mag, orient = imedge(img, [method], [border])
```

Edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple `(grad_x, grad_y, mag, orient)`, which are the horizontal
gradient, vertical gradient, and the magnitude and orientation of the strongest
edge, respectively.


*source:*
[Images/src/edge.jl:248](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L248)

---

<a id="method__imgradients.1" class="lexicon_definition"></a>
#### imgradients(img::AbstractArray{T, N}) [¶](#method__imgradients.1)
```
grad_x, grad_y = imgradients(img, [method], [border])
```

performs edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple containing `x` (horizontal) and `y` (vertical) gradient images
of the same size as `img`, calculated using the requested method and border.


*source:*
[Images/src/edge.jl:126](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L126)

---

<a id="method__imgradients.2" class="lexicon_definition"></a>
#### imgradients(img::AbstractArray{T, N},  method::AbstractString) [¶](#method__imgradients.2)
```
grad_x, grad_y = imgradients(img, [method], [border])
```

performs edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple containing `x` (horizontal) and `y` (vertical) gradient images
of the same size as `img`, calculated using the requested method and border.


*source:*
[Images/src/edge.jl:126](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L126)

---

<a id="method__imgradients.3" class="lexicon_definition"></a>
#### imgradients(img::AbstractArray{T, N},  method::AbstractString,  border::AbstractString) [¶](#method__imgradients.3)
```
grad_x, grad_y = imgradients(img, [method], [border])
```

performs edge-detection filtering. `method` is one of `"sobel"`, `"prewitt"`, `"ando3"`,
`"ando4"`, `"ando4_sep"`, `"ando5"`, or `"ando5_sep"`, defaulting to `"ando3"`
(see the functions of the same name for more information).  `border` is any of
the boundary conditions specified in `padarray`.

Returns a tuple containing `x` (horizontal) and `y` (vertical) gradient images
of the same size as `img`, calculated using the requested method and border.


*source:*
[Images/src/edge.jl:126](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L126)

---

<a id="method__isdirect.1" class="lexicon_definition"></a>
#### isdirect(img::AbstractArray{T, N}) [¶](#method__isdirect.1)
`isdirect(img)` returns true if `img` encodes its values directly,
rather than via an indexed colormap.


*source:*
[Images/src/core.jl:848](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L848)

---

<a id="method__isxfirst.1" class="lexicon_definition"></a>
#### isxfirst(img::AbstractArray{T, N}) [¶](#method__isxfirst.1)
`tf = isxfirst(img)` tests whether the first spatial dimension is `"x"`.

See also: `isyfirst`, `assert_xfirst`.


*source:*
[Images/src/core.jl:1160](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1160)

---

<a id="method__isyfirst.1" class="lexicon_definition"></a>
#### isyfirst(img::AbstractArray{T, N}) [¶](#method__isyfirst.1)
`tf = isyfirst(img)` tests whether the first spatial dimension is `"y"`.

See also: `isxfirst`, `assert_yfirst`.


*source:*
[Images/src/core.jl:1144](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1144)

---

<a id="method__label_components.1" class="lexicon_definition"></a>
#### label_components(A) [¶](#method__label_components.1)
```
label = label_components(tf, [connectivity])
label = label_components(tf, [region])
```

Find the connected components in a binary array `tf`. There are two forms that
`connectivity` can take:

- It can be a boolean array of the same dimensionality as `tf`, of size 1 or 3
along each dimension. Each entry in the array determines whether a given
neighbor is used for connectivity analyses. For example, `connectivity = trues(3,3)`
would use 8-connectivity and test all pixels that touch the current one, even
the corners.

- You can provide a list indicating which dimensions are used to
determine connectivity. For example, `region = [1,3]` would not test
neighbors along dimension 2 for connectivity. This corresponds to just
the nearest neighbors, i.e., 4-connectivity in 2d and 6-connectivity
in 3d.

The default is `region = 1:ndims(A)`.

The output `label` is an integer array, where 0 is used for background
pixels, and each connected region gets a different integer index.


*source:*
[Images/src/connected.jl:29](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/connected.jl#L29)

---

<a id="method__label_components.2" class="lexicon_definition"></a>
#### label_components(A,  connectivity) [¶](#method__label_components.2)
```
label = label_components(tf, [connectivity])
label = label_components(tf, [region])
```

Find the connected components in a binary array `tf`. There are two forms that
`connectivity` can take:

- It can be a boolean array of the same dimensionality as `tf`, of size 1 or 3
along each dimension. Each entry in the array determines whether a given
neighbor is used for connectivity analyses. For example, `connectivity = trues(3,3)`
would use 8-connectivity and test all pixels that touch the current one, even
the corners.

- You can provide a list indicating which dimensions are used to
determine connectivity. For example, `region = [1,3]` would not test
neighbors along dimension 2 for connectivity. This corresponds to just
the nearest neighbors, i.e., 4-connectivity in 2d and 6-connectivity
in 3d.

The default is `region = 1:ndims(A)`.

The output `label` is an integer array, where 0 is used for background
pixels, and each connected region gets a different integer index.


*source:*
[Images/src/connected.jl:29](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/connected.jl#L29)

---

<a id="method__label_components.3" class="lexicon_definition"></a>
#### label_components(A,  connectivity,  bkg) [¶](#method__label_components.3)
```
label = label_components(tf, [connectivity])
label = label_components(tf, [region])
```

Find the connected components in a binary array `tf`. There are two forms that
`connectivity` can take:

- It can be a boolean array of the same dimensionality as `tf`, of size 1 or 3
along each dimension. Each entry in the array determines whether a given
neighbor is used for connectivity analyses. For example, `connectivity = trues(3,3)`
would use 8-connectivity and test all pixels that touch the current one, even
the corners.

- You can provide a list indicating which dimensions are used to
determine connectivity. For example, `region = [1,3]` would not test
neighbors along dimension 2 for connectivity. This corresponds to just
the nearest neighbors, i.e., 4-connectivity in 2d and 6-connectivity
in 3d.

The default is `region = 1:ndims(A)`.

The output `label` is an integer array, where 0 is used for background
pixels, and each connected region gets a different integer index.


*source:*
[Images/src/connected.jl:29](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/connected.jl#L29)

---

<a id="method__magnitude.1" class="lexicon_definition"></a>
#### magnitude(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N}) [¶](#method__magnitude.1)
```
m = magnitude(grad_x, grad_y)
```

Calculates the magnitude of the gradient images given by `grad_x` and `grad_y`.
Equivalent to ``sqrt(grad_x.^2 + grad_y.^2)``.

Returns a magnitude image the same size as `grad_x` and `grad_y`.


*source:*
[Images/src/edge.jl:159](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L159)

---

<a id="method__magnitude_phase.1" class="lexicon_definition"></a>
#### magnitude_phase(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N}) [¶](#method__magnitude_phase.1)
`m, p = magnitude_phase(grad_x, grad_y)`

Convenience function for calculating the magnitude and phase of the gradient
images given in `grad_x` and `grad_y`.  Returns a tuple containing the magnitude
and phase images.  See `magnitude` and `phase` for details.


*source:*
[Images/src/edge.jl:225](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L225)

---

<a id="method__mapinfo.1" class="lexicon_definition"></a>
#### mapinfo{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}}(::Type{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}},  img::AbstractArray{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}, N}) [¶](#method__mapinfo.1)
`mapi = mapinf(T, img)` returns a `MapInfo` object that is deemed
appropriate for converting pixels of `img` to be of type `T`. `T` can
either be a specific type (e.g., `RGB24`), or you can specify an
abstract type like `Clamp` and it will return one of the `Clamp`
family of `MapInfo` objects.

You can define your own rules for `mapinfo`.  For example, the
`ImageMagick` package defines methods for how pixels values should be
converted before saving images to disk.


*source:*
[Images/src/map.jl:556](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L556)

---

<a id="method__ncolorelem.1" class="lexicon_definition"></a>
#### ncolorelem(img) [¶](#method__ncolorelem.1)
`n = ncolorelem(img)` returns the number of color elements/voxel, or 1 if color is not a separate dimension of the array.


*source:*
[Images/src/core.jl:1052](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1052)

---

<a id="method__nimages.1" class="lexicon_definition"></a>
#### nimages(img) [¶](#method__nimages.1)
`n = nimages(img)` returns the number of time-points in the image
array. This is safer than `size(img, "t")` because it also works for
plain `AbstractArray` types.


*source:*
[Images/src/core.jl:1040](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1040)

---

<a id="method__orientation.1" class="lexicon_definition"></a>
#### orientation{T}(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N}) [¶](#method__orientation.1)
```
orient = orientation(grad_x, grad_y)
```

Calculates the orientation angle of the strongest edge from gradient images
given by `grad_x` and `grad_y`.  Equivalent to ``atan2(grad_x, grad_y)``.  When
both `grad_x[i]` and `grad_y[i]` are zero, the corresponding angle is set to
zero.

Returns a phase image the same size as `grad_x` and `grad_y`, with values in
[-pi,pi].


*source:*
[Images/src/edge.jl:203](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L203)

---

<a id="method__phase.1" class="lexicon_definition"></a>
#### phase{T}(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N}) [¶](#method__phase.1)
```
p = phase(grad_x, grad_y)
```

Calculates the rotation angle of the gradient images given by `grad_x` and
`grad_y`. Equivalent to ``atan2(-grad_y, grad_x)``.  When both ``grad_x[i]`` and
``grad_y[i]`` are zero, the corresponding angle is set to zero.

Returns a phase image the same size as `grad_x` and `grad_y`, with values in [-pi,pi].


*source:*
[Images/src/edge.jl:173](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L173)

---

<a id="method__pixelspacing.1" class="lexicon_definition"></a>
#### pixelspacing{T}(img::AbstractArray{T, 3}) [¶](#method__pixelspacing.1)
```
ps = pixelspacing(img)
```

Returns a vector `ps` containing the spacing between adjacent pixels along each
dimension. If this property is not available, it will be computed from
`"spacedirections"` if present; otherwise it defaults to `ones(sdims(img))`. If
desired, you can set this property in terms of physical
[units](https://github.com/Keno/SIUnits.jl).

See also: `spacedirections`.


*source:*
[Images/src/core.jl:948](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L948)

---

<a id="method__prewitt.1" class="lexicon_definition"></a>
#### prewitt() [¶](#method__prewitt.1)
`kern1, kern2 = prewitt()` returns Prewitt filters for dimensions 1 and 2 of your image

*source:*
[Images/src/edge.jl:14](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L14)

---

<a id="method__properties.1" class="lexicon_definition"></a>
#### properties(A::AbstractArray{T, N}) [¶](#method__properties.1)
`prop = properties(img)` returns the properties-dictionary for an
`AbstractImage`, or creates one if `img` is an `AbstractArray`.


*source:*
[Images/src/core.jl:789](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L789)

---

<a id="method__raw.1" class="lexicon_definition"></a>
#### raw(img::AbstractArray{T, N}) [¶](#method__raw.1)
```
imgraw = raw(img)
```
returns a reference to the array data in raw (machine-native) storage
format. This is particularly useful when Images.jl wraps image data in
a `FixedPointNumbers` type, and raw data access is desired. For
example

```
img = load("someimage.tif")
typeof( data(img) )  # return Array{UFixed{UInt8,8},2}
typeof( raw(img) )   # returns Array{UInt8,2}
```


*source:*
[Images/src/core.jl:310](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L310)

---

<a id="method__sc.1" class="lexicon_definition"></a>
#### sc(img::AbstractArray{T, N}) [¶](#method__sc.1)
```
imgsc = sc(img)
imgsc = sc(img, min, max)
```

Applies default or specified `ScaleMinMax` mapping to the image.


*source:*
[Images/src/map.jl:673](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L673)

---

<a id="method__sdims.1" class="lexicon_definition"></a>
#### sdims(img) [¶](#method__sdims.1)
`n = sdims(img)` is similar to `ndims`, but it returns just the number of *spatial* dimensions in
the image array (excluding color and time).


*source:*
[Images/src/core.jl:1032](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1032)

---

<a id="method__separate.1" class="lexicon_definition"></a>
#### separate{CV<:ColorTypes.Colorant{T, N}}(img::Images.AbstractImage{CV<:ColorTypes.Colorant{T, N}, N}) [¶](#method__separate.1)
`imgs = separate(img)` separates the color channels of `img`, for
example returning an `m-by-n-by-3` array from an `m-by-n` array of
`RGB`.


*source:*
[Images/src/core.jl:389](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L389)

---

<a id="method__shareproperties.1" class="lexicon_definition"></a>
#### shareproperties(img::AbstractArray{T, N},  data::AbstractArray{T, N}) [¶](#method__shareproperties.1)
```
imgnew = shareproperties(img, data)
```
Creates a new image from the data array `data`, *sharing* the properties of
Image `img`. Any modifications made to the properties of one will affect the
other.


*source:*
[Images/src/core.jl:183](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L183)

---

<a id="method__size_spatial.1" class="lexicon_definition"></a>
#### size_spatial(img) [¶](#method__size_spatial.1)
```
ssz = size_spatial(img)
```

Returns a tuple listing the sizes of the spatial dimensions of the image. For
example, an `Array` of size `(m,n,3)` would return `(m,n)`.

See also: `nimages`, `width`, `height`, `widthheight`.


*source:*
[Images/src/core.jl:1097](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1097)

---

<a id="method__sliceim.1" class="lexicon_definition"></a>
#### sliceim(img::Images.AbstractImage{T, N},  I::Union{Colon, Int64, Range{Int64}}...) [¶](#method__sliceim.1)
```
imgs = sliceim(img, i, j, k, ...)
imgs = sliceim(img, "x", 100:200, "y", 400:600)
```
returns an `Image` with `SubArray` data, with indexing semantics similar to `slice`.


*source:*
[Images/src/core.jl:533](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L533)

---

<a id="method__sobel.1" class="lexicon_definition"></a>
#### sobel() [¶](#method__sobel.1)
`kern1, kern2 = sobel()` returns Sobel filters for dimensions 1 and 2 of your image

*source:*
[Images/src/edge.jl:6](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L6)

---

<a id="method__spacedirections.1" class="lexicon_definition"></a>
#### spacedirections(img::AbstractArray{T, N}) [¶](#method__spacedirections.1)
```
sd = spacedirections(img)
```

Returns a vector-of-vectors `sd`, each `sd[i]`indicating the displacement between adjacent
pixels along spatial axis `i` of the image array, relative to some external
coordinate system ("physical coordinates").  For example, you could indicate
that a photograph was taken with the camera tilted 30-degree relative to
vertical using

```
img["spacedirections"] = [[0.866025,-0.5],[0.5,0.866025]]
```

If not specified, it will be computed from `pixelspacing(img)`, placing the
spacing along the "diagonal".  If desired, you can set this property in terms of
physical [units](https://github.com/loladiro/SIUnits.jl).

See also: `pixelspacing`.


*source:*
[Images/src/core.jl:981](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L981)

---

<a id="method__spatialorder.1" class="lexicon_definition"></a>
#### spatialorder(::Type{Array{T, 2}}) [¶](#method__spatialorder.1)
```
order = spatialorder(img)
order = spatialorder(ImageType)
```

Returns the storage order of the _spatial_ coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`. See
`storageorder`, `timedim`, and `colordim` for related properties.


*source:*
[Images/src/core.jl:841](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L841)

---

<a id="method__spatialorder.2" class="lexicon_definition"></a>
#### spatialorder(img::Images.AbstractImage{T, N}) [¶](#method__spatialorder.2)
```
so = spatialorder(img)
so = spatialorder(ImageType)
```

Returns the storage order of the *spatial* coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`.

See also: `storageorder`, `coords_spatial`, `timedim`, and `colordim`.


*source:*
[Images/src/core.jl:1000](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1000)

---

<a id="method__spatialpermutation.1" class="lexicon_definition"></a>
#### spatialpermutation(to,  img::AbstractArray{T, N}) [¶](#method__spatialpermutation.1)
```
p = spatialpermutation(to, img)
```

Calculates the *spatial* permutation needed to convert the spatial dimensions to
a given order. This is probably easiest to understand by examples: for an
`Array` `A` of size `(m,n,3)`, `spatialorder(A)` would yield `["y", "x"]`, so
`spatialpermutation(["y", "x"], A) = [1,2]` and `spatialpermutation(["x", "y"],
A) = [2,1]`.  For an image type, here's a demonstration:

```
julia> Aimg = convert(Image, A)
RGB Image with:
  data: 4x5x3 Array{Float64,3}
  properties:
    colordim: 3
    spatialorder:  y x
    colorspace: RGB

julia> Ap = permutedims(Aimg, [3, 1, 2])
RGB Image with:
  data: 3x4x5 Array{Float64,3}
  properties:
    colordim: 1
    spatialorder:  y x
    colorspace: RGB

julia> spatialpermutation(["x","y"], Ap)
2-element Array{Int64,1}:
 2
 1
```


*source:*
[Images/src/core.jl:1236](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1236)

---

<a id="method__spatialproperties.1" class="lexicon_definition"></a>
#### spatialproperties(img::Images.AbstractImage{T, N}) [¶](#method__spatialproperties.1)
```
sp = spatialproperties(img)
```

Returns all properties whose values are of the form of an array or tuple, with
one entry per spatial dimension. If you have a custom type with additional
spatial properties, you can set `img["spatialproperties"] = ["property1",
"property2", ...]`. An advantage is that functions that change spatial
dimensions, like `permutedims` and `slice`, will also adjust the properties. The
default is `["spatialorder", "pixelspacing"]`; however, if you override the
setting then these are not included automatically (you'll want to do so
manually, if applicable).


*source:*
[Images/src/core.jl:1324](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1324)

---

<a id="method__storageorder.1" class="lexicon_definition"></a>
#### storageorder(img::AbstractArray{T, N}) [¶](#method__storageorder.1)
```
so = storageorder(img)
```

Returns the complete storage order of the image array, including `"t"` for time
and `"color"` for color.

See also: `spatialorder`, `colordim`, `timedim`.


*source:*
[Images/src/core.jl:1013](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1013)

---

<a id="method__subim.1" class="lexicon_definition"></a>
#### subim(img::Images.AbstractImage{T, N},  I::Union{Colon, Int64, Range{Int64}}...) [¶](#method__subim.1)
```
imgs = subim(img, i, j, k, ...)
imgs = subim(img, "x", 100:200, "y", 400:600)
```
returns an `Image` with `SubArray` data, with indexing semantics similar to `sub`.


*source:*
[Images/src/core.jl:521](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L521)

---

<a id="method__thin_edges.1" class="lexicon_definition"></a>
#### thin_edges{T}(img::AbstractArray{T, 2},  gradientangles::AbstractArray{T, N}) [¶](#method__thin_edges.1)
```
thinned = thin_edges(img, gradientangle, [border])
thinned, subpix = thin_edges_subpix(img, gradientangle, [border])
thinned, subpix = thin_edges_nonmaxsup(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
thinned, subpix = thin_edges_nonmaxsup_subpix(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
```

Edge thinning for 2D edge images.  Currently the only algorithm available is
non-maximal suppression, which takes an edge image and its gradient angle, and
checks each edge point for local maximality in the direction of the gradient.
The returned image is non-zero only at maximal edge locations.

`border` is any of the boundary conditions specified in `padarray`.

In addition to the maximal edge image, the `_subpix` versions of these functions
also return an estimate of the subpixel location of each local maxima, as a 2D
array or image of `Graphics.Point` objects.  Additionally, each local maxima is
adjusted to the estimated value at the subpixel location.

Currently, the `_nonmaxsup` functions are identical to the first two function
calls, except that they also accept additional keyword arguments.  `radius`
indicates the step size to use when searching in the direction of the gradient;
values between 1.2 and 1.5 are suggested (default 1.35).  `theta` indicates the
step size to use when discretizing angles in the `gradientangle` image, in
radians (default: 1 degree in radians = pi/180).

Example:

```
g = rgb2gray(rgb_image)
gx, gy = imgradients(g)
mag, grad_angle = magnitude_phase(gx,gy)
mag[mag .< 0.5] = 0.0  # Threshold magnitude image
thinned, subpix =  thin_edges_subpix(mag, gradient)
```


*source:*
[Images/src/edge.jl:293](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L293)

---

<a id="method__thin_edges.2" class="lexicon_definition"></a>
#### thin_edges{T}(img::AbstractArray{T, 2},  gradientangles::AbstractArray{T, N},  border::AbstractString) [¶](#method__thin_edges.2)
```
thinned = thin_edges(img, gradientangle, [border])
thinned, subpix = thin_edges_subpix(img, gradientangle, [border])
thinned, subpix = thin_edges_nonmaxsup(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
thinned, subpix = thin_edges_nonmaxsup_subpix(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
```

Edge thinning for 2D edge images.  Currently the only algorithm available is
non-maximal suppression, which takes an edge image and its gradient angle, and
checks each edge point for local maximality in the direction of the gradient.
The returned image is non-zero only at maximal edge locations.

`border` is any of the boundary conditions specified in `padarray`.

In addition to the maximal edge image, the `_subpix` versions of these functions
also return an estimate of the subpixel location of each local maxima, as a 2D
array or image of `Graphics.Point` objects.  Additionally, each local maxima is
adjusted to the estimated value at the subpixel location.

Currently, the `_nonmaxsup` functions are identical to the first two function
calls, except that they also accept additional keyword arguments.  `radius`
indicates the step size to use when searching in the direction of the gradient;
values between 1.2 and 1.5 are suggested (default 1.35).  `theta` indicates the
step size to use when discretizing angles in the `gradientangle` image, in
radians (default: 1 degree in radians = pi/180).

Example:

```
g = rgb2gray(rgb_image)
gx, gy = imgradients(g)
mag, grad_angle = magnitude_phase(gx,gy)
mag[mag .< 0.5] = 0.0  # Threshold magnitude image
thinned, subpix =  thin_edges_subpix(mag, gradient)
```


*source:*
[Images/src/edge.jl:293](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L293)

---

<a id="method__timedim.1" class="lexicon_definition"></a>
#### timedim(img) [¶](#method__timedim.1)
`dim = timedim(img)` returns the dimension used to represent time, or
0 if this is a single image.

See also: `nimages`, `colordim`.


*source:*
[Images/src/core.jl:926](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L926)

---

<a id="method__width.1" class="lexicon_definition"></a>
#### width(img::AbstractArray{T, N}) [¶](#method__width.1)
`w = width(img)` returns the horizontal size of the image, regardless
of storage order. By default horizontal corresponds to dimension
`"x"`, but see `spatialpermutation` for other options.


*source:*
[Images/src/core.jl:1194](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1194)

---

<a id="method__widthheight.1" class="lexicon_definition"></a>
#### widthheight(img::AbstractArray{T, N},  p) [¶](#method__widthheight.1)
`w, h = widthheight(img)` returns the width and height of an image, regardless of storage order.

See also: `width`, `height`.


*source:*
[Images/src/core.jl:1182](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L1182)

---

<a id="type__abstractimagedirect.1" class="lexicon_definition"></a>
#### Images.AbstractImageDirect{T, N} [¶](#type__abstractimagedirect.1)
`AbstractImageDirect` is the supertype of all images where pixel
values are stored directly in an `AbstractArray`.  See also
`AbstractImageIndexed`.


*source:*
[Images/src/core.jl:11](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L11)

---

<a id="type__abstractimageindexed.1" class="lexicon_definition"></a>
#### Images.AbstractImageIndexed{T, N} [¶](#type__abstractimageindexed.1)
`AbstractImageIndexed` is the supertype of all "colormap" images,
where pixel values are accessed from a lookup table.  See also
`AbstractImageDirect`.


*source:*
[Images/src/core.jl:17](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L17)

---

<a id="type__bitshift.1" class="lexicon_definition"></a>
#### Images.BitShift{T, N} [¶](#type__bitshift.1)
`BitShift{T,N}` performs a "saturating rightward bit-shift" operation.
It is particularly useful in converting high bit-depth images to 8-bit
images for the purpose of display.  For example,

```
map(BitShift(UFixed8, 8), 0xa2d5uf16) === 0xa2uf8
```

converts a `UFixed16` to the corresponding `UFixed8` by discarding the
least significant byte.  However,

```
map(BitShift(UFixed8, 7), 0xa2d5uf16) == 0xffuf8
```

because `0xa2d5>>7 == 0x0145 > typemax(UInt8)`.

When applicable, the main advantage of using `BitShift` rather than
`MapNone` or `ScaleMinMax` is speed.


*source:*
[Images/src/map.jl:98](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L98)

---

<a id="type__clamp01nan.1" class="lexicon_definition"></a>
#### Images.Clamp01NaN{T} [¶](#type__clamp01nan.1)
`Clamp01NaN(T)` constructs a `MapInfo` object that clamps grayscale or
color pixels to the interval `[0,1]`, sending `NaN` pixels to zero.


*source:*
[Images/src/map.jl:330](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L330)

---

<a id="type__clampmax.1" class="lexicon_definition"></a>
#### Images.ClampMax{T, From} [¶](#type__clampmax.1)
`ClampMax(T, maxvalue)` is a `MapInfo` object that clamps pixel values
to be less than or equal to `maxvalue` before converting to type `T`.

See also: `ClampMin`, `ClampMinMax`.


*source:*
[Images/src/map.jl:136](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L136)

---

<a id="type__clampmin.1" class="lexicon_definition"></a>
#### Images.ClampMin{T, From} [¶](#type__clampmin.1)
`ClampMin(T, minvalue)` is a `MapInfo` object that clamps pixel values
to be greater than or equal to `minvalue` before converting to type `T`.

See also: `ClampMax`, `ClampMinMax`.


*source:*
[Images/src/map.jl:125](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L125)

---

<a id="type__clamp.1" class="lexicon_definition"></a>
#### Images.Clamp{T} [¶](#type__clamp.1)
`Clamp(C)` is a `MapInfo` object that clamps color values to be within
gamut.  For example,

```
map(Clamp(RGB{U8}), RGB(1.2, -0.4, 0.6)) === RGB{U8}(1, 0, 0.6)
```


*source:*
[Images/src/map.jl:162](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L162)

---

<a id="type__imagecmap.1" class="lexicon_definition"></a>
#### Images.ImageCmap{T<:ColorTypes.Colorant{T, N}, N, A<:AbstractArray{T, N}} [¶](#type__imagecmap.1)
```
ImageCmap(data, cmap, [properties])
```
creates an indexed (colormap) image.


*source:*
[Images/src/core.jl:46](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L46)

---

<a id="type__image.1" class="lexicon_definition"></a>
#### Images.Image{T, N, A<:AbstractArray{T, N}} [¶](#type__image.1)
```
Image(data, [properties])
Image(data, prop1=val1, prop2=val2, ...)
```
creates a new "direct" image, one in which the values in `data`
correspond to the pixel values. In contrast with `convert`, `grayim`
and `colorim`, this does not permute the data array or attempt to
guess any of the `properties`. If `data` encodes color information
along one of the dimensions of the array (as opposed to using a
`Color` array, from the `Colors.jl` package), be sure to specify the
`"colordim"` and `"colorspace"` in `properties`.


*source:*
[Images/src/core.jl:32](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L32)

---

<a id="type__mapinfo.1" class="lexicon_definition"></a>
#### Images.MapInfo{T} [¶](#type__mapinfo.1)
`MapInfo{T}` is an abstract type that encompasses objects designed to
perform intensity or color transformations on pixels.  For example,
before displaying an image in a window, you might need to adjust the
contrast settings; `MapInfo` objects provide a means to describe these
transformations without calculating them immediately.  This delayed
execution can be useful in many contexts.  For example, if you want to
display a movie, it would be quite wasteful to have to first transform
the entire movie; instead, `MapInfo` objects allow one to specify a
transformation to be performed on-the-fly as particular frames are
displayed.

You can create your own custom `MapInfo` objects. For example, given a
grayscale image, you could color "saturated" pixels red using

```jl
immutable ColorSaturated{C<:AbstractRGB} <: MapInfo{C}
end

Base.map{C}(::ColorSaturated{C}, val::Union{Number,Gray}) = ifelse(val == 1, C(1,0,0), C(val,val,val))

imgc = map(ColorSaturated{RGB{U8}}(), img)
```

For pre-defined types see `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`,
`ScaleAutoMinMax`, and `ScaleSigned`.


*source:*
[Images/src/map.jl:50](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L50)

---

<a id="type__mapnone.1" class="lexicon_definition"></a>
#### Images.MapNone{T} [¶](#type__mapnone.1)
`MapNone(T)` is a `MapInfo` object that converts `x` to have type `T`.

*source:*
[Images/src/map.jl:56](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L56)

---

<a id="type__overlay.1" class="lexicon_definition"></a>
#### Images.Overlay{T, N, NC, AT<:Tuple{Vararg{AbstractArray{T, N}}}, MITypes<:Tuple{Vararg{Images.MapInfo{T}}}} [¶](#type__overlay.1)
```
A = Overlay(channels, colors, clim)
A = Overlay(channels, colors, mapi)
```

Create an `Overlay` array from grayscale channels.  `channels = (channel1,
channel2, ...)`, `colors` is a vector or tuple of `Color`s, and `clim` is a
vector or tuple of min/max values, e.g., `clim = ((min1,max1),(min2,max2),...)`.
Alternatively, you can supply a list of `MapInfo` objects.

See also: `OverlayImage`.


*source:*
[Images/src/overlays.jl:15](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/overlays.jl#L15)

---

<a id="type__scaleautominmax.1" class="lexicon_definition"></a>
#### Images.ScaleAutoMinMax{T} [¶](#type__scaleautominmax.1)
`ScaleAutoMinMax(T)` constructs a `MapInfo` object that causes images
to be dynamically scaled to their specific min/max values, using the
same algorithm for `ScaleMinMax`. When displaying a movie, the min/max
will be recalculated for each frame, so this can result in
inconsistent contrast scaling.


*source:*
[Images/src/map.jl:308](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L308)

---

<a id="type__scaleminmaxnan.1" class="lexicon_definition"></a>
#### Images.ScaleMinMaxNaN{To, From, S} [¶](#type__scaleminmaxnan.1)
`ScaleMinMaxNaN(smm)` constructs a `MapInfo` object from a
`ScaleMinMax` object `smm`, with the additional property that `NaN`
values map to zero.

See also: `ScaleMinMax`.


*source:*
[Images/src/map.jl:322](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L322)

---

<a id="type__scaleminmax.1" class="lexicon_definition"></a>
#### Images.ScaleMinMax{To, From, S<:AbstractFloat} [¶](#type__scaleminmax.1)
`ScaleMinMax(T, min, max, [scalefactor])` is a `MapInfo` object that
clamps the image at the specified `min`/`max` values, subtracts the
`min` value, scales the result by multiplying by `scalefactor`, and
finally converts to type `T`.  If `scalefactor` is not specified, it
defaults to scaling the interval `[min,max]` to `[0,1]`.

Alternative constructors include `ScaleMinMax(T, img)` for which
`min`, `max`, and `scalefactor` are computed from the minimum and
maximum values found in `img`.

See also: `ScaleMinMaxNaN`, `ScaleAutoMinMax`, `MapNone`, `BitShift`.


*source:*
[Images/src/map.jl:222](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L222)

---

<a id="type__scalesigned.1" class="lexicon_definition"></a>
#### Images.ScaleSigned{T, S<:AbstractFloat} [¶](#type__scalesigned.1)
`ScaleSigned(T, scalefactor)` is a `MapInfo` object designed for
visualization of images where the pixel's sign has special meaning.
It multiplies the pixel value by `scalefactor`, then clamps to the
interval `[-1,1]`. If `T` is a floating-point type, it stays in this
representation.  If `T` is an `AbstractRGB`, then it is encoded as a
magenta (positive)/green (negative) image, with the intensity of the
color proportional to the clamped absolute value.


*source:*
[Images/src/map.jl:280](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L280)

## Internal

---

<a id="method__ando4_sep.1" class="lexicon_definition"></a>
#### ando4_sep() [¶](#method__ando4_sep.1)
`kern1, kern2 = ando4_sep()` returns separable approximations of the
optimal 4x4 filters for dimensions 1 and 2 of your image, as defined
in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3,
March 2000.

See also: `ando4`.


*source:*
[Images/src/edge.jl:72](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L72)

---

<a id="method__ando5_sep.1" class="lexicon_definition"></a>
#### ando5_sep() [¶](#method__ando5_sep.1)
`kern1, kern2 = ando5_sep()` returns separable approximations of the
optimal 5x5 filters for dimensions 1 and 2 of your image, as defined
in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3,
March 2000.

See also: `ando5`.


*source:*
[Images/src/edge.jl:103](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/edge.jl#L103)

---

<a id="method__call.1" class="lexicon_definition"></a>
#### call{T, From}(::Type{Images.ClampMinMax{T, From}},  ::Type{T},  min::From,  max::From) [¶](#method__call.1)
`ClampMinMax(T, minvalue, maxvalue)` is a `MapInfo` object that clamps
pixel values to be between `minvalue` and `maxvalue` before converting
to type `T`.

See also: `ClampMin`, `ClampMax`, and `Clamp`.


*source:*
[Images/src/map.jl:152](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/map.jl#L152)

---

<a id="method__convert.1" class="lexicon_definition"></a>
#### convert{T<:Real, N}(::Type{Array{T<:Real, N}},  img::Images.AbstractImageDirect{T<:Real, N}) [¶](#method__convert.1)
`A = convert(Array, img)` converts an Image `img` to an Array,
permuting dimensions (if needed) to put it in vertical-major (Matlab)
storage order.

See also `data`.


*source:*
[Images/src/core.jl:355](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L355)

---

<a id="method__convert.2" class="lexicon_definition"></a>
#### convert{T}(::Type{Images.Image{T, N, A<:AbstractArray{T, N}}},  img::Images.Image{T, N, A<:AbstractArray{T, N}}) [¶](#method__convert.2)
```
img = convert(Image, A)
img = convert(Image{HSV}, img)
```
Create a 2d Image from an array, setting up default properties. The
data array is assumed to be in "vertical-major" order, and an m-by-n-by-3 array
will be assumed to encode color along its third dimension.

Optionally, you can specify the desired colorspace of the returned `img`.

See also: `Image`, `grayim`, `colorim`.


*source:*
[Images/src/core.jl:337](https://github.com/timholy/Images.jl/tree/2bdb07dd39e71006375541b55496174042f2fc1f/src/core.jl#L337)

