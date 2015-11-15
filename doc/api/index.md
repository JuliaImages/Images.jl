# API-INDEX


## MODULE: Images

---

## Modules [Exported]

[Images](Images.md#module__images.1)  `Images` is a package for representing and processing images.

---

## Methods [Exported]

[OverlayImage(channels::Tuple{Vararg{AbstractArray{T, N}}},  colors::Tuple{Vararg{ColorTypes.Colorant{T, N}}},  arg)](Images.md#method__overlayimage.1)  `OverlayImage` is identical to `Overlay`, except that it returns an Image.

[ando3()](Images.md#method__ando3.1)  `kern1, kern2 = ando3()` returns optimal 3x3 filters for dimensions 1 and 2 of your image, as defined in

[ando4()](Images.md#method__ando4.1)  `kern1, kern2 = ando4()` returns optimal 4x4 filters for dimensions 1 and 2 of your image, as defined in

[ando5()](Images.md#method__ando5.1)  `kern1, kern2 = ando5()` returns optimal 5x5 filters for dimensions 1 and 2 of your image, as defined in

[assert2d(img::AbstractArray{T, N})](Images.md#method__assert2d.1)  `assert2d(img)` triggers an error if the image has more than two spatial

[assert_scalar_color(img::AbstractArray{T, N})](Images.md#method__assert_scalar_color.1)  `assert_scalar_color(img)` triggers an error if the image uses an

[assert_timedim_last(img::AbstractArray{T, N})](Images.md#method__assert_timedim_last.1)  `assert_timedim_last(img)` triggers an error if the image has a time

[assert_xfirst(img::AbstractArray{T, N})](Images.md#method__assert_xfirst.1)  `assert_xfirst(img)` triggers an error if the first spatial dimension

[assert_yfirst(img)](Images.md#method__assert_yfirst.1)  `assert_yfirst(img)` triggers an error if the first spatial dimension

[colordim{C<:ColorTypes.Colorant{T, N}}(img::AbstractArray{C<:ColorTypes.Colorant{T, N}, 1})](Images.md#method__colordim.1)  `dim = colordim(img)` returns the dimension used to encode color, or 0

[colorim(A::Images.AbstractImage{T, N})](Images.md#method__colorim.1)  ```

[colorspace{C<:ColorTypes.Colorant{T, N}}(img::AbstractArray{C<:ColorTypes.Colorant{T, N}, 1})](Images.md#method__colorspace.1)  `cs = colorspace(img)` returns a string specifying the colorspace

[coords_spatial(img)](Images.md#method__coords_spatial.1)  `c = coords_spatial(img)` returns a vector listing the spatial

[copyproperties(img::AbstractArray{T, N},  data::AbstractArray{T, N})](Images.md#method__copyproperties.1)  ```

[data(img::AbstractArray{T, N})](Images.md#method__data.1)  ```

[getindexim(img::Images.AbstractImage{T, N},  I::Union{AbstractArray{T<:Real, N}, T<:Real}...)](Images.md#method__getindexim.1)  ```

[grayim(A::Images.AbstractImage{T, N})](Images.md#method__grayim.1)  ```

[height(img::AbstractArray{T, N})](Images.md#method__height.1)  `h = height(img)` returns the vertical size of the image, regardless

[imedge(img::AbstractArray{T, N})](Images.md#method__imedge.1)  ```

[imedge(img::AbstractArray{T, N},  method::AbstractString)](Images.md#method__imedge.2)  ```

[imedge(img::AbstractArray{T, N},  method::AbstractString,  border::AbstractString)](Images.md#method__imedge.3)  ```

[imgradients(img::AbstractArray{T, N})](Images.md#method__imgradients.1)  ```

[imgradients(img::AbstractArray{T, N},  method::AbstractString)](Images.md#method__imgradients.2)  ```

[imgradients(img::AbstractArray{T, N},  method::AbstractString,  border::AbstractString)](Images.md#method__imgradients.3)  ```

[isdirect(img::AbstractArray{T, N})](Images.md#method__isdirect.1)  `isdirect(img)` returns true if `img` encodes its values directly,

[isxfirst(img::AbstractArray{T, N})](Images.md#method__isxfirst.1)  `tf = isxfirst(img)` tests whether the first spatial dimension is `"x"`.

[isyfirst(img::AbstractArray{T, N})](Images.md#method__isyfirst.1)  `tf = isyfirst(img)` tests whether the first spatial dimension is `"y"`.

[label_components(A)](Images.md#method__label_components.1)  ```

[label_components(A,  connectivity)](Images.md#method__label_components.2)  ```

[label_components(A,  connectivity,  bkg)](Images.md#method__label_components.3)  ```

[magnitude(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N})](Images.md#method__magnitude.1)  ```

[magnitude_phase(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N})](Images.md#method__magnitude_phase.1)  `m, p = magnitude_phase(grad_x, grad_y)`

[mapinfo{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}}(::Type{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}},  img::AbstractArray{T<:FixedPointNumbers.UFixed{T<:Unsigned, f}, N})](Images.md#method__mapinfo.1)  `mapi = mapinf(T, img)` returns a `MapInfo` object that is deemed

[ncolorelem(img)](Images.md#method__ncolorelem.1)  `n = ncolorelem(img)` returns the number of color elements/voxel, or 1 if color is not a separate dimension of the array.

[nimages(img)](Images.md#method__nimages.1)  `n = nimages(img)` returns the number of time-points in the image

[orientation{T}(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N})](Images.md#method__orientation.1)  ```

[phase{T}(grad_x::AbstractArray{T, N},  grad_y::AbstractArray{T, N})](Images.md#method__phase.1)  ```

[pixelspacing{T}(img::AbstractArray{T, 3})](Images.md#method__pixelspacing.1)  ```

[prewitt()](Images.md#method__prewitt.1)  `kern1, kern2 = prewitt()` returns Prewitt filters for dimensions 1 and 2 of your image

[properties(A::AbstractArray{T, N})](Images.md#method__properties.1)  `prop = properties(img)` returns the properties-dictionary for an

[raw(img::AbstractArray{T, N})](Images.md#method__raw.1)  ```

[sc(img::AbstractArray{T, N})](Images.md#method__sc.1)  ```

[sdims(img)](Images.md#method__sdims.1)  `n = sdims(img)` is similar to `ndims`, but it returns just the number of *spatial* dimensions in

[separate{CV<:ColorTypes.Colorant{T, N}}(img::Images.AbstractImage{CV<:ColorTypes.Colorant{T, N}, N})](Images.md#method__separate.1)  `imgs = separate(img)` separates the color channels of `img`, for

[shareproperties(img::AbstractArray{T, N},  data::AbstractArray{T, N})](Images.md#method__shareproperties.1)  ```

[size_spatial(img)](Images.md#method__size_spatial.1)  ```

[sliceim(img::Images.AbstractImage{T, N},  I::Union{Colon, Int64, Range{Int64}}...)](Images.md#method__sliceim.1)  ```

[sobel()](Images.md#method__sobel.1)  `kern1, kern2 = sobel()` returns Sobel filters for dimensions 1 and 2 of your image

[spacedirections(img::AbstractArray{T, N})](Images.md#method__spacedirections.1)  ```

[spatialorder(::Type{Array{T, 2}})](Images.md#method__spatialorder.1)  ```

[spatialorder(img::Images.AbstractImage{T, N})](Images.md#method__spatialorder.2)  ```

[spatialpermutation(to,  img::AbstractArray{T, N})](Images.md#method__spatialpermutation.1)  ```

[spatialproperties(img::Images.AbstractImage{T, N})](Images.md#method__spatialproperties.1)  ```

[storageorder(img::AbstractArray{T, N})](Images.md#method__storageorder.1)  ```

[subim(img::Images.AbstractImage{T, N},  I::Union{Colon, Int64, Range{Int64}}...)](Images.md#method__subim.1)  ```

[thin_edges{T}(img::AbstractArray{T, 2},  gradientangles::AbstractArray{T, N})](Images.md#method__thin_edges.1)  ```

[thin_edges{T}(img::AbstractArray{T, 2},  gradientangles::AbstractArray{T, N},  border::AbstractString)](Images.md#method__thin_edges.2)  ```

[timedim(img)](Images.md#method__timedim.1)  `dim = timedim(img)` returns the dimension used to represent time, or

[width(img::AbstractArray{T, N})](Images.md#method__width.1)  `w = width(img)` returns the horizontal size of the image, regardless

[widthheight(img::AbstractArray{T, N},  p)](Images.md#method__widthheight.1)  `w, h = widthheight(img)` returns the width and height of an image, regardless of storage order.

---

## Types [Exported]

[Images.AbstractImageDirect{T, N}](Images.md#type__abstractimagedirect.1)  `AbstractImageDirect` is the supertype of all images where pixel

[Images.AbstractImageIndexed{T, N}](Images.md#type__abstractimageindexed.1)  `AbstractImageIndexed` is the supertype of all "colormap" images,

[Images.BitShift{T, N}](Images.md#type__bitshift.1)  `BitShift{T,N}` performs a "saturating rightward bit-shift" operation.

[Images.Clamp01NaN{T}](Images.md#type__clamp01nan.1)  `Clamp01NaN(T)` constructs a `MapInfo` object that clamps grayscale or

[Images.ClampMax{T, From}](Images.md#type__clampmax.1)  `ClampMax(T, maxvalue)` is a `MapInfo` object that clamps pixel values

[Images.ClampMin{T, From}](Images.md#type__clampmin.1)  `ClampMin(T, minvalue)` is a `MapInfo` object that clamps pixel values

[Images.Clamp{T}](Images.md#type__clamp.1)  `Clamp(C)` is a `MapInfo` object that clamps color values to be within

[Images.ImageCmap{T<:ColorTypes.Colorant{T, N}, N, A<:AbstractArray{T, N}}](Images.md#type__imagecmap.1)  ```

[Images.Image{T, N, A<:AbstractArray{T, N}}](Images.md#type__image.1)  ```

[Images.MapInfo{T}](Images.md#type__mapinfo.1)  `MapInfo{T}` is an abstract type that encompasses objects designed to

[Images.MapNone{T}](Images.md#type__mapnone.1)  `MapNone(T)` is a `MapInfo` object that converts `x` to have type `T`.

[Images.Overlay{T, N, NC, AT<:Tuple{Vararg{AbstractArray{T, N}}}, MITypes<:Tuple{Vararg{Images.MapInfo{T}}}}](Images.md#type__overlay.1)  ```

[Images.ScaleAutoMinMax{T}](Images.md#type__scaleautominmax.1)  `ScaleAutoMinMax(T)` constructs a `MapInfo` object that causes images

[Images.ScaleMinMaxNaN{To, From, S}](Images.md#type__scaleminmaxnan.1)  `ScaleMinMaxNaN(smm)` constructs a `MapInfo` object from a

[Images.ScaleMinMax{To, From, S<:AbstractFloat}](Images.md#type__scaleminmax.1)  `ScaleMinMax(T, min, max, [scalefactor])` is a `MapInfo` object that

[Images.ScaleSigned{T, S<:AbstractFloat}](Images.md#type__scalesigned.1)  `ScaleSigned(T, scalefactor)` is a `MapInfo` object designed for

---

## Methods [Internal]

[ando4_sep()](Images.md#method__ando4_sep.1)  `kern1, kern2 = ando4_sep()` returns separable approximations of the

[ando5_sep()](Images.md#method__ando5_sep.1)  `kern1, kern2 = ando5_sep()` returns separable approximations of the

[call{T, From}(::Type{Images.ClampMinMax{T, From}},  ::Type{T},  min::From,  max::From)](Images.md#method__call.1)  `ClampMinMax(T, minvalue, maxvalue)` is a `MapInfo` object that clamps

[convert{T<:Real, N}(::Type{Array{T<:Real, N}},  img::Images.AbstractImageDirect{T<:Real, N})](Images.md#method__convert.1)  `A = convert(Array, img)` converts an Image `img` to an Array,

[convert{T}(::Type{Images.Image{T, N, A<:AbstractArray{T, N}}},  img::Images.Image{T, N, A<:AbstractArray{T, N}})](Images.md#method__convert.2)  ```

