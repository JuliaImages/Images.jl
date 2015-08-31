---
title: Overlays
author: Tim Holy
order: 60
...

<h1>Overlays</h1>

Frequently one wants to combine two (or more) grayscale images into a single
colorized image.  `Images` defines an `AbstractArray` type, `Overlay`, making
this straightforward.

To create an overlay, use the following syntax:

```{.julia execute="false"}
O = Overlay((gray1,gray2,...), (color1,color2,...), (clim1,clim2,...))
```

Here `gray1` and `gray2` are the arrays representing individual "channels" of
information, each equivalent to a grayscale image.  `color1` and `color2` are
the [Colors](https://github.com/JuliaGraphics/Colors.jl) that will be used for the
corresponding grayscale arrays; for example, to put `gray1` in the red channel,
you'd use `color1 = RGB(1,0,0)`.  (You can choose any RGB value you want, it
doesn't have to be a "pure" RGB channel.)  Finally, `clim1` and `clim2`
represent the intensity-scaling applied to each image; setting `clim1 =
(400,3000)` would send any values in `gray1` less than 400 to black, and any
values bigger than 3000 to red, with other values between encoded linearly.

Once constructed, an `Overlay` acts as an `Array{RGB}`. You can embed it in an
`Image` or just use it directly.
