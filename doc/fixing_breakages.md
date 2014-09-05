# Fixing breakages from changes in Images

There were some big changes between the 0.3 and 0.4 series.
Every attempt has been made to introduce deprecations; for example, the old `ImageView` works
out-of-the-box with the new Images. Nevertheless, omissions are likely,
and certain things cannot be deprecated.
Here are some tips on moving over to the new version of Images:

- Images are returned as `ColorValue` arrays, meaning that a color, two-dimensional image will
now have two dimensions. If your code needs a color dimension to the array, try `reinterpret` or
`separate` as described in the [README](../README.md). That said, it's probably preferable to leave
it in ColorValue format, so you can leverage the power of the [Color package](https://github.com/JuliaLang/Color.jl).

- Many images will be returned in terms of a `Ufixed8` type, which acts like a "floating point" (really, fractional)
number but internally is represented as a Uint8. If you're used to thinking of 255 as "saturated" when you
happen to know that the underlying image is of integral type, now you should adjust to thinking of 1.0 as
saturated, no matter whether your image is "integral" (`Ufixed`) or floating-point.

- `Scale*` has generally been replaced by `Map*` (examples: `ScaleInfo -> MapInfo`, `ScaleNone -> MapNone`).
If you used the functions you should get deprecation warnings, but any direct usage of the types
(e.g., `ScaleNone{Float32}()`) cannot be deprecated. So you'll get errors, rather than warnings, in such circumstances.

If you discover other forms of breakage, please feel free to add to this page (click the pencil icon at the top to edit it).
