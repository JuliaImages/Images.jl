# Fixing errors:

- Errors about properties like `timedim` should have self-explanatory
  messages, but be aware that switching to the recommended `img =
  AxisArray(A, :y, :x, :time)` may require other changes in your
  code. For example, rather than `view(img, "t", t)` you should use
  `tax = timeaxis(img); view(img, tax(t))`. (The new API has the
  advantage of being inferrable and thus can be used in
  performance-critical applications.)

- `MethodError: no method matching getindex(::Tuple{Int64,Int64,Int64}, ::Tuple{Int64,Int64})`
  in conjunction with `coords_spatial`: the return type is now a tuple
  rather than an array. You can use `[coords_spatial(img)...]` to get
  an array, although in some cases you may prefer to use the tuple
  directly. For certain applications the tuple improves inferrability,
  for example `size(img, coords_spatial(img)...)` now returns a tuple
  of inferrable length, which is helpful for writing type-stable
  algorithms.

- `padarray` now returns arrays with unconventional indices: if you
  pad a 1-dimensional array by 3 elements at the beginning and end,
  the output's indices will be `-2:sz+3` rather than `1:sz+6`. If you
  find it easier to work with arrays that start at 1, you can call
  `parent` on the output.
