# Images.jl

[![][action-img]][action-url]
[![][pkgeval-img]][pkgeval-url]
[![][codecov-img]][codecov-url]
[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]

Images.jl is an open-source image processing library for [Julia](http://julialang.org/).

## Project organization

Images.jl is increasingly becoming an "umbrella package" that exports a [set of packages](https://juliaimages.org/latest/pkgs/) which are useful for common image processing tasks.
Most of these packages are hosted at
[JuliaImages](https://github.com/JuliaImages),
[JuliaArrays](https://github.com/JuliaArrays),
[JuliaIO](https://github.com/JuliaIO),
[JuliaGraphics](https://github.com/JuliaGraphics), and
[JuliaMath](https://github.com/JuliaMath).

## Getting Help

You can join the Julia community by joining [slack](https://julialang.slack.com) (get an invite from https://julialang.org/slack/), [zulip](https://julialang.zulipchat.com/), and/or [discourse](https://discourse.julialang.org/).
For questions and discussions related to the JuliaImages ecosystem, you can [open an discussion](https://github.com/JuliaImages/Images.jl/discussions); issues are reserved to bug reports and feature tracking.
Any question about [the documentation](https://juliaimages.org/) is considered an issue, hence if you have any questions please feel free to ask it in [the documentation repo](https://github.com/JuliaImages/juliaimages.github.io/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started contributing to JuliaImages.

We are a welcoming and supportive community made up of volunteer developers.
All participants are expected to abide by the [Julia Community Standards](https://julialang.org/community/standards/).

## Compatibility

The codebase and dependency of Images toolbox is delibrately maintained to support all Julia minor
versions since the latest long-term-support(LTS) Julia version, thus you can expect it to work on
Julia >= v1.6. Note that this is only true for the latest Images version, and currently we do not
have Images LTS version. For the best experience, we recommend you to use the latest stable Julia
version even if it is not LTS version.

Images v0.24 is the last minor version that is compatible to the Julia 1.0. It will still be under
maintenance, but only with minimal efforts from the community. No forward compatibility guarantee
will be made, which means that you might see APIs and behaviors of Images v0.24 are quite different
from that of the latest Images version.

## Credits

Elements of this package descend from "image.jl"
that once lived in Julia's `extras/` directory.
That file had several authors, of which the primary were
Jeff Bezanson, Stefan Kroboth, Tim Holy, Mike Nolta, and Stefan Karpinski.
This repository has been quite heavily reworked;
please see the "contributors" tab above and on many of the other repositories at [JuliaImages](https://github.com/JuliaImages) and elsewhere.

<!-- URLS -->

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/I/Images.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html
[action-img]: https://github.com/JuliaImages/Images.jl/workflows/Unit%20test/badge.svg
[action-url]: https://github.com/JuliaImages/Images.jl/actions
[codecov-img]: https://codecov.io/github/JuliaImages/Images.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/JuliaImages/Images.jl?branch=master
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliaimages.org/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliaimages.org/latest
