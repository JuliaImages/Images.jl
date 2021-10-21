# Notes for JuliaImages Contributors

Welcome, and thanks for considering JuliaImages! Please be sure to respect our [community standards](https://julialang.org/community/standards) in all interactions.

We gratefully acknowledge the general [Julia CONTRIBUTING.md document](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md), from which much of this was adapted.

## Learning Julia

A pre-requisite for using JuliaImages is to know at least a little about Julia itself. [The learning page](https://julialang.org/learning) has a great list of resources for new and experienced users alike. The [Julia documentation](https://docs.julialang.org) covers the language and core library features, and is searchable.

## Learning JuliaImages

Our [main documentaion](https://juliaimages.org/stable/) provides an overview and some examples of using JuliaImages.
Many of the core packages are hosted at [JuliaImages](https://github.com/JuliaImages), but core components can also be found at [JuliaMath](https://github.com/JuliaMath), [JuliaGraphics](https://github.com/JuliaGraphics), and
[JuliaArrays](https://github.com/JuliaArrays).

## Before filing an issue

Julia's own "[How to file a bug report](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md#how-to-file-a-bug-report)" has many useful tips to help make sure that all necessary information is included.

Try to report the issue in the package responsible for the error.
Remember that `Images.jl` is primarily an umbrella package, and that most of the functionality is developed elsewhere.
You can often make good guesses by examining the backtrace (in cases where an
error is thrown), using `@which`, stepping in with the debugger, or just
using the search bar at the top left of [JuliaImages](https://github.com/JuliaImages).

## Contributing documentation

*By contributing you agree to be bound by JuliaImages' MIT license*

Many documentation issues are easy! Our narrative documentation at JuliaImages has its source at [juliaimages.org](https://github.com/JuliaImages/juliaimages.github.io). For small changes, you can just click on one of the files in the `docs/src` directory, click on the "pencil icon," and [edit it in your browser](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-another-users-repository). Any changes you suggest will first be vetted by an experienced developer, so there is no need to worry that you'll mess something up.

Changes to the "docstrings" (the string preceding a method in source code) should be made in the package in which they appear.

For bigger documentation changes, it is probably best to clone the package and submit the changes as an ordinary pull request, as described below under "Contributing code." You can build the package locally if you install [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl), and run `include("make.jl")` in the `docs/` folder. To see the completed documentation, open the `build/index.md` file in your browser.

Possible documentation contributions could be:
- Typo fixes
- Wording fixes to make something more clear
- Additional examples that highlight some functionality
- Find more [documentation issues](https://github.com/JuliaImages/Images.jl/labels/documentation)

## Contributing code

*By contributing you agree to be bound by JuliaImages' MIT license*

If you've never submitted a pull request before, it can take a little while to become familiar with the process. In addition to the steps below, [GitHub has a tutorial and exercises](https://try.github.io/). See also the excellent [Git book](https://git-scm.com/book/en/v2). There are also many good external tutorials on this subject, like [this one](https://yangsu.github.io/pull-request-tutorial/).

### Contributor Checklist

* Create a [GitHub account](https://github.com/signup/free).

* If you plan to fix a bug, feel free to first report the bug as an issue on its own.
  In the text, you can mention whether you're planning on addressing it yourself.
  *Pro tip*: if you do submit a pull request to fix it, put "Fixes #<issue number>" in the commit message and it will close automatically when your pull request is merged.

  If you're concerned your change might be controversial, you can also use an issue to propose your change in general terms and discuss it before implementation.

* Fork whatever repository you plan to commit to by clicking on the "Fork" button at the upper-right of the home page.

* If you haven't already implemented your changes, check the package out for development: hit `]` in the Julia REPL and then type (for example) `dev Images`.
You'll get a copy of the full repository in your `~/.julia/dev` folder. See the [package manager documentation](https://julialang.github.io/Pkg.jl/v1/) for further details.

* Make your changes. Generally you should be working on a branch, so your work doesn't conflict with ongoing development in the `master` branch. Ensure you follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/index.html) for your contribution.

* Test your changes. We aspire to have test coverage for every bit of "user visible" functionality. Tests are stored, appropriately, in the `test/` folder of each package. You can run existing tests yourself and add new ones. Sometimes testing is more work than the actual change itself, but having tests ensures that no well-meaning future developer will accidentally mess up your functionality---it's worth it!  *"A fix is for today. A test is forever."*

* Submit your changes up to your fork and then submit a pull request---whoopee!

* See what happens to the automated tests that run on Travis and/or AppVeyor. If there are errors, check the logs and see whether they look like they are related to your changes; if so, try to fix the problem by adding new commits to your pull request. Once the tests pass, hooray! :tada:

* Relax and wait for feedback. We try to review contributions quickly and courteously. But we are human, and sometimes we get busy with other things or fail to notice an email; if it's been a while since you submitted your pull request, try posting a polite reminder about the existence of your pull request.

* Discuss any feedback you receive as necessary. It's fine to defend your approach, but also be open to making changes based on suggestions you receive.

* Sooner or later, the fate of your pull request will become clear. If it gets approved, an established contributor will merge it. It's not officially released into the wild until a contributor releases a new version of the package; if that doesn't happen quickly, don't hesitate to make an inquiry in case it's simply been overlooked.

From the whole team, thanks in advance for your contribution!

### Contribution tips

* [Revise](https://github.com/timholy/Revise.jl) is a package that
tracks changes in source files and automatically updates function
definitions in your running Julia session. Using it, you can make
extensive changes without needing to rebuild the package in order to test
your changes.

* Debuggers can help you get to the root of a problem. There are many choices and interfaces:
  + [VSCode](https://code.visualstudio.com/docs/languages/julia#_debugging) has a polished GUI for debugging
  + [Debugger](https://github.com/JuliaDebug/Debugger.jl) has a polished command-line interface
  + [Rebugger](https://github.com/timholy/Rebugger.jl) has an innovative but somewhat less-polished command-line interface
  + [Infiltrator](https://github.com/JuliaDebug/Infiltrator.jl) offers more limited debugging, but often it's precisely what you need while avoiding the performance penalties that some of the other options suffer from.
