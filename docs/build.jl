using Images, Docile

makedocs()

cp(joinpath(dirname(@__FILE__), "src", "img"), joinpath(dirname(@__FILE__), "build", "img"), remove_destination=true)
cp(joinpath(dirname(@__FILE__), "..", "LICENSE.md"), joinpath(dirname(@__FILE__), "build", "LICENSE.md"), remove_destination=true)

# When ready, deploy from top level Images dir with:
#   mkdocs gh-deploy --clean
