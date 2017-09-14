using Documenter, Images

makedocs(doctest=false)

deploydocs(deps=Deps.pip("pygments", "mkdocs", "mkdocs-material"),
           repo="github.com/timholy/Images.jl.git",
           julia="0.4",
           osname="linux")
