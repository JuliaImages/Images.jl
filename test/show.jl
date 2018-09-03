using Images

module ShowTmp
# run this in a separate module WITHOUT Images until
# the module trickery in the ImageShow tests is sorted.
import ImageShow
include(joinpath(ImageShow.testdir(),"runtests.jl"))
end
