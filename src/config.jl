##########  Configuration   #########

have_imagemagick = false
if system("which convert > /dev/null") != 0
    println("Warning: ImageMagick utilities not found. Install for more file format support.")
else
    have_imagemagick = true
end
use_imshow_cmd = false
imshow_cmd = ""
use_gaston = false
# TODO: Windows
imshow_cmd_list = ["feh", "gwenview"]
for thiscmd in imshow_cmd_list
    if system("which $thiscmd > /dev/null") == 0
        use_imshow_cmd = true
        imshow_cmd = thiscmd
        break
    end
end
if !use_imshow_cmd
    try
        x = gnuplot_state
        use_gaston = true
    catch
        println("Warning: no image viewer found. You will not be able to see images.")
    end
end


