using Requires

@require Juno begin
    import Media
    if Juno.isactive()
        Juno.media(Images.ColorantMatrix, Media.Graphical)

        function Juno.render(pane::Juno.PlotPane, img::Images.ColorantMatrix)
            Juno.render(pane, HTML("<img src=\"data:image/png;base64,$(stringmime(MIME("image/png"), img))\">"))
        end
    end
end
