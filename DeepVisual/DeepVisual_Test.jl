using Test
include("./DeepVisual.jl")

@test DeepVisual.searchdir("DATA/S01/Session2/", "*.edf")[1] == "NeoRec_2018-10-04_13-57-22.edf"
