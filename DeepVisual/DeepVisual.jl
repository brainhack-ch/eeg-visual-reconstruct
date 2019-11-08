module DeepVisual
using EDF
using Glob
using DataFrames, CSV
using Plots

searchdir(path,key) = glob(key, path)

mutable struct Session
    rec::EDF.File
    ann::DataFrame
    bad_channels::Array{String}
    Session()=new()
    Session(rec, ann, bad_channels)=new(rec, ann, bad_channels)
    Session(ann, bad_channels)=begin
        s = Session()
        s.ann = ann
        s.bad_channels = bad_channels
        s
    end
    Session(path) = begin
        #edfpath = searchdir(path, "*.edf")[1]
        ann = CSV.read("$path/ann.csv")
        ann = ann[completecases(ann), :]
        bad_channels = open("$path/../BAD_CHANNELS.txt") |> readlines
        Session(EDF.read(edfpath), ann, bad_channels)
    end
end

struct Individual
    session1::Session
    session2::Session
    Individual(session1, session2)=new(session1, session2)
end

individuals = Individual[]
for indiv in 1:1
    if ind in [7 13 17]
        continue
    if (div(indiv, 10) == 0)
        n = "0$indiv"
    else
        n = "$indiv"
    end
    dir = "DATA/S$n"
    s1 = Session("$dir/Session1")
    s2 = Session("$dir/Session2")
    push!(individuals, Individual(s1, s2))
end

data = [ searchdir("DATA/_screenshots/", "$i*")|>length for i in 0:4 ]
labels = CSV.read("DATA/Label Meaning.txt", delim = ' ', header = -1)[:, 3]
bar(labels, data, title = "Sequences class distribution", titlefontsize = 12, legend=false)

# TODO read from signal dataframe stacked vectors of signals with intervals from anno
# at 500 Hz (round(sec*500) is image begin)
# minframes = individuals[1].session1.rec.signals
# for

end
