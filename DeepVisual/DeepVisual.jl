module DeepVisual
using EDF
using Glob
using DataFrames, CSV
using Plots
using Flux
using Flux: mse, batch
using ClassImbalance
using Random: shuffle

searchdir(path,key) = glob(key, path)

mutable struct Session
    rec::EDF.File
    ann::DataFrame
    bad_channels::Array{String}
    sampling_rate::Int
    Session()=new()
    Session(rec, ann, bad_channels, sampling_rate)=new(rec, ann, bad_channels, sampling_rate)
    Session(path) = begin
        edfpath = searchdir(path, "*.edf")[1]
        ann = CSV.read("$path/ann.csv")
        ann = ann[completecases(ann), :]
        bad_channels = open("$path/../BAD_CHANNELS.txt") |> readlines
        sampling_rate = 500 #Hz
        Session(EDF.read(edfpath), ann, bad_channels, sampling_rate)
    end
end

struct Individual
    sessions::Array{Session,1}
    Individual(session1, session2)=new([session1, session2])
end


datarel = "./DeepVisual"

individuals = Individual[]
for indiv in 1:1
    if indiv in [7 13 17] #too many bad_channels
        continue
    end
    if (div(indiv, 10) == 0)
        n = "0$indiv"
    else
        n = "$indiv"
    end
    dir = "$datarel/DATA/S$n"
    s1 = Session("$dir/Session1")
    s2 = Session("$dir/Session2")
    push!(individuals, Individual(s1, s2))
end

data = [ searchdir("$datarel/DATA/_screenshots/", "$i*")|>length for i in 0:4 ]
labels = CSV.read("$datarel/DATA/Label Meaning.txt", delim = ' ', header = -1)[:, 3]
bar(labels, data, title = "Sequences class distribution", titlefontsize = 12, legend=false)

# TODO read from signal dataframe stacked vectors of signals with intervals from anno
# at 500 Hz (round(sec*500) is image begin)
# minframes = individuals[1].session1.rec.signals
function workSession!(d::Dict{String, Array{Float32, 2}}, s::Session)
    signalArray = [ st for st in EDF.decode.(s.rec.signals) ] |> batch
    for timeframe in eachrow(s.ann)
        startframe = trunc(Int, timeframe[:onsets_start]*s.sampling_rate)
        endframe = trunc(Int, timeframe[:onsets_end]*s.sampling_rate)
        labelord = timeframe[:labels] + 1#label successive number
        if labels[labelord] in keys(d)
            d[labels[labelord]] = vcat(d[labels[labelord]],
            signalArray[startframe:endframe,:])
        else
            d[labels[labelord]] = signalArray[startframe:endframe,:]
        end
    end
end

classSignalVectors = Dict{String, Array{Float32, 2}}()
for individual in individuals
    for session in individual.sessions
        workSession!(classSignalVectors, session)
    end
end

indexof(el, a) = findall(x -> x == el, a)[1]

x = zeros(0)
y = zeros(0)


data = map(k -> begin
    trc = 230000
    v = classSignalVectors[k]
    if size(v)[1] > trc
        hcat(v[1:trc,:], fill(indexof(k, labels), trc))
    end
end, [ k for k in keys(classSignalVectors)])
dm = reduce(vcat, filter(x -> x !== nothing, data))

function partitionData(data, at = 0.7)
    n = size(data)[1]
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

train, testval = partitionData(dm, 0.7)
test, val = partitionData(testval, 0.66)

m = Chain(
  Dense(256, 256, relu),
  Dense(256, 32, relu),
  Dense(32, 1),
  relu) |> gpu

loss(x, y) = mse(m(x), y)
opt = ADAM()

dataset = (train[:, 1:131], train[:, 132])

Flux.train!(loss, params(m), dataset, opt)

end
