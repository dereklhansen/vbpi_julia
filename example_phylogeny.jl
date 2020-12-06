using Revise
using PhyloTrees, Plots, StatsBase
using Flux
using Distributions

include("Sbn.jl")
include("VBPI.jl")

import .Sbn.plot_and_annotate

example_tree = Tree()
addnode!(example_tree)

branch!(example_tree, 1, 1.0)
branch!(example_tree, 2, 1.0)
branch!(example_tree, 3, 1.0)
branch!(example_tree, 3, 1.0)
branch!(example_tree, 2, 1.0)
branch!(example_tree, 1, 1.0)
# branch!(example_tree, 1, rand())
example_tree.node_to_label[7] = :D
example_tree.node_to_label[6] = :C
example_tree.node_to_label[5] = :B
example_tree.node_to_label[4] = :A

example_tree.label_to_node[:D] = 7
example_tree.label_to_node[:C] = 6
example_tree.label_to_node[:B] = 5
example_tree.label_to_node[:A] = 4


plot_and_annotate(example_tree)

example_sbn = Sbn.make_sbn(example_tree, 1, missing, 1, missing)
Sbn.hashtree(example_sbn)
tree_from_sbn = Sbn.make_tree_from_sbn(example_sbn)
plot_and_annotate(tree_from_sbn, true)

## VBPI
## Can we use Flux with the dictionaries directly?
using Flux

table = VBPILookup.VBPI([:A, :B, :C, :D])
flx   = params(table.probs)
table

table_grad = gradient(flx) do 
    p = table[[:A, :B, :C, :D], [:A, :B, :C]]
    (p - 0.5)^2
end 

opt=Descent()

Flux.update!(opt, flx, table_grad)
table[[:A, :B, :C, :D], [:A, :B, :C]]



leaves = [:A, :B, :C, :D]
table = VBPILookup.VBPI(leaves)
all_sbns=VBPILookup.enumerate_all_sbns(leaves, 1, length(leaves)-1);
# Sbn.hashtree(all_sbns[8]) == Sbn.hashtree(example_sbn)

β = 0.008
all_sbn_probs = rand(Dirichlet(β*fill(1.0, length(all_sbns))))
plot_and_annotate(Sbn.make_tree_from_sbn(all_sbns[5000]), true)

all_sbn_hash = Sbn.hashtree.(all_sbns)
all_sbn_dict = Dict(zip(all_sbn_hash, all_sbn_probs))
# all_sbn_dict[Sbn.hashtree(example_sbn)]

rand_tree, lprob = VBPILookup.rsbn(leaves, table, 1, length(leaves)-1)
VBPILookup.Sbn.plot_and_annotate(rand_tree)

VBPILookup.dsbn(leaves, rand_tree, table, 1, length(leaves)-1)

## Calculate the ELBO given a particular sbn

## We "know" the true evidence is zero, because 
## we have a prespecified distribution over trees

VBPILookup.est_elbo(leaves, table, all_sbn_dict)
# elbo(rand_tree)


out=VBPILookup.vimco_objective(leaves, table, all_sbn_dict, 100)


table = VBPILookup.VBPI(leaves)
losses = VBPILookup.train_sbn(leaves, table, all_sbn_dict; epochs=50000, cb=100)
function expsmooth(xs, alpha)
    y = deepcopy(xs)
    y[1] = xs[1]
    for i in 2:length(xs)
        y[i] = alpha * y[i-1] + (1-alpha) * xs[i]
    end
    return y
end
plot(expsmooth(-losses, .99))

logq=VBPILookup.dsbn.([leaves], all_sbns, [table], [1], [length(leaves)-1])

hcat(all_sbn_probs, exp.(logq))





leaves = [:A, :B, :C, :D, :E, :F, :G]
table = VBPILookup.VBPI(leaves)
typeof(table)

all_sbns=VBPILookup.enumerate_all_sbns(leaves, 1, length(leaves)-1);

all_sbn_probs = rand(Dirichlet(0.01*fill(1.0, length(all_sbns))))

all_sbn_hash = Sbn.hashtree.(all_sbns)
all_sbn_dict = Dict(zip(all_sbn_hash, all_sbn_probs))

@time losses = VBPILookup.train_sbn(leaves, table, all_sbn_dict; epochs=10, cb=1)
ProfileView.@profview losses = VBPILookup.train_sbn(leaves, table, all_sbn_dict; epochs=100, cb=100)

function vimcos(leaves, table, all_sbn_dict, N)
    out = 0.0
    for i in 1:N
        out += VBPILookup.vimco_objective(leaves, table, all_sbn_dict, 50)
    end
    return out
end

@time vimcos(leaves, table, all_sbn_dict, 10)

ProfileView.@profview vimcos(leaves, table, all_sbn_dict, 1000)