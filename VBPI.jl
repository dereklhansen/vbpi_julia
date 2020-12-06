module VBPILookup
    using Combinatorics
    using Zygote
    using Flux
    using StatsBase
    using KissThreading: tmap
    using Dates
    using Printf
    include("Sbn.jl")
    struct VBPI{Q}
        lookup_table::Dict{Tuple{Vector{Symbol}, Vector{Symbol}}, Int}
        split_dict::Dict{Vector{Symbol}, Vector{Vector{Symbol}}}
        probs::Vector{Q}
    end

    function VBPI(leaves::Vector{Symbol})
        table               = generate_lookup_table(leaves)
        tableparams,  d     = make_flux_params_for_table!(table)
        split_dict          = Dict{Vector{Symbol}, Vector{Vector{Symbol}}}(zip(keys(table), (collect(keys(x[2])) for x in table)))
        VBPI(d, split_dict, tableparams)
    end

    function split_prob(table::VBPI, clade::Vector{Symbol}, split::Vector{Symbol})

    end
    import Base.getindex
    function getindex(x::VBPI, clade::Vector{Symbol}, split::Vector{Symbol})
        i = prob_table_lookup_ng(x.lookup_table, clade, split)
        x.probs[i]
    end

    function generate_lookup_table(leaves)
        # Get all possible clades which have a non-deterministic split
        # This is all clades of size 3 or greater
        clades = powerset(leaves, 3, length(leaves))
        prob_tables = initialize_prob_lookup_table_clade.(clades)
        return Dict(zip(clades, prob_tables))
    end

    function generate_possible_clade_splits(clade)
        splits = powerset(clade[2:end], 0, length(clade) - 2)
        return vcat.(clade[1], splits)
    end

    function initialize_prob_lookup_table_clade(clade)
        possible_splits = generate_possible_clade_splits(clade)
        probs           = fill(0.0, length(possible_splits))
        probs           = map(x -> [x], probs)
        return Dict{Vector{Symbol}, AbstractVector{Float64}}(zip(possible_splits, probs))
    end

    function make_flux_params_for_table!(table)
        ps_flat = Vector{Float64}()
        # ps = Flux.Params()
        d = Dict{Tuple{Vector{Symbol}, Vector{Symbol}}, Int}()
        let i = 1
            for (clade, probs) in table
                for (split, prob) in probs
                    push!(ps_flat, prob[1])
                    probs[split] = view(ps_flat, length(ps_flat):length(ps_flat))
                    d[(clade, split)] = i
                    i += 1
                end
            end
        end
        # ps = params(ps_flat)
        return ps_flat, d
    end

    # No-gradient table-lookup 
    using Zygote
    function prob_table_lookup_ng(table, clade, split)
        table[(clade,split)]
    end

    # No-gradient get-intex
    function get_index_ng(x, i)
        x[i]
    end


    Zygote.@nograd prob_table_lookup_ng
    Zygote.@nograd get_index_ng

    function enumerate_all_sbns(clade, depth, max_depth)
        trees  = Vector{Any}(undef, 0)
        if length(clade) == 1
            clade_obj   = Sbn.SubsplitClade(clade, missing)
            if depth < max_depth
                childtrees1 = enumerate_all_sbns(clade, depth+1, max_depth)
                childtrees2 = enumerate_all_sbns(clade, depth+1, max_depth)
                for child1 in childtrees1
                    for child2 in childtrees2
                        node = Sbn.BinaryNode(clade_obj)
                        child1.parent=node
                        child2.parent=node
                        node.left=child1
                        node.right=child2
                        push!(trees, node)
                    end
                end
            else
                node = Sbn.BinaryNode(clade_obj)
                push!(trees, node)
            end
        end

        splits = VBPILookup.generate_possible_clade_splits(clade)
        for split in splits
            other       = [c for c in clade if !(c in split)]
            clades      = Sbn.SubsplitClade(split, other)
            # node        = Sbn.BinaryNode(clades)
            if depth < max_depth
                childtrees1 = enumerate_all_sbns(clades.clade1, depth+1, max_depth)
                childtrees2 = enumerate_all_sbns(clades.clade2, depth+1, max_depth)
                for child1 in childtrees1
                    for child2 in childtrees2
                        node = Sbn.BinaryNode(clades)
                        child1.parent=node
                        child2.parent=node
                        node.left=child1
                        node.right=child2
                        push!(trees, node)
                    end
                end
            else
                node        = Sbn.BinaryNode(clades)
                push!(trees, node)
            end
        end
        return trees
    end

    function rsubsplit(clade, table)
        splits       = table.split_dict[clade]
        logit_probs  = map(split -> table[clade, split], splits)
        probs        = Flux.softmax(logit_probs)
        # if train
        #     error("Need to implement relaxation of multinomial for training")
        # else
        split_i      = wsample(probs)
        split = splits[split_i]
        other = [c for c in clade if !(c in split)]
        prob  = probs[split_i]
        return split, other, prob
    end

    function dsubsplit(clade, table, split)
        splits       = table.split_dict[clade]
        logit_probs  = map(s -> table[clade, s], splits)
        logit_i      = table[clade, split]
        return exp(logit_i - logsumexp(logit_probs))
    end

    function rsbn(clade, table, depth, max_depth)
        if length(clade) == 1
            clade1 = clade
            clade2 = missing
            lprob   = 0.0
        elseif length(clade) == 2
            clade1, clade2 = clade[1:1], clade[2:2]
            lprob = 0.0
        else
            clade1, clade2, prob = rsubsplit(clade, table)
            lprob = log(prob)
        end
        clade_obj      = Sbn.SubsplitClade(clade1, clade2)
        node        = Sbn.BinaryNode(clade_obj)

        ## Add children if not at max depth
        if depth < max_depth
            child1, lprob1 = rsbn(clade1, table, depth+1, max_depth)
            if ismissing(clade2)
                clade2 = clade1
            end
            child2, lprob2 = rsbn(clade2, table, depth+1, max_depth)
            child1.parent=node
            child2.parent=node
            node.left=child1
            node.right=child2

            lprob += lprob1 
            lprob += lprob2
        end
        return node, lprob
    end

    function dsbn(clade, node, table, depth, max_depth)
        clade1, clade2 = node.data.clade1, node.data.clade2
        if ismissing(clade2)
            return 0.0
        elseif length(clade1) == 1 && length(clade2) == 1
            return 0.0
        else
            prob_split = log(dsubsplit(clade, table, clade1))
            prob1      = dsbn(clade1, node.left, table, depth+1, max_depth)
            prob2      = dsbn(clade2, node.right, table, depth+1, max_depth)
            return prob_split + prob1 + prob2 
        end
    end

    function est_elbo(leaves, table, all_sbn_dict)
        sbn, lq = rsbn(leaves, table, 1, length(leaves)-1)
        p  = all_sbn_dict[Sbn.hashtree(sbn)]
        lp = log(p)
        lp - lq, sbn
    end


    using StatsFuns: logsumexp

    function est_iwelbo(leaves, table, all_sbn_dict, K)
        out = tmap((k) -> est_elbo(leaves, table, all_sbn_dict)[1], 1:K)
        logsumexp(out) - log(K)
    end


    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
    function lhat_j(log_fs, j)
        lse_others  = logsumexp(log_fs[i] for i in 1:length(log_fs) if i != j)
        mean_others = mean(log_fs[i] for i in 1:length(log_fs) if i != j)
        logsumexp([lse_others, mean_others])
    end

    function local_weights(log_fs, K)
        logsumexp(log_fs) .- lhat_j.([log_fs], 1:K) .- Flux.softmax(log_fs)
    end
    using Zygote
    Zygote.@nograd local_weights

    # We define this as a function so we can mark it to not be differentiated
    function vimco_local_weights_and_sbns(leaves, table, all_sbn_dict, K)
        # out = [est_elbo(leaves, table, all_sbn_dict) for _ in 1:K]
        # out = tmap((k)->est_elbo(leaves, table, all_sbn_dict), 1:K)
        out = map((k)->est_elbo(leaves, table, all_sbn_dict), 1:K)
        log_fs, sbns = unzip(out)
        lws          = local_weights(log_fs, K)
        return lws, sbns
    end
    Zygote.@nograd vimco_local_weights_and_sbns

    function vimco_objective(leaves, table, all_sbn_dict, K)
        lws, sbns = vimco_local_weights_and_sbns(leaves, table, all_sbn_dict, K)
        out = lws .* dsbn.([leaves], sbns, [table], [1], [length(leaves)-1])
        sum(out)
        # return log_fs, sbns
    end


    function gradient_with_loss(f, args...)
        y, back = Zygote.pullback(f, args...)
        return y, back(Zygote.sensitivity(y))
    end

    function train_sbn(leaves, table, all_sbn_dict; epochs=100, cb=10, calc_loss_every=10, K=50, opt=ADAM())
        starttime = Dates.now()
        flx    = params(table.probs)
        train_obj()  = -vimco_objective(leaves, table, all_sbn_dict, K)
        real_obj()   = -est_iwelbo(leaves, table, all_sbn_dict, K)
        training_losses = Vector{Float64}(undef, epochs)
        losses = Vector{Float64}(undef, div(epochs, calc_loss_every))

        for i in 1:epochs
            loss, grad = gradient_with_loss(train_obj, flx)
            Flux.update!(opt, flx, grad)
            training_losses[i] = loss
            if i % calc_loss_every == 0
                losses[div(i, calc_loss_every)]=real_obj()
            end
            if i % cb == 0
                time_seconds = (Dates.now() - starttime).value / 1000
                outstr = @sprintf("%06d: loss = % .3f (Time elapsed = %.3f sec)", i, losses[div(i, calc_loss_every)], time_seconds)
                println(outstr)
            end
        end
        timetaken = (Dates.now() - starttime).value / 1000
        println("Total time taken: ", timetaken, " sec")
        return losses
    end
end
