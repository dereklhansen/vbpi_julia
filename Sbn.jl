module Sbn
    using Revise
    using PhyloTrees, Plots, StatsBase
    import PhyloTrees: Tree
    include("example_binary_tree.jl")
    function clade(tree, node)
        if isleaf(tree, node)
            return [tree.node_to_label[node]]
        else
            nodes = sort!(leafnodes(tree, node))
            return [tree.node_to_label[node] for node in nodes]
        end
    end

    struct SubsplitClade
        clade1::Vector{Symbol}
        clade2::Union{Vector{Symbol}, Missing}
    end
    import Base.length
    function length(x::SubsplitClade)
        if !ismissing(x.clade2)
            return 2
        else
            return 1
        end
    end
    import Base.maximum
    function maximum(x::SubsplitClade)
        if ismissing(x.clade2)
            return maximum(x.clade1)
        else
            return max(maximum(x.clade1), maximum(x.clade2))
        end
    end

    function leaves(x::SubsplitClade)
        return sort(vcat(x.clade1, x.clade2))
    end

    import Base.hash
    function hash(x::SubsplitClade)
        hash((hash(x.clade1), hash(x.clade2)))
    end

    function make_clade_split(tree, node)
        if isleaf(tree, node)
            return SubsplitClade([tree.node_to_label[node]], missing)
        else
            child_nodes = childnodes(tree, node)
            clade1      = clade(tree, child_nodes[1])
            clade2      = clade(tree, child_nodes[2])
            if clade1[1] > clade2[1]
                clade2, clade1 = clade1, clade2
            end
            return SubsplitClade(clade1, clade2)
        end
    end

    function make_sbn(tree, node, parent, depth, max_depth)
        clades        = make_clade_split(tree, node)
        if ismissing(parent)
            subsplit_node = BinaryNode(clades)
            max_depth     = length(leafnodes(tree, node)) - 1
        else
            subsplit_node = typeof(parent)(clades, parent)
        end
        child_nodes         = childnodes(tree, node)
        if length(child_nodes) > 0 && (depth < max_depth)
                subsplit_node.left  = make_sbn(tree, child_nodes[1], subsplit_node, depth+1, max_depth)
                subsplit_node.right = make_sbn(tree, child_nodes[2], subsplit_node, depth+1, max_depth)
        elseif depth < max_depth
                subsplit_node.left  = make_sbn(tree, node, subsplit_node, depth+1, max_depth)
                subsplit_node.right = make_sbn(tree, node, subsplit_node, depth+1, max_depth)
        end
        return subsplit_node
    end

    function make_tree_from_sbn(example_sbn)
        tree = Tree()
        addnode!(tree)
        addnodes!(tree, leaves(example_sbn.data))
        # return tree
        split_clade!(tree, 1, example_sbn)
    end

    function split_clade!(tree, node, sbn)
        if !ismissing(sbn.data.clade2)
            if branch_clade!(tree, node, sbn.data.clade1)
                # addnode!(tree)
                split_clade!(tree, maximum(keys(tree.nodes)), sbn.left)
            end
            if branch_clade!(tree, node, sbn.data.clade2)
                
                split_clade!(tree, maximum(keys(tree.nodes)), sbn.right)
            end
        end
        return tree
    end

    function branch_clade!(tree, node, clade)
        if (length(clade) == 1)
            addbranch!(tree, node, tree.label_to_node[clade[1]], 1.0)
            return false
        else
            branch!(tree, node, 1.0)
            return true
        end
    end

    function annotate_tree_plot!(tree)
        tree_x, tree_y, leaf_indices, node_ids = PhyloTrees._treeplot(tree)
        for leaf in leaf_indices
            if haskey(tree.node_to_label, node_ids[leaf])
                label = string(tree.node_to_label[node_ids[leaf]])
            else
                label = string(node_ids[leaf])
            end
            annotate!(tree_x[leaf][1]+0.03, tree_y[leaf][1], label)
        end
    end

    function normalize_paths!(tree, total_len=1.0)
        leaves = leafnodes(tree, 1)
        max_branches_to_leaf = maximum(length(branchpath(tree, leaf)) for leaf in leaves)
        branch_len = total_len/max_branches_to_leaf
        for leaf in leaves
            branches = branchpath(tree, leaf)
            for (idx, branch) in enumerate(branches)
                if idx > 1
                    tree.branches[branch] = PhyloTrees.Branch(tree.branches[branch].source,
                        tree.branches[branch].target,
                        branch_len)
                else
                    tree.branches[branch] = PhyloTrees.Branch(tree.branches[branch].source,
                        tree.branches[branch].target,
                        total_len - branch_len*(length(branches) - 1))
                end
            end
        end
        return tree
    end

    function plot_and_annotate(tree, normalize_path=false)
        if normalize_path
            normalize_paths!(tree)
        end
        plot(tree)
        annotate_tree_plot!(tree)
        annotate!(0.0,0.0,"")
    end

    function plot_and_annotate(sbn::BinaryNode{SubsplitClade})
        plot_and_annotate(Sbn.make_tree_from_sbn(sbn), true)
    end
    function plot_and_annotate_sbn(sbn)
        plot_and_annotate(Sbn.make_tree_from_sbn(sbn), true)
    end
end
