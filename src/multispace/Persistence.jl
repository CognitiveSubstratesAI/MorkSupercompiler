"""
Persistence — save/load spaces to/from disk.

Backed by MORK's existing space_backup_tree / space_restore_tree!.
Maps to the Hyperon Whitepaper §2.6 SMS CheckpointRef concept.

File format: MORK binary path serialization (.act compatible).
"""

"""
    save_space!(reg, name, path) → Nothing

Persist the named space to `path` using MORK's path serialization.
Updates `reg.disk_paths[id]` so subsequent `load-space` knows the path.
"""
function save_space!(reg::SpaceRegistry, name::AbstractString, path::AbstractString)
    s = get_space(reg, name)
    id = NamedSpaceID(name)
    space_backup_tree(s, path)
    reg.disk_paths[id] = String(path)
    @info "MultiSpace: saved \"$name\" → \"$path\" ($(space_val_count(s)) atoms)"
    nothing
end

"""
    load_space!(reg, name, path; create_if_missing=true) → Space

Load (or restore) a space from `path`.
If `name` is not yet registered, creates a new :app space first.

Lazy loading: only loads when explicitly called — does NOT auto-load on startup.
"""
function load_space!(
    reg::SpaceRegistry, name::AbstractString, path::AbstractString;
    create_if_missing::Bool=true
)::Space
    id = NamedSpaceID(name)
    if !haskey(reg.spaces, id)
        create_if_missing || error("space \"$name\" not registered")
        reg.spaces[id] = new_space()
        reg.roles[id] = :app
    end
    s = reg.spaces[id]
    space_restore_tree!(s, path)
    reg.disk_paths[id] = String(path)
    @info "MultiSpace: loaded \"$name\" ← \"$path\" ($(space_val_count(s)) atoms)"
    s
end

"""
    checkpoint_all!(reg, dir) → Nothing

Save all spaces to `dir/<name>.act`. Mirrors SMS CheckpointRef batch snapshot.
"""
function checkpoint_all!(reg::SpaceRegistry, dir::AbstractString)
    isdir(dir) || mkpath(dir)
    for (id, s) in reg.spaces
        path = joinpath(dir, "$(id.name).act")
        space_backup_tree(s, path)
        reg.disk_paths[id] = path
    end
    @info "MultiSpace: checkpoint → $(dir) ($(length(reg.spaces)) spaces)"
end

export save_space!, load_space!, checkpoint_all!
