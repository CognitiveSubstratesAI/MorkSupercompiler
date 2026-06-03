#!/usr/bin/env julia
# tools/launch.jl — MorkSupercompiler multi-space launcher
#
# Usage:
#   julia --project=. tools/launch.jl                         # empty registry, interactive
#   julia --project=. tools/launch.jl config.yaml            # load from YAML config
#   julia --project=. -i tools/launch.jl config.yaml         # interactive after loading
#   julia --project=. -e 'include("tools/launch.jl"); launch_multi_space()' # from script
#
# Example config.yaml:
#   spaces:
#     - name: "knowledge-base"
#       role: common
#     - name: "my-app"
#       role: app
#       load_from: "my-app.act"   # optional: load from file

try
    ;
    using Revise;
catch
    ;
end
using MorkSupercompiler, MORK

"""
    launch_multi_space(config_path=nothing) → SpaceRegistry

Enable multi-space and optionally load initial spaces from a YAML config file.
If config_path is nothing or the file doesn't exist, starts with an empty registry.

Returns the SpaceRegistry so the caller can manipulate it directly.
"""
function launch_multi_space(config_path::Union{String, Nothing}=nothing)::SpaceRegistry
    enable_multi_space!(true)
    reg = get_registry()

    if config_path !== nothing && isfile(config_path)
        _load_config!(reg, config_path)
    elseif config_path !== nothing
        @warn "Config file not found: $config_path — starting with empty registry"
    end

    if isinteractive()
        println("MorkSupercompiler multi-space ready.")
        println("  reg                            — SpaceRegistry")
        println("  new_space!(reg, \"name\", :app)   — create app space")
        println("  new_space!(reg, \"name\", :common)— create common space")
        println("  list_spaces(reg)               — show all spaces")
        println("  save_space!(reg, \"name\", \"f\")  — save to file")
        println("  load_space!(reg, \"name\", \"f\")  — load from file")
        println()
        println("Or use MM2 commands in any run!/plan!/execute! call:")
        println("  run!(s, \"(new-space \\\"my-domain\\\" :app)\")")
        println()
        if !isempty(reg.spaces)
            println("Loaded spaces:")
            for e in list_spaces(reg)
                println("  $(rpad(e.name, 24)) :$(e.role)  $(e.atoms) atoms")
            end
        else
            println("(No spaces loaded — create your own topology)")
        end
    end

    reg
end

function _load_config!(reg::SpaceRegistry, path::String)
    # Simple YAML parser — avoids YAML.jl dependency by parsing manually
    # Supports the minimal schema: spaces: [{name: "x", role: y, load_from: "f"}]
    lines = readlines(path)
    in_spaces = false
    cur_name = nothing
    cur_role = :app
    cur_load = nothing

    function _flush!()
        cur_name === nothing && return nothing
        new_space!(reg, cur_name, cur_role)
        if cur_load !== nothing && isfile(cur_load)
            load_space!(reg, cur_name, cur_load)
        end
        cur_name = nothing;
        cur_role = :app;
        cur_load = nothing
    end

    for line in lines
        stripped = strip(line)
        startswith(stripped, "#") && continue
        startswith(stripped, "spaces:") && (in_spaces=true; continue)
        !in_spaces && continue

        if startswith(stripped, "- name:")
            _flush!()
            cur_name = strip(replace(stripped, "- name:" => ""), ['"', '\'', ' '])
        elseif startswith(stripped, "name:")
            _flush!()
            cur_name = strip(replace(stripped, "name:" => ""), ['"', '\'', ' '])
        elseif startswith(stripped, "role:")
            role_str = strip(replace(stripped, "role:" => ""), ['"', '\'', ' ', ':'])
            cur_role = role_str == "common" ? :common : :app
        elseif startswith(stripped, "load_from:")
            cur_load = strip(replace(stripped, "load_from:" => ""), ['"', '\'', ' '])
        end
    end
    _flush!()
end

# ── If called directly, run the launcher ─────────────────────────────────────

if !isinteractive() && abspath(PROGRAM_FILE) == @__FILE__
    config = length(ARGS) > 0 ? ARGS[1] : nothing
    reg = launch_multi_space(config)
elseif isinteractive()
    # Called with include() in interactive session
    config = length(ARGS) > 0 ? ARGS[1] : nothing
    reg = launch_multi_space(config)
end
