using Documenter
using MorkSupercompiler

DocMeta.setdocmeta!(
    MorkSupercompiler, :DocTestSetup, :(using MorkSupercompiler); recursive=true
)

makedocs(;
    modules=[MorkSupercompiler],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "MorkSupercompiler"),
    sitename="MorkSupercompiler.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/MorkSupercompiler/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=["Home" => "index.md", "Architecture" => "architecture.md"],
    # Pages link to in-repo audit records / source files outside docs/src; tolerate warnings.
    warnonly=true
)

deploydocs(; repo="github.com/CognitiveSubstratesAI/MorkSupercompiler", devbranch="main")
