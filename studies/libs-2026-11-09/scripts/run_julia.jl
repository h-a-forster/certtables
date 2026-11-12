# Evaluate Distributions.jl at the shared points. Double precision only: Distributions'
# documented default, and StatsFuns' Rmath paths are double-only.
# usage: julia --project=proj run_julia.jl points.tsv out.tsv
# Same format and resume rules as run_lib.py.
using Distributions

function parse_id(pid)
    head, rest = split(pid, "("; limit=2)
    fam, fn = split(head, ".")
    p = Dict{String,String}()
    for item in split(rest[1:end-1], ";")
        k, v = split(item, "="; limit=2)
        p[k] = v
    end
    return String(fam), String(fn), p
end

fl(p, k) = parse(Float64, p[k])
int(p, k) = parse(Int, p[k])

const FN = Dict("cdf" => (cdf, "cdf"), "sf" => (ccdf, "ccdf"),
                "ppf" => (quantile, "quantile"), "isf" => (cquantile, "cquantile"))

function api(fam, fn, p)
    if fam == "cp"
        n, k, alpha = int(p, "n"), int(p, "k"), fl(p, "alpha")
        fn == "lower" && k == 0 && return nothing
        fn == "upper" && k == n && return nothing
        fn == "lower" && return ("quantile(Beta)", () -> quantile(Beta(Float64(k), Float64(n - k + 1)), alpha / 2))
        return ("cquantile(Beta)", () -> cquantile(Beta(Float64(k + 1), Float64(n - k)), alpha / 2))
    elseif fam == "binom"
        f, name = FN[fn]
        return ("$name(Binomial)", () -> f(Binomial(int(p, "n"), fl(p, "p")), int(p, "k")))
    elseif fam == "fisher"
        a, b, c, d = parse.(Int, split(p["table"], ","))
        D = Hypergeometric(a + b, c + d, a + c)
        p["alternative"] == "less" && return ("cdf(Hypergeometric)", () -> cdf(D, a))
        p["alternative"] == "greater" && return ("ccdf(Hypergeometric)", () -> ccdf(D, a - 1))
        return nothing
    elseif fam == "ks"
        f, name = FN[fn]
        return ("$name(KSDist)", () -> f(KSDist(int(p, "n")), fl(p, "d")))
    end
    arg = fn in ("cdf", "sf") ? fl(p, "x") : fl(p, "p")
    f, name = FN[fn]
    if fam == "norm"
        return ("$name(Normal)", () -> f(Normal(0.0, 1.0), arg))
    elseif fam == "t"
        return ("$name(TDist)", () -> f(TDist(fl(p, "df")), arg))
    elseif fam == "chi2"
        return ("$name(Chisq)", () -> f(Chisq(fl(p, "df")), arg))
    elseif fam == "f"
        return ("$name(FDist)", () -> f(FDist(fl(p, "dfn"), fl(p, "dfd")), arg))
    elseif fam == "beta"
        return ("$name(Beta)", () -> f(Beta(fl(p, "a"), fl(p, "b")), arg))
    end
    return nothing
end

function main(points, out)
    println("julia ", VERSION, " Distributions ", pkgversion(Distributions))
    flush(stdout)
    done = Set{Tuple{String,String}}()
    if isfile(out)
        for ln in eachline(out)
            r = split(ln, "\t")
            length(r) == 3 && push!(done, (String(r[1]), String(r[2])))
        end
    end
    pending = out * ".pending"
    fh = open(out, "a")
    if isfile(pending)
        txt = strip(read(pending, String))
        if !isempty(txt)
            r = split(txt, "\t")
            if !((String(r[1]), String(r[2])) in done)
                println(fh, txt, "\tHANG"); flush(fh)
                push!(done, (String(r[1]), String(r[2])))
            end
        end
    end
    ids = [split(ln, "\t")[2] for ln in eachline(points)]
    t0 = time(); n = 0
    for (c, pid) in enumerate(ids)
        fam, fn, p = parse_id(pid)
        a = api(fam, fn, p)
        if a !== nothing && !((pid, a[1]) in done)
            write(pending, "$pid\t$(a[1])")
            v = try
                repr(Float64(a[2]()))
            catch e
                replace("ERR:" * first(sprint(showerror, e), 90), "\t" => " ", "\n" => " ")
            end
            println(fh, pid, "\t", a[1], "\t", v)
            flush(fh)
            n += 1
        end
        c % 500 == 0 && (println("$c/$(length(ids)) evals=$n $(round(Int, time() - t0))s"); flush(stdout))
    end
    isfile(pending) && rm(pending)
    println("finished points=$(length(ids)) evals=$n $(round(Int, time() - t0))s")
end

main(ARGS[1], ARGS[2])
