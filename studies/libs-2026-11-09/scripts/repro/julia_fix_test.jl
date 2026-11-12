# Does computing the complement directly clear the t and F classes? Copies of the StatsFuns formulas only.
using SpecialFunctions, Distributions
f(s) = parse(Float64, s)
open(ARGS[2], "w") do out
    for ln in eachline(ARGS[1])
        pid = split(ln, ",")[1]
        m = match(r"^(t|f)\.(cdf|sf|ppf|isf)\(", pid); m === nothing && continue
        fam, fn = m[1], m[2]
        kv = Dict(k => f(v) for (k, v) in (split(i, "=") for i in split(pid[length(m.match)+1:end-1], ";")))
        v = NaN
        try
            if fam == "f" && fn in ("cdf", "sf")
                d1, d2, x = kv["dfn"], kv["dfd"], kv["x"]
                den = d1 * x + d2
                p, q = beta_inc(d1 / 2, d2 / 2, d1 * x / den, d2 / den)
                v = fn == "cdf" ? p : q
            elseif fam == "t" && fn in ("cdf", "sf")
                df, x = kv["df"], kv["x"]
                s = fn == "cdf" ? -x : x          # upper tail at s
                den = x^2 + df
                _, tail = beta_inc(0.5, df / 2, x^2 / den, df / den)
                v = x == 0 ? 0.5 : (s > 0 ? tail / 2 : 1 - tail / 2)
            elseif fam == "f"
                d1, d2, y = kv["dfn"], kv["dfd"], kv["p"]
                ps, qs = fn == "ppf" ? beta_inc_inv(d1 / 2, d2 / 2, y) : reverse(beta_inc_inv(d2 / 2, d1 / 2, y))
                v = ps / qs * d2 / d1
            else
                df, y = kv["df"], kv["p"]
                ps, qs = beta_inc_inv(df / 2, 0.5, 2 * min(y, 1 - y))   # y small side: x^2/(x^2+df) = qs
                mag = sqrt(qs / ps * df)
                up = fn == "isf" ? y < 0.5 : y > 0.5
                v = y == 0.5 ? 0.0 : (up ? mag : -mag)
            end
        catch e
            v = NaN
        end
        println(out, pid, "\t", repr(v))
    end
end
