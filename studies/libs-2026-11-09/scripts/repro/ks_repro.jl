using Distributions
import Distributions: ceil_rems_mult, stirling
# Same as Distributions.cdf_durbin except the one marked line.
function durbin_fixed(n, x)
    k, ch, h = ceil_rems_mult(n, x)
    m = 2k - 1
    H = Matrix{Float64}(undef, m, m)
    for i in 1:m, j in 1:m
        H[i, j] = i - j + 1 >= 0 ? 1 : 0
    end
    r = 1.0
    for i in 1:m
        H[i, 1] = H[m, m - i + 1] = ch * r
        r += h^i
    end
    H[m, 1] += h <= 0.5 ? -h^m : -h^m + (2h - 1)^m   # was (h - ch), which is 2h - 1 without the power
    for i in 1:m, j in 1:m
        for g in 1:max(i - j + 1, 0)
            H[i, j] /= g
        end
        H[i, j] /= ℯ
    end
    return (H^n)[k, k] * stirling(n)
end
for line in eachline(ARGS[1])
    pid = split(line, "\t")[2]
    startswith(pid, "ks.cdf") || continue
    n = parse(Int, match(r"n=(\d+)", pid)[1]); d = parse(Float64, match(r"d=([0-9.e-]+)", pid)[1])
    b = n * d
    (b > 1 && d < 1 && b < n - 1 && b * d <= 4) || continue
    println(pid, "\t", repr(cdf(KSDist(n), d)), "\t", repr(durbin_fixed(n, d)))
end
