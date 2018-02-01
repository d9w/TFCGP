using Gadfly
using Distributions
using Colors
using DataFrames
using Query

Gadfly.push_theme(Theme(major_label_font="Droid Sans",
                        minor_label_font="Droid Sans",
                        major_label_font_size=18pt, minor_label_font_size=16pt,
                        line_width=0.8mm, key_label_font="Droid Sans",
                        lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2),
                        key_label_font_size=14pt,
                        default_color=colorant"#000000"))

colors = [colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a",
          colorant"#984ea3", colorant"#ff7f00", colorant"#ffff33"]

names = [:type, :logname, :gen, :eval, :epoch, :total_epochs, :loss, :acc, :fit]

function get_res(log::String)
    readtable(log, header=false, separator=',', names=names)
end

function reducef(df, xmax)
    r = 1:length(df[:gen])
    if df[:gen][end] != xmax
        r = 1:(length(df[:gen])+1)
    end
    map(i->mapf(i, df, xmax), r)
end

function mapf(i::Int64, df, xmax::Int64)
    if i > length(df[:gen])
        if df[:gen][end] <= xmax
            return df[:fit][end] * ones(xmax - df[:gen][end])
        else
            return []
        end
    end
    lower = 0
    if i >= 2
        lower = df[:gen][i-1]
    end
    if df[:gen][i] > lower
        return df[:fit][i] * ones(df[:gen][i] - lower)
    # else
        # return df[:fit][i] * ones(df[:gen][i])
    end
    return []
end

function get_stats(res::DataFrame; xmax::Int64 = maximum(res[:gen]))
    limited = res
    if xmax < maximum(res[:gen])
        limited = @from i in res begin
            @where i.gen <= xmax
            @select i
            @collect DataFrame
        end
    end
    filled = by(limited, [:logname, :total_epochs],
                df->reduce(vcat, reducef(df, xmax)))
    filled[:xs] = repeat(1:xmax, outer=Int64(size(filled,1)/xmax))
    stats = by(filled, [:xs, :total_epochs],
               df->DataFrame(stds=std(df[:x1]), means=mean(df[:x1]), mins=minimum(df[:x1]),
                             maxs=maximum(df[:x1])))
    stats[:stds][isnan.(stats[:stds])] = 0;
    stats[:lower] = stats[:means]-0.5*stats[:stds]
    stats[:upper] = stats[:means]+0.5*stats[:stds]
    stats[:epochs] = stats[:total_epochs]
    stats
end

function plot_evo(stats::DataFrame, title::String, filename::String="training";
                  xmin=minimum(stats[:xs]),
                  xmax=maximum(stats[:xs]),
                  ymin=minimum(stats[:lower]),
                  ymax=maximum(stats[:upper]),
                  ylabel="Accuracy",
                  key_position=:right)
    plt = plot(stats, x="xs", y="means", ymin="lower", ymax="upper", color="epochs",
               Geom.line, Geom.ribbon,
               Scale.color_discrete_manual(colors...),
               Guide.title(title),
               Guide.xlabel("Generations"),
               Guide.ylabel(ylabel),
               Coord.cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
               style(key_position=key_position))
    draw(PDF(string(filename, ".pdf"), 8inch, 6inch), plt)
    plt
end
