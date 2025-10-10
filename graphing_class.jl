COLORS = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
]
COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
    "#F0E442",  # Yellow
    "#000000",  # Black
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
]

# Line styles
# LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", ]
LINE_STYLES = ["solid", "solid", "solid", "solid", "solid", "solid", ]

# Marker symbols
MARKER_SYMBOLS = ["circle", "square", "pentagon", "cross", "diamond", "star"]

# struct for storing data about the graph
#
# plot: Any - the plotly plot
# traces: Vector{GenericTrace} - array of traces to be plotted
# location: String - save location
# ------------------------------------------------------------
mutable struct Graph
	plot::Any
	traces::Vector{GenericTrace} # array of traces to be plotted
	location::String # save location

	# constructor for the Graph struct
	function Graph(location::String)
		new(nothing, Vector{GenericTrace}(), location)
	end
end


# a function for adding a scatter plot to the graph
# 
# graph: Graph - the graph to add the scatter plot to
# x: Array{Int} - x-axis values
# y: Array{Int} - y-axis values
# name: String - name of the trace ( optional )
# color: String - color of the trace ( optional )
# ------------------------------------------------------------
function add_scatter(graph::Graph, x::Array, y::Array, name::String="default name", style_index::Int64=1, mode::String="lines+markers")
	color = COLORS[mod1(style_index, length(COLORS))]
	line_style = LINE_STYLES[mod1(style_index, length(LINE_STYLES))]
	marker_symbol = MARKER_SYMBOLS[mod1(style_index, length(MARKER_SYMBOLS))]
	trace = PlotlyJS.scatter(x=x, y=y,
		mode=mode,
		name=name,
		marker_color=color,
		line = attr(color = color, dash = line_style, width = 2),
		marker = attr(
            color = color,
            symbol = marker_symbol,
            size = 7,
            line = attr(color = "black", width = 1)
        ),
		hoverinfo="x+y", # Ensure hover displays both x and y values
		hovertemplate="%{x}, %{y:.2f}<extra></extra>") # Custom hover text format

	push!(graph.traces, trace)
end


# a function for creating the plot with 
#
# graph: Graph - the graph to be created from the traces
# title: String - title of the plot
# x_label: String - x-axis label
# y_label: String - y-axis label
# ------------------------------------------------------------
# ***
# NOTE: This function used to be, keeping it here for records and in case we want old design
# ***
# function create_plot(graph::Graph, title::String, x_label::String, y_label::String)

# 	# conditions
# 	if (length(graph.traces) < 1) throw("cannot create plot, no traces to plot") end

# 	# create the layout
# 	layout = Layout(
# 		title=title,
# 		xaxis=attr(title=x_label, tickangle=-45, tickmode="linear", tick0=0, dtick=1),
# 		yaxis=attr(title=y_label, hoverformat=".2f"),
# 		showlegend=true
# 	)

# 	# create the plot from the traces and the layout
# 	print(graph.traces)
# 	graph.plot = PlotlyJS.plot(graph.traces, layout)
# end

# This is the new function for creating a plot
function create_plot(graph, title, x_label, y_label, Legend = (0.01, 0.98))
	legend_x = Legend[1]
	legend_y = Legend[2]
    layout = Layout(
        title = attr(
            text = title,
            font = attr(family = "Computer Modern", size = 20),
            x = 0.5,  # Center the title
            xanchor = "center",
            y = 0.875,  # Move the title down (closer to the plot)
            yanchor = "top"
        ),
        xaxis = attr(
            title = x_label,
            titlefont = attr(family = "Computer Modern", size = 18),
            tickfont = attr(family = "Computer Modern", size = 14),
			tickangle=-45,
            showgrid = true,
            gridcolor = "rgb(230, 230, 230)",
            linecolor = "black",
            linewidth = 1,
            mirror = true
        ),
        yaxis = attr(
            title = y_label,
            titlefont = attr(family = "Computer Modern", size = 18),
            tickfont = attr(family = "Computer Modern", size = 14),
            showgrid = true,
            gridcolor = "rgb(230, 230, 230)",
            linecolor = "black",
            linewidth = 1,
            mirror = true
        ),
        legend = attr(
            x = legend_x,
            y = legend_y,
            bgcolor = "rgba(255, 255, 255, 0.5)",
            bordercolor = "rgba(0, 0, 0, 0.5)",
            borderwidth = 1,
            font = attr(family = "Computer Modern", size = 12)
        ),
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        width = 800,
        height = 600,
        margin = attr(l = 80, r = 50, t = 100, b = 80),
        shapes = [
            attr(
                type = "rect",
                xref = "paper", yref = "paper",
                x0 = 0, y0 = 0, x1 = 1, y1 = 1,
                line = attr(color = "black", width = 1.5)
            )
        ]
    )

    # Update the traces for better visibility
    # for trace in graph.traces
    #     trace.line = attr(width = 2)  # Increase line thickness
    #     trace.marker = attr(size = 8)  # Increase marker size
    # end

    # create the plot from the traces and the layout
	print(graph.traces)
	graph.plot = PlotlyJS.plot(graph.traces, layout)
end

# a function for saving a plot that is currently being create_model
#
# graph: Graph - the graph to be saved
# ------------------------------------------------------------
function save_graph(graph::Graph)

	# Cretes the directory if it does not already exist
	mkpath(dirname(graph.location))
	
	print(graph.location)
	# minimum conditions to be met before saving
	if (graph.location == "") 
		throw("cannot save, location is not set") 
	end
	if (isnothing(graph.plot)) 
		throw("cannot save, plot is not set")
	end
	if (length(graph.traces) < 1)
		throw("cannot save, no traces to plot")
	end

	# save the plot in the location
	PlotlyJS.savefig(graph.plot, graph.location)
end


# a function for displaying the graph
#
# graph: Graph - the graph to be displayed
# ------------------------------------------------------------
function display_graph(graph::Graph)

	# check if the plot is set
	if (!graph.plot) throw("cannot display, plot is not set") end

	display(graph.plot)
end



#=
# test
graph = Graph("output/plot2.html")
add_scatter(graph, ["a", "b", "c", "d", "e"], [1, 2, 3, 4, 5], "trace 1", "blue")
add_scatter(graph, ["a", "b", "c", "d", "e"], [5, 4, 3, 2, 1], "trace 2", "red")
create_plot(graph, "my plot", "x-axis", "y-axis")
save_graph(graph)
=#