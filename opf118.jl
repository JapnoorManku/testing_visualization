using PowerModels
using Ipopt
using JuMP
using Plots
using Random

Random.seed!(1234)

time_data_start = time()
case_file = "case118.m"
data = PowerModels.parse_file(case_file)
PowerModels.calc_thermal_limits!(data)
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
data_load_time = time() - time_data_start

println("Bus keys: ", keys(data["bus"]))
println("Generator keys: ", keys(data["gen"]))
println("Load keys: ", keys(data["load"]))

# Function to create a modified data copy with adjusted inputs
function adjust_inputs(data, pv_change=0.05, pq_change=0.05)
    data_copy = deepcopy(data)
    ref_copy = PowerModels.build_ref(data_copy)[:it][:pm][:nw][0]
    
    for (i, bus) in ref_copy[:bus]
        if bus["bus_type"] == 2  # PV buses
            data_copy["bus"][string(i)]["vm"] *= (1.0 + randn() * pv_change)
        end
    end
    for (g, gen) in ref_copy[:gen]
        bus = ref_copy[:bus][gen["gen_bus"]]
        if bus["bus_type"] == 2  # Generators at PV buses
            data_copy["gen"][string(g)]["pg"] *= (1.0 + randn() * pv_change)
        end
    end
    
    for (l, load) in ref_copy[:load]
        bus = ref_copy[:bus][load["load_bus"]]
        if bus["bus_type"] == 1  # PQ buses
            data_copy["load"][string(l)]["pd"] *= (1.0 + randn() * pq_change)
            data_copy["load"][string(l)]["qd"] *= (1.0 + randn() * pq_change)
        end
    end
    
    return data_copy
end

# Function to solve power flow
function solve_power_flow(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    
    model = JuMP.Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "max_iter", 1000)
    set_optimizer_attribute(model, "print_level", 5)
    
    @variable(model, va[i in keys(ref[:bus])], start=ref[:bus][i]["va"] * pi/180)
    @variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=ref[:bus][i]["vm"])
    @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"], start=ref[:gen][i]["pg"])
    @variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"], start=ref[:gen][i]["qg"])
    @variable(model, p[(l,i,j) in ref[:arcs]], start=0.0)
    @variable(model, q[(l,i,j) in ref[:arcs]], start=0.0)
    
    for (i, bus) in ref[:bus]
        if bus["bus_type"] != 1
            @constraint(model, bus["vm"] * 0.95 <= vm[i] <= bus["vm"] * 1.05)
        end
    end
    for (g, gen) in ref[:gen]
        bus = ref[:bus][gen["gen_bus"]]
        if bus["bus_type"] != 3
            @constraint(model, gen["pg"] * 0.95 <= pg[g] <= gen["pg"] * 1.05)
        end
    end
    # Reference bus angle
    for (i, bus) in ref[:ref_buses]
        @constraint(model, va[i] == 0)
    end
    
    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        
        @constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts) * vm[i]^2
        )
        
        @constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts) * vm[i]^2
        )
    end
    
    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])
        
        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]
        
        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]
        
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]
        
        @constraint(model, p_fr == (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)))
        @constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)))
        
        @constraint(model, p_to == (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)))
        @constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)))
    end
    
    @objective(model, Min, 0)
    
    optimize!(model)
    feasible = (termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED)
    
    return model, ref, feasible
end

# Test multiple scenarios
num_scenarios = 3
results = []
for s in 1:num_scenarios
    println("\n\033[1mScenario $s\033[0m")
    data_modified = adjust_inputs(data, 0.05, 0.05)
    
    time_solve_start = time()
    model, ref, feasible = solve_power_flow(data_modified)
    solve_time = time() - time_solve_start
    
    println("Input Changes:")
    for (i, bus) in ref[:bus]
        if bus["bus_type"] == 2
            println("PV Bus $i: vm = $(data_modified["bus"][string(i)]["vm"]) (original: $(data["bus"][string(i)]["vm"]))")
        end
    end
    for (g, gen) in ref[:gen]
        bus = ref[:bus][gen["gen_bus"]]
        if bus["bus_type"] == 2
            println("Gen $g at PV bus $(gen["gen_bus"]): pg = $(data_modified["gen"][string(g)]["pg"]) (original: $(data["gen"][string(g)]["pg"]))")
        end
    end
    for (l, load) in ref[:load]
        bus = ref[:bus][load["load_bus"]]
        if bus["bus_type"] == 1
            println("Load $l at PQ bus $(load["load_bus"]): pd = $(data_modified["load"][string(l)]["pd"]), qd = $(data_modified["load"][string(l)]["qd"]) (original: pd=$(data["load"][string(l)]["pd"]), qd=$(data["load"][string(l)]["qd"]))")
        end
    end
    
    if feasible
        println("\nResults for Scenario $s (Feasible):")
        println("Bus Voltages and Angles:")
        for (i, bus) in ref[:bus]
            bus_type = bus["bus_type"] == 1 ? "PQ" : bus["bus_type"] == 2 ? "PV" : "Slack"
            println("Bus $i ($bus_type): vm = $(value(model[:vm][i])), va = $(value(model[:va][i]) * 180/pi) degrees")
        end
        println("\nGenerator Outputs:")
        for (g, gen) in ref[:gen]
            qg_val = value(model[:qg][g])
            println("Gen $g at bus $(gen["gen_bus"]): pg = $(value(model[:pg][g])), qg = $qg_val, qmin=$(gen["qmin"]), qmax=$(gen["qmax"])")
            if qg_val < gen["qmin"] || qg_val > gen["qmax"]
                println("  Warning: qg outside limits for Gen $g")
            end
        end
    else
        println("\nScenario $s Infeasible")
    end
    
    push!(results, (feasible=feasible, model=model, ref=ref, solve_time=solve_time))
end

# Plot voltage magnitudes for first feasible scenario
for (s, result) in enumerate(results)
    if result.feasible
        model = result.model
        ref = result.ref
        bus_ids = collect(keys(ref[:bus]))
        vm_values = [value(model[:vm][i]) for i in bus_ids]
        p = plot(bus_ids, vm_values, label="Voltage Magnitude", title="Voltage Magnitude per Bus (Scenario $s)", xlabel="Bus ID", ylabel="Voltage (p.u.)", marker=:circle)
        display(p)
        break
    end
end