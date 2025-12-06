import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------
# Generate initial position on source surface
# ---------------------------------------
def generate_neutrons_source_from_surface_position(*, source_bounds: tuple) -> tuple:
    """
    Neutrons will be created in the surface of the material
    Args:
        - *: the label 'source_bounds' will be needed to use the argument source_bounds
            For a better reading of the code
        - source_bounds: coordinates of the corners (x-min, x-max, y-min, y-max)
            of a block (nuclear fuel/absorber)
    Returns:
        The coordinate of the leaving neutron
    """
    surface_xmin, surface_xmax, surface_ymin, surface_ymax = source_bounds
    # Choose randomly the side where the neutron is leaving from
    edge = np.random.choice(['left', 'right', 'top', 'bottom'])
    # based on the side, a random position from this side
    if edge == 'left':
        x = surface_xmin
        y = np.random.uniform(surface_ymin, surface_ymax)
    elif edge == 'right':
        x = surface_xmax
        y = np.random.uniform(surface_ymin, surface_ymax)
    elif edge == 'top':
        x = np.random.uniform(surface_xmin, surface_xmax)
        y = surface_ymax
    else:  # bottom
        x = np.random.uniform(surface_xmin, surface_xmax)
        y = surface_ymin
    return x, y

# ---------------------------------------
# Reflection of a particle at hitting a wall
# ---------------------------------------    
def reflect(velocity_x: float, velocity_y: float, normal: np.ndarray) -> tuple:
    """
    Final velocity(v_ref) calculated as the decomposition of the initial velocity
        (v) in its x- and y- components.
    Te component perpendicular to the normal is not modify and the component parallel
        to the normal flips its direction (2 * np.dot(v, n) * n)
    Finally, those components are summed to get the reflected vector
    Args:
        velocity_x / velocity_y: direction of the velocity in the x/y- component
        normal: normal vector to the surface of contact

    Returns:
        the components of the reflected vector
    """
    # normalization
    n = np.array(normal) / np.linalg.norm(normal)
    # Initial velocity vector
    v = np.array([velocity_x, velocity_y])
    # final velocity as a vector
    v_ref = v - 2 * np.dot(v, n) * n
    return v_ref[0], v_ref[1]
    
# ---------------------------------------
# Single neutron motion step
# ---------------------------------------
def neutron_step(x: float, y: float, reactor_bounds: tuple, step_size: float, 
                 step_index: int) -> tuple:
    """
    Computes a single neutron motion step inside the reactor, applying
    true elastic reflections with angle of incidence = angle of reflection.
    Args:
        x, y: coordinates of the neutron in the nuclear reactor
        reactor_bounds : coordinates of the corners of the reactor, area where 
            neutron can freely move.
        step_size : step amplitude
        step_index : current step number (for boosted first steps)
    Returns:
        New coordinates after the random motion
    """
    # Area of freely motion of the neutrons
    reactor_xmin, reactor_xmax, reactor_ymin, reactor_ymax = reactor_bounds
    # First step is 3 times (approximately) larger then the others
    multiplier = 3 if step_index < 2 else 1
    # x,y displacement
    dx = np.random.uniform(-step_size, step_size) * multiplier
    dy = np.random.uniform(-step_size, step_size) * multiplier
    
    # x/y-component of velocity as the dx/dy displacement * unit vector
    velocity_x, velocity_y = dx, dy
    
    x_new_coordinate = x + dx
    y_new_coordinate = y + dy
    
    # Left wall
    if x_new_coordinate < reactor_xmin:
        normal = np.array([1, 0])     # normal pointing right
        # reflected vector
        velocity_x, velocity_y = reflect(velocity_x, velocity_y, normal)
        x_new_coordinate = reactor_xmin + (reactor_xmin - x_new_coordinate)

    # Right wall
    elif x_new_coordinate > reactor_xmax:
        normal = np.array([-1, 0])    # normal pointing left
        # reflected vector
        velocity_x, velocity_y = reflect(velocity_x, velocity_y, normal)
        x_new_coordinate = reactor_xmax - (x_new_coordinate - reactor_xmax)

    # Bottom wall
    if y_new_coordinate < reactor_ymin:
        normal = np.array([0, 1])     # normal pointing upward
        # reflected vector
        velocity_x, velocity_y = reflect(velocity_x, velocity_y, normal)
        y_new_coordinate = reactor_ymin + (reactor_ymin - y_new_coordinate)

    # Top wall
    elif y_new_coordinate > reactor_ymax:
        normal = np.array([0, -1])    # normal pointing downward
        # reflected vector
        velocity_x, velocity_y = reflect(velocity_x, velocity_y, normal)
        y_new_coordinate = reactor_ymax - (y_new_coordinate - reactor_ymax)
        
        # Prevent neutron being out of the reactor
        x_new_coordinate = float(np.clip(x_new_coordinate, reactor_xmin, reactor_xmax))
        y_new_coordinate = float(np.clip(y_new_coordinate, reactor_ymin, reactor_ymax))

    return x_new_coordinate, y_new_coordinate

# ---------------------------------------
# Multi-neutron simulation
# ---------------------------------------
def simulate_neutrons(
    reactor_bounds: tuple, absorber_list: list, source_bounds: tuple,
    step_size: float, max_steps: int, max_neutrons=10) -> list:
    """
    Notion of upt o n-neutrons that can be absorbed or can create more neutrons
    Args:
        source_bounds: area where neutrons come from (nuclear fuel)
    Returns:
        list: _description_
    """
    # Start with one neutron, dictionary with x-,y-position, absorbed/active-state and 
    # number of the step 
    neutrons = [{"xs":[], "ys":[], "status":"active", "step_index":0}]
    # First neutron
    x0, y0 = generate_neutrons_source_from_surface_position(source_bounds=source_bounds)
    neutrons[0]["xs"].append(x0)
    neutrons[0]["ys"].append(y0)

    # Store all the positions of the neutron
    all_trajs = []

    # If there are still activated neutrons and the number of neutron are lower of the max
    while any(n["status"]=="active" for n in neutrons) and (len(all_trajs) < max_neutrons):
        new_neutrons = []
        for n in neutrons:
            if n["status"] != "active":
                continue    # neutron absorbed, pass to next neutron in list

            # Take the last step to create the next step
            x_prev_step, y_prev_step = n["xs"][-1], n["ys"][-1]
            x_new_step, y_new_step = neutron_step(x_prev_step, y_prev_step, reactor_bounds, 
                                                  step_size, n["step_index"])
            # Steps in the initial neutron
            n["xs"].append(x_new_step)
            n["ys"].append(y_new_step)
            # Increase the step counter
            n["step_index"] += 1

            # Check absorbers
            absorbed = False
            # Verify if the neutron is inside if the absorber
            for ax_min, ax_max, ay_min, ay_max in absorber_list:
                if (ax_min <= x_new_step <= ax_max) and (ay_min <= y_new_step <= ay_max):
                    n["status"] = "absorbed"
                    absorbed = True
                    break   # out of for statement
            if absorbed:
                continue    # neutron absorbed, pass to next neutron in list

            # Coordinates of the neutron source
            surface_xmin, surface_xmax, surface_ymin, surface_ymax = source_bounds
            # Check if hits source block
            if (surface_xmin <= x_new_step <= surface_xmax) and (surface_ymin <= y_new_step <= surface_ymax):
                # If the max number of neutrons is not exceed
                if len(all_trajs) + len(new_neutrons) < max_neutrons:
                    # Create a new neutron from source
                    x_s, y_s = generate_neutrons_source_from_surface_position(source_bounds=source_bounds)
                    # Store in new_neutrons array
                    new_neutrons.append({"xs":[x_s], "ys":[y_s], "status":"active", "step_index":0})

        # Elements of new neutrons are added to neutrons list, adding the elements of position, step_index
        #   for each correspondent neutron
        neutrons.extend(new_neutrons)
    # all_trajs points to neutrons list
    all_trajs = neutrons
    
    return all_trajs    #list of dictionaries

# ---------------------------------------
# Plot + animate multiple neutrons
# ---------------------------------------
def plot_multi_neutrons(all_trajs: list, reactor_bounds:tuple, absorber_list: list, 
                        source_bounds: list, grid_size=2) -> None:
    # Reactor boundaries
    reactor_xmin, reactor_xmax, reactor_ymin, reactor_ymax = reactor_bounds

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_axisbelow(True)
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.6)

    # Reactor
    reactor_patch = plt.Rectangle((reactor_xmin, reactor_ymin), reactor_xmax - reactor_xmin, 
                                  reactor_ymax - reactor_ymin, fill=False, linewidth=2)
    # Add geometric shapes
    ax.add_patch(reactor_patch)

    # Absorbers
    for absorber_xmin, absorber_xmax, absorber_ymin, absorber_ymax in absorber_list:
        absorber_patch = plt.Rectangle((absorber_xmin, absorber_ymin), absorber_xmax - absorber_xmin, 
                                       absorber_ymax - absorber_ymin, color='blue', alpha=0.4)
        ax.add_patch(absorber_patch)

    # Neutron source
    surface_xmin, surface_xmax, surface_ymin, surface_ymax = source_bounds
    # plot
    source_patch = plt.Rectangle((surface_xmin, surface_ymin), surface_xmax - surface_xmin, 
                                 surface_ymax - surface_ymin,
                                 color='red', alpha=0.3, label='Neutron Source')
    ax.add_patch(source_patch)
    # Show lines with grid_size-space
    ax.set_xticks(np.arange(reactor_xmin, reactor_xmax + grid_size, grid_size))
    ax.set_yticks(np.arange(reactor_ymin, reactor_ymax + grid_size, grid_size))

    # Lines for each neutron
    lines = []
    # number of frames based on the maximum  number of steps in all the trajectories
    max_len = max(len(traj["xs"]) for traj in all_trajs)
    
    for traj in all_trajs:
        # First element of plot and store in line variable
        line, = ax.plot([], [], color='gray', linewidth=0.8, marker='o', markersize=2)
        lines.append(line)

    def update(frame):
        for line, traj in zip(lines, all_trajs):
            # Do not exceed the trajectory size of each neutron
            end_frame = min(frame + 1, len(traj["xs"]))
            # Update the line showing all the points up to the current frame
            line.set_data(traj["xs"][:end_frame], traj["ys"][:end_frame])
        return lines

    ani = FuncAnimation(fig, update, frames=max_len, interval=80, 
                        blit=False, repeat=False)

    ax.set_xlim(reactor_xmin - 1, reactor_xmax + 1)
    ax.set_ylim(reactor_ymin - 1, reactor_ymax + 1)
    ax.set_title(f"Multi-Neutron Motion with Source-Induced Neutrons")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.show()

# ---------------------------------------
# Main
# ---------------------------------------
def main():
    reactor_bounds = (-12, 12, -12, 12)
    
    # setting the geometry of the reactor
    Absorber1 = (-8, -6, 8, 10)
    Absorber2 = (4, 6, 4, 6)
    Absorber3 = (-8, -6, 0, 2)
    Absorber4 = (-10, -8, -6, -4)
    Absorber5 = (4, 6, -4, -2)
    source_bounds = (0, 2, 0, 2)
    
    absorber_list = [Absorber1, Absorber2, Absorber3, Absorber4, Absorber5]

    all_trajs = simulate_neutrons(
        reactor_bounds=reactor_bounds,
        absorber_list=absorber_list,
        source_bounds=source_bounds,
        step_size=1.5,
        max_steps=200,
        max_neutrons=10
    )

    plot_multi_neutrons(all_trajs, reactor_bounds, absorber_list, source_bounds, grid_size=2)

if __name__ == "__main__":
    main()
