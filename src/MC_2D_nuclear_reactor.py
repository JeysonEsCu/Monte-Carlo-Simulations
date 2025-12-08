import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------
# Helper: Liang-Barsky line-rect intersection
# (robust test whether segment (x0,y0)->(x1,y1) intersects rectangle)
# ---------------------------------------
def segment_intersects_detector(position_x0: float, position_y0: float, position_x1: float, 
                            position_y1: float, detector_coordinates: tuple) -> bool:
    """
    Liang-Barsky algorithm to test whether the segment (x0,y0)---(x1,y1)
    intersects the rectangle defined by 
    detector_coordinates = (detect_coord_xmin, detect_coord_xmax, detect_coord_ymin, detect_coord_ymax).
    Returns True if segment intersects or either endpoint is inside the rect.
    
    Defining the arrays p and q using:
    - parametric position of x in the segment position_x0 -> position_x1, for u in [0, 1]
     x(u) = position_x0 + u * dx -> x(u) = u*ypos_xposition_x1 + (1-u)*position_x0 
     For u=0: x(0) = position_x0 
     and for u=1: x(1) = position_x1
     Expressing the inequality as: p*u <= q for each edge
    e.g. constrain in left edge: 
        xmin <= x   -->     xmin <= pos_x0 + u*dx   -->     -u*dx <= pos_x0 - xmin
            (pi*u <= qi) for left edge: p1 = -dx || q1 = pos_x0 - xmin
    """
    detect_coord_xmin, detect_coord_xmax, detect_coord_ymin, detect_coord_ymax = detector_coordinates
    # Quick check: endpoint inside rect
    if (detect_coord_xmin <= position_x0 <= detect_coord_xmax and detect_coord_ymin <= position_y0 <= detect_coord_ymax) or \
       (detect_coord_xmin <= position_x1 <= detect_coord_xmax and detect_coord_ymin <= position_y1 <= detect_coord_ymax):
        return True
    
    # Step delta 
    dx = position_x1 - position_x0
    dy = position_y1 - position_y0
    # p/q array [left, right, top, bottom]
    p = [-dx, dx, -dy, dy]
    q = [position_x0 - detect_coord_xmin, detect_coord_xmax - position_x0, position_y0 - detect_coord_ymin, detect_coord_ymax - position_y0]

    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if pi == 0:          # position_x1 = position_x0 / position_y1 = position_y0
            if qi < 0:       # position_(x0/y0) outside the corresponding side, the segment will never enter
                return False # parallel and outside
            else:
                continue
        # From: x(u) = position_x0 + u * dx, u = t: where the segment reaches the edge line in question
        t = qi / pi          # from: p_i*u <= q_i --> u <= q_i / p_i = t
        if pi < 0:  # u >= t
            if t > u2:       # u >= t > u2(=1.): new lower limit above the current upper limit
                return False
            if t > u1:       # u >= t > u1(=0.): new lower limit below the current upper limit
                u1 = t
        else:   # u <= t
            if t < u1:       # u <= t < u1(=0.)
                return False
            if t < u2:       # u <= t < u2(=1.)
                u2 = t
    return u1 <= u2 # at least one u in [0,1] that places the point of the segment inside the rectangle

# ---------------------------------------
# Generate initial neutron from a source surface
# ---------------------------------------
def generate_neutron_from_source(source_bounds: tuple) -> tuple:
    """
    Generate a neutron randomly on the surface of the material
    (Hype: spawn that little particle like POW)
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
    return float(x), float(y)

# ---------------------------------------
# Reflect a neutron off a wall
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
    return float(v_ref[0]), float(v_ref[1])

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
    x_new_coordinate = x + dx
    y_new_coordinate = y + dy

    # Reflect from vertical walls
    # Left wall
    if x_new_coordinate < reactor_xmin:
        x_new_coordinate = reactor_xmin + (reactor_xmin - x_new_coordinate)
        
    # Right wall
    elif x_new_coordinate > reactor_xmax:
        x_new_coordinate = reactor_xmax - (x_new_coordinate - reactor_xmax)

    # Reflect from horizontal walls
    # Bottom wall
    if y_new_coordinate < reactor_ymin:
        y_new_coordinate = reactor_ymin + (reactor_ymin - y_new_coordinate)

    # Top wall
    elif y_new_coordinate > reactor_ymax:
        y_new_coordinate = reactor_ymax - (y_new_coordinate - reactor_ymax)
        
        # Prevent neutron being out of the reactor
        x_new_coordinate = float(np.clip(x_new_coordinate, reactor_xmin, reactor_xmax))
        y_new_coordinate = float(np.clip(y_new_coordinate, reactor_ymin, reactor_ymax))

    return x_new_coordinate, y_new_coordinate

# ---------------------------------------
# Multi-neutron simulation with detectors
# ---------------------------------------
def simulate_neutrons(
    reactor_bounds: tuple, absorber_list: list, source_list: list,
    detector_list: list = None, step_size: float = 1.5, max_steps: int = 1000,
    max_neutrons: int = 500) -> tuple:
    """
    Simulate neutrons in the reactor.

    Important semantics:
      - max_neutrons limits the maximum number of SIMULTANEOUS active neutrons,
        not the historical total spawned.
      - A neutron spawns a new neutron when its path (previous -> new position)
        intersects a neutron source. This avoids missing fast crossings.
      - An absorber immediately absorbs the neutron (disappears that frame).
    Returns:
      - neutrons: list of dicts with 'xs','ys','status','step_index','alive_until'
      - detector_counts: cumulative counts per detector
    """
    neutrons = []
    # For several neutron source, create one neutron per source initially
    for source_bounds in source_list:
        x0, y0 = generate_neutron_from_source(source_bounds)
        neutrons.append({
            "xs": [x0],
            "ys": [y0],
            "status": "active",
            "step_index": 0,
            "alive_until": None  # will be set later
        })

    # Simulate a neutron flux detector (does not distinguish between neutron
    # types: ultra cold/cold/epithermal/thermal/fast/ultra fast)
    detector_counts = [0] * len(detector_list) if detector_list else []

    # main stepping loop
    for _ in range(max_steps):
        # If there are no active neutrons, simulation can stop early
        if not any(n["status"] == "active" for n in neutrons):
            break

        new_neutrons = []
        # compute current number of active neutrons
        current_active = sum(1 for n in neutrons if n["status"] == "active")

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

            # Check absorbers -> immediate absorption. If absorbed, hide the particle
            absorbed = False
            # Verify if the neutron is inside if the absorber
            for absorber_xmin, absorber_xmax, absorber_ymin, absorber_ymax in absorber_list:
                if (absorber_xmin <= x_new_step <= absorber_xmax) and (absorber_ymin <= y_new_step <= absorber_ymax):
                    n["status"] = "absorbed"
                    # visible up to previous frame, so set alive_until = len(xs)-1 (exclusive)
                    n["alive_until"] = len(n["xs"]) - 1
                    absorbed = True
                    break   # out of for statement
            if absorbed:
                # absorbed particles do not spawn or trigger detectors this step
                continue    # neutron absorbed, pass to next neutron in list

            # Check detectors -> increment counters (counts every time a neutron is inside detector region)
            if detector_list:
                # ith_detector, (coordinates of the detector)
                for i, (detector_xmin, detector_xmax, detector_ymin, detector_ymax) in enumerate(detector_list):
                    if (detector_xmin <= x_new_step <= detector_xmax) and (detector_ymin <= x_new_step <= detector_ymax):
                        detector_counts[i] += 1

            # Check sources for multiplication (spawn new neutron at SAME source)
            for source_bounds in source_list:
                surface_xmin, surface_xmax, surface_ymin, surface_ymax = source_bounds
                if (surface_xmin <= x_new_step <= surface_xmax) and (surface_ymin <= x_new_step <= surface_ymax):

                    # Spawn only if allowed by simultaneous neutron cap
                    if len(neutrons) + len(new_neutrons) < max_neutrons:
                        # Spawn child FROM THIS SAME SOURCE
                        sx, sy = generate_neutron_from_source(source_bounds)

                        new_neutrons.append({
                            "xs": [sx],
                            "ys": [sy],
                            "status": "active",
                            "step_index": 0,
                            "alive_until": None,
                            "parent_source": source_bounds  # for debugging or plotting
                        })
                    break  # important! only spawn from ONE source
        # append newly spawned neutrons at the end of this step
        neutrons.extend(new_neutrons)

    # After simulation finishes, set alive_until for particles still not set
    for n in neutrons:
        if n["alive_until"] is None:
            # visible frames are indices 0 .. len(xs)-1 inclusive -> set alive_until = len(xs)
            n["alive_until"] = len(n["xs"])

    return neutrons, detector_counts

# ---------------------------------------
# Plot current position only
# ---------------------------------------
def plot_neutrons_current_position(all_trajs: list, reactor_bounds: tuple, absorber_list: list, 
                source_list: list, detector_list: list = None, grid_step: int = 2) -> None:
    """
    Animate only current positions of neutrons (no trails).
    10 steps == 1 second for time display, used to calculate the neutron flux.
    """
    reactor_xmin, reactor_xmax, reactor_ymin, reactor_ymax = reactor_bounds
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_axisbelow(True)
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.6)

    # Reactor frame
    reactor_patch = plt.Rectangle((reactor_xmin, reactor_ymin), reactor_xmax - reactor_xmin, 
                                  reactor_ymax - reactor_ymin, fill=False, linewidth=2)
    # Add geometric shapes
    ax.add_patch(reactor_patch)
    
    # Draw Neutron absorbers (blue)
    for absorber_xmin, absorber_xmax, absorber_ymin, absorber_ymax in absorber_list:
        ax.add_patch(plt.Rectangle((absorber_xmin, absorber_ymin), absorber_xmax - absorber_xmin, 
                                   absorber_ymax - absorber_ymin, color='blue', alpha=0.4))

    # Draw neutron sources (red)
    for sx_min, sx_max, sy_min, sy_max in source_list:
        ax.add_patch(plt.Rectangle((sx_min, sy_min), sx_max - sx_min, sy_max - sy_min, color='red', alpha=0.3))

    # Draw detectors (green) and prepare their counter texts
    detector_texts = []
    if detector_list:
        for i, (detector_xmin, detector_xmax, detector_ymin, detector_ymax) in enumerate(detector_list):
            detector_patch = plt.Rectangle((detector_xmin, detector_ymin), detector_xmax - detector_xmin, 
                                           detector_ymax - detector_ymin, color='green', alpha=0.3)
            ax.add_patch(detector_patch)
            txt = ax.text(detector_xmin, detector_ymax + 0.5, f"Detector {i+1}: 0", color='green', fontsize=10, weight='bold')
            detector_texts.append(txt)

    # Grid ticks (mimic element placement grid)
    ax.set_xticks(np.arange(reactor_xmin, reactor_xmax + grid_step, grid_step))
    ax.set_yticks(np.arange(reactor_ymin, reactor_ymax + grid_step, grid_step))

    # Create a marker for each possible trajectory (keep them fixed for simplicity)
    markers = [ax.plot([], [], 'o', color='gray', markersize=6)[0] for _ in all_trajs]

    max_len = max(len(traj["xs"]) for traj in all_trajs)

    def update(frame):
        time_sec = frame / 10.0  # 10 steps = 1 second

        # For each trajectory decide whether to show it and where
        for marker, traj in zip(markers, all_trajs):
            alive_until = traj.get("alive_until", len(traj["xs"]))
            # particle visible in frames 0 .. alive_until-1
            if frame < alive_until:
                # ensure we have coordinates for this frame index
                if frame < len(traj["xs"]):
                    x = traj["xs"][frame]
                    y = traj["ys"][frame]
                    # pass sequences to set_data to avoid Matplotlib errors
                    marker.set_data([x], [y])
                    marker.set_visible(True)
                else:
                    # not created yet -> hide
                    marker.set_data([], [])
                    marker.set_visible(False)
            else:
                # no longer exists (absorbed or beyond lifetime) -> hide
                marker.set_data([], [])
                marker.set_visible(False)

        # Update detector counters text (show cumulative counts up to this frame)
        if detector_list and detector_texts:
            for i, txt in enumerate(detector_texts):
                dx_min, dx_max, dy_min, dy_max = detector_list[i]
                count = 0
                for traj in all_trajs:
                    steps_to_check = min(frame + 1, len(traj["xs"]))
                    for x, y in zip(traj["xs"][:steps_to_check], traj["ys"][:steps_to_check]):
                        if dx_min <= x <= dx_max and dy_min <= y <= dy_max:
                            count += 1
                txt.set_text(f"Detector {i+1}: {count}")

        ax.set_title("Simplified 2D-Nuclear reactor")
        #ax.set_title(f"Neutron Positions (current only) — Time: {time_sec:.1f}s")
        return markers + detector_texts

    ani = FuncAnimation(fig, update, frames=max_len, interval=80, blit=False, repeat=False)

    ax.set_xlim(reactor_xmin - 1, reactor_xmax + 1)
    ax.set_ylim(reactor_ymin - 1, reactor_ymax + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

# ---------------------------------------
# Main function — configure and run (hype mode ON)
# ---------------------------------------
def main():
    reactor_bounds = (-12, 12, -12, 12)
    # Geometry setting
    # coordinates must be in the interval of the "reactor_bounds"
    # the coordinate must be a multiple of 2 to fit with the grid
    
    absorber1 = (-8, -6, 8, 10)
    absorber2 = (4, 6, 4, 6)
    absorber3 = (-8, -6, 0, 2)
    absorber4 = (-10, -8, -6, -4)
    absorber5 = (4, 6, -4, -2)
    
    neutron_source1 = (0, 2, 0, 2)
    neutron_source2 = (-4, -2, 6, 8)
    
    detector1 = (4, 6, 0, 2)
    detector2 = (-10, -8, 4, 6)
    
    absorber_list = [absorber1, absorber2, absorber3, absorber4, absorber5]
    source_list = [neutron_source1, neutron_source2]
    detector_list = [detector1, detector2]

    # simulate with max simultaneous neutrons = 60
    all_trajs, detector_counts = simulate_neutrons(
        reactor_bounds=reactor_bounds,
        absorber_list=absorber_list,
        source_list=source_list,
        detector_list=detector_list,
        step_size=1.5,
        max_steps=500,
        max_neutrons=60
    )

    # print final counts
    for ith_detector, counter in enumerate(detector_counts):
        print(f"Detector {ith_detector+1} total neutrons counted: {counter}")

    # animate (current positions only)
    plot_neutrons_current_position(
        all_trajs=all_trajs,
        reactor_bounds=reactor_bounds,
        absorber_list=absorber_list,
        source_list=source_list,
        detector_list=detector_list,
        grid_step=2
    )

if __name__ == "__main__":
    main()
