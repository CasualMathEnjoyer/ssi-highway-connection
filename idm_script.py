import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')

class Model:
    def __init__(self, const, seed=None):
        params = self.generate_random_highway_car(
            const['N_cars'], const, seed)

        self.const = const
        self.car_data = self.init_car_data(*params)
        self.all_left = False

    def generate_random_highway_car(self, num, const, seed=None):
        if seed is not None:
            np.random.seed(seed)

        car_generate = 0.5
        st = np.arange(0, num * car_generate, car_generate)[:num]

        x = np.zeros(num)  # all cars start at the beginning of the highway

        # Randomly assign cars to highway or connection lane
        lane = np.random.choice([True, False], num)  # True for connection lane, False for highway
        y = np.where(lane, const['connection_lane_y'], const['highway_y'])

        # Initial velocities for cars
        vx = np.random.uniform(const['v_min'], const['v_max'], num)
        vy = np.zeros(num)  # No initial vertical movement on the highway

        # Exit times and connection lane status
        ex = np.full(num, np.nan)  # Exit time is unknown initially
        con = np.full(num, np.nan)  # Time when the car connects to the highway (if it does)
        entered = np.full(num, False)  # Time when car enters the simulation
        stopped_at_lane = np.full(num, False)

        # Car type and size assignment
        car_type = np.random.choice(['personal', 'truck'], num)  # Randomly assign car types
        car_length = np.where(car_type == 'personal', const['personal_length'], const['truck_length'])

        return st, x, y, vx, vy, ex, con, lane, car_type, car_length, entered, stopped_at_lane

    def init_car_data(self, st, x, y, vx, vy, ex, con, lane, car_type, car_length, entered, stopped_at_lane):
        car_data = pd.DataFrame({
            'car_id': np.arange(len(x)),              # Car IDs
            't': [[st[i]] for i in range(len(x))],     # List of recorded times
            'x': [[x[i]] for i in range(len(x))],     # List of recorded x positions
            'y': [[y[i]] for i in range(len(y))],     # List of recorded y positions
            'vx': [[vx[i]] for i in range(len(vx))],  # List of recorded x velocities
            'vy': [[vy[i]] for i in range(len(vy))],  # List of recorded y velocities
            'time_entry': st,                         # Time when car appeared
            'time_left': ex,                          # Time when car left - or NaN
            'time_connection': con,                   # Time when car connected to highway - or NaN
            'connection_lane_car': lane,              # Boolean: If the car is on the connection lane or not
            'car_origin': lane,
            'car_type': car_type,                     # Car type (personal or truck)
            'car_length': car_length,
            'entered': [entered[i] for i in range(len(entered))],
            'stopped_at_lane': [stopped_at_lane[i] for i in range(len(stopped_at_lane))],
        })
        return car_data

    def nearest_car_ahead(self, car_idx, target_lanes=['highway']):
        """
        Find the nearest car ahead of the car, in the specified lanes.
        target_lanes: List of lanes. if includes 'highway', checks the highway lane;
                      if includes 'connection', checks the connection lane.
                      If None, only consider cars in the same lane as car_idx.
        """
        car_x = self.car_data.at[car_idx, 'x'][-1]
        car_y = self.car_data.at[car_idx, 'y'][-1]

        target_y_values = []
        if 'highway' in target_lanes:
            target_y_values.append(self.const['highway_y'])
        if 'connection' in target_lanes:
            target_y_values.append(self.const['connection_lane_y'])

        distances = []
        for i in range(len(self.car_data)):
            if i != car_idx:
                if not self.car_data.at[i, 'entered'] or not np.isnan(self.car_data.at[i, 'time_left']):
                    continue
                other_x = self.car_data.at[i, 'x'][-1]
                other_y = self.car_data.at[i, 'y'][-1]

                # Check if the car is ahead and in one of the target lanes
                if len(target_y_values) == 1:
                    if other_x >= car_x and other_y in target_y_values:
                        distance = other_x - car_x
                        distances.append((distance, i))
                else:
                    if other_x >= car_x:
                        distance = other_x - car_x
                        distances.append((distance, i))

        # Return the index of the nearest car ahead, if any, in the specified lanes
        if distances:
            return min(distances, key=lambda x: x[0])[1]
        return None

    def nearest_car_behind(self, car_idx, target_lanes=['highway']):
        """
        Find the nearest car behind the given car, in the specified target lanes.
        target_lanes: List of lanes to check. If includes 'highway', checks the highway lane;
                      if includes 'connection', checks the connection lane.
                      If None, only consider cars in the same lane as car_idx.
        """
        car_x = self.car_data.at[car_idx, 'x'][-1]
        car_y = self.car_data.at[car_idx, 'y'][-1]

        # Determine y-values for target lanes
        target_y_values = []
        if 'highway' in target_lanes:
            target_y_values.append(self.const['highway_y'])
        if 'connection' in target_lanes:
            target_y_values.append(self.const['connection_lane_y'])

        distances = []
        for i in range(len(self.car_data)):
            if i != car_idx:
                other_x = self.car_data.at[i, 'x'][-1]
                other_y = self.car_data.at[i, 'y'][-1]

                # Check if the car is behind and in one of the target lanes
                if other_x < car_x and other_y in target_y_values:
                    distance = car_x - other_x
                    distances.append((distance, i))

        # Return the index of the nearest car behind, if any, in the specified lanes
        if distances:
            return min(distances, key=lambda x: x[0])[1]
        return None  # No car behind in target lanes

    def update_velocity_idm(self, car_idx, direction='x'):
        """
        Intelligent Driver Model (IDM) for velocity update based on car ahead.
        Cars only consider their own lane or both lanes.
        """
        if direction == 'x': velocity = 'vx'
        else: velocity = 'vy'

        car = self.car_data.loc[car_idx]
        v = car[velocity][-1]

        # Parameters for IDM
        v0 = self.const['v_opt']  # Desired speed [m/s]
        T = self.const['time_headway']  # Safe time headway [s]
        a = self.const['max_acc']  # Maximum acceleration [m/s^2]
        b = self.const['max_dec']  # Maximum deceleration [m/s^2]
        s0 = self.const['min_spacing']  # Minimum distance to car ahead [m]

        car_y = self.car_data.at[car_idx, 'y'][-1]

        # Check for the nearest car ahead
        if car_y == self.const['highway_y']: # Car is on the highway; only consider cars ahead in the highway lane
            nearest_car_idx = self.nearest_car_ahead(car_idx, target_lanes=['highway'])
        elif car_y == self.const['connection_lane_y']: # Car is in the connection lane; consider cars ahead in both lanes
            nearest_car_idx = self.nearest_car_ahead(car_idx, target_lanes=['highway', 'connection'])
        elif car_y > self.const['connection_lane_y'] and car_y < self.const['highway_y']:
            nearest_car_idx = self.nearest_car_ahead(car_idx, target_lanes=['highway'])
        else:
            raise Exception('Car not in highway or connection lane')

        if nearest_car_idx is None:
            # No car ahead, just accelerate towards max speed
            acc = a * (1 - (v / v0) ** 4)
        else:
            car_ahead = self.car_data.loc[nearest_car_idx]
            delta_v = v - car_ahead[velocity][-1]  # Relative speed to car ahead
            s = car_ahead[direction][-1] - car[direction][-1]  # Distance to car ahead

            if s == 0: s = 0.05

            s_star = s0 + max(0, v * T + (v * delta_v) / (2 * math.sqrt(a * b)))
            acc = a * (1 - (v / v0) ** 4 - (s_star / s) ** 2)

        new_v = max(0, v + acc * self.const['dt'])

        return new_v

    def one_car_step(self, car_idx, act_t):
        vx = self.update_velocity_idm(car_idx, 'x')  # Update velocity based on IDM
        if (self.const['connection_lane_start'] <= self.car_data.at[car_idx, 'x'][-1] <= self.const['connection_lane_end']) or \
                self.car_data.at[car_idx, 'x'][-1] >= self.const['connection_lane_end']:
            vy = self.update_velocity_idm(car_idx, 'y')  # Update velocity based on IDM
            if self.car_data.at[car_idx, 'y'][-1] > 0:
                self.car_data.at[car_idx, 'connection_lane_car'] = False
        else:
            vy = 0

        # Check if the car is still in the connection lane and has reached the end of that lane
        if self.car_data.at[car_idx, 'connection_lane_car'] and self.car_data.at[car_idx, 'x'][-1] >= self.const['connection_lane_end']:
            vx = 0  # Prevent forward movement until the car switches
            self.car_data.at[car_idx, 'stopped_at_lane'] = True

        if vx > 0 or not self.car_data.at[car_idx, 'connection_lane_car']:  # Only allow position change if in highway or moving
            new_x = self.car_data.at[car_idx, 'x'][-1] + vx * self.const['dt']
        else:
            new_x = self.car_data.at[car_idx, 'x'][-1]  # Keep position constant if at end of connection lane and not switched

        new_y = self.car_data.at[car_idx, 'y'][-1] + vy * self.const['dt']
        if new_y > self.const["highway_y"]: new_y = self.const["highway_y"]

        self.car_data.at[car_idx, 'vx'].append(vx)
        self.car_data.at[car_idx, 'vy'].append(vy)
        self.car_data.at[car_idx, 'x'].append(new_x)
        self.car_data.at[car_idx, 'y'].append(new_y)
        self.car_data.at[car_idx, 't'].append(act_t)

    def car_exit(self, car_idx, act_t):
        car = self.car_data.loc[car_idx]
        if car['x'][-1] >= self.const['highway_length']:
            self.car_data.at[car_idx, 'time_left'] = act_t
            # print("CAR EXIT:", car_idx)

    def check_if_all_left(self):
        self.all_left = self.car_data['time_left'].notna().all()
        return self.all_left

    def main_loop(self):
        i = 0
        # print(self.car_data.head())
        while True:
            act_t = (i + 1) * self.const['dt']  # Current iteration time

            # if i % 1000 == 0: print("TIME:", act_t)

            for j in range(self.const['N_cars']):

                if not self.car_data.at[j, 'entered']:
                    if self.car_data.at[j, 't'][-1] <= act_t and np.isnan(self.car_data.at[j, 'time_left']):
                        self.car_data.at[j, 'entered'] = True

                if not self.car_data.at[j, 'entered']:
                    self.car_data.at[j, 'vx'].append(self.car_data.at[j, 'vx'][-1]) # append the same value
                    self.car_data.at[j, 'vy'].append(self.car_data.at[j, 'vy'][-1])  # append the same value
                    self.car_data.at[j, 'x'].append(self.car_data.at[j, 'x'][-1])
                    self.car_data.at[j, 'y'].append(self.car_data.at[j, 'y'][-1])
                    self.car_data.at[j, 't'].append(self.car_data.at[j, 't'][-1])
                    continue
                else:
                    if np.isnan(self.car_data.at[j, 'time_left']):
                        self.one_car_step(j, act_t)
                        self.car_exit(j, act_t)

            if self.check_if_all_left() or i > const['max_iter']:
                break

            i += 1

    def animate_model(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, self.const['highway_length'])
        ax.set_ylim(self.const['connection_lane_y'] - 1, self.const['highway_y'] + 1)
        ax.set_xlabel("Position X [m]")
        ax.set_ylabel("Position Y [m]")
        ax.set_title("Highway Car Animation")

        scat = ax.scatter([], [], s=100) # Initialize scatter plot for cars

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return scat,

        # Update function for the animation
        def update(frame):
            act_t = frame * self.const['dt']  # Calculate the current time

            # Positions and sizes to display
            x_vals, y_vals = [], []
            sizes, colors = [], []

            for i in range(self.const['N_cars']):
                # Only include cars that have appeared by this time step
                if self.car_data.at[i, 'time_entry'] <= act_t:
                    # Use the latest recorded position up to the current frame
                    x_vals.append(self.car_data.at[i, 'x'][frame] if frame < len(self.car_data.at[i, 'x']) else
                                  self.car_data.at[i, 'x'][-1])
                    y_vals.append(self.car_data.at[i, 'y'][frame] if frame < len(self.car_data.at[i, 'y']) else
                                  self.car_data.at[i, 'y'][-1])

                    # Set size and color based on car type
                    if self.car_data.at[i, 'car_type'] == 'personal':
                        sizes.append(self.const['personal_length'])
                        colors.append('blue')
                    else:
                        sizes.append(self.const['truck_length'])
                        colors.append('red')

            positions = np.column_stack((x_vals, y_vals))
            scat.set_offsets(positions)
            scat.set_sizes(sizes)
            scat.set_color(colors)
            return scat,

        # Use the length of the longest 'x' series for the frames
        max_frames = max(len(self.car_data.at[i, 'x']) for i in range(self.const['N_cars']))
        ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, repeat=False)
        plt.show()

def plot_boxplot_cars_exit(data, max_cars, num_runs, name, seed):
    tick_labels = list(range(1, max_cars+1))

    plt.figure(figsize=(12, 8))
    box = plt.boxplot(data.T,
                      labels=tick_labels,
                      patch_artist=True,  # Fill with colors
                      showmeans=True,  # Show mean values
                      meanline=True,  # Represent mean as a line
                      medianprops={'color': 'black'},  # Median line color
                      meanprops={'color': 'red', 'linewidth': 2})  # Mean line properties

    colors = plt.cm.viridis(np.linspace(0, 1, max_cars - 1))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title(f'{name} by number of cars, runs: {num_runs}', fontsize=16)
    plt.xlabel('number of cars', fontsize=14)
    plt.ylabel(f'{name}', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend for mean and median
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'num_{max_cars}_runs_{num_runs}_{name}_seed_{seed}.png')
    plt.show()

def plot_boxplot_v_opt_exit(data, num_runs, name, seed):
    """
    Plots a boxplot of exit times based on desired velocity (v_opt).
    The data should be a dictionary where each key is a v_opt value and
    each value is a sub-dictionary of simulation run results.
    """
    sorted_v_opts = sorted(data.keys())
    box_data = [list(data[v].values()) for v in sorted_v_opts]
    plt.figure(figsize=(12, 8))
    box = plt.boxplot(
        box_data,
        labels=sorted_v_opts,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={'color': 'black'},
        meanprops={'color': 'red', 'linewidth': 2}
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_v_opts)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(f'{name} by desired velocity (v_opt), runs: {num_runs}', fontsize=16)
    plt.xlabel('desired velocity (v_opt)', fontsize=14)
    plt.ylabel(name, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'v_opt_runs_{num_runs}_{name}_seed_{seed}.png')
    plt.show()
if __name__ == "__main__":
    max_cars = 1
    num_runs = 10
    base_seed = 42
    v_range = (40, 160)
    v_opts = [i for i in range(v_range[0], v_range[1], 20)]

    max_exit_times = {}
    mean_exit_times = {}

    for variable in range(v_range[0], v_range[1], 20):
        for j in tqdm(range(num_runs), desc=f"variable: {variable}"):
            const = {
                'N_cars': 5,
                'highway_length': 500,      # Length of the highway in meters
                'highway_y': 1,             # y-coordinate for highway lane
                'connection_lane_y': 0,     # y-coordinate for connection lane
                'connection_lane_start': 50,
                'connection_lane_end': 250,
                'dt': 0.1,                  # Time step for simulation [s]
                'v_min': 20,                # Minimum velocity for initialization [m/s]  # if vmin too small the cars get stuck at the entry
                'v_max': 25,                # Maximum velocity for initialization [m/s]

                'v_opt': variable,           # Desired speed (120 km/h) [m/s]
                'time_headway': 1.5,    # Safe time headway [s]
                'max_acc': 1.5,         # Maximum acceleration [m/s^2]
                'max_dec': 2.0,         # Maximum deceleration [m/s^2]
                'min_spacing': 2.0,     # Minimum spacing to the car ahead [m]

                'personal_length': 4.5,     # Length of a personal car [m]
                'truck_length': 12.0,       # Length of a truck [m]

                'max_iter': 2000
            }
            seed = base_seed + j
            model = Model(const, seed=seed)
            model.main_loop()

            # Extract and calculate max and mean exit times, ignoring NaN values
            time_left_all_cars = model.car_data['time_left']
            max_exit_time = np.nanmax(time_left_all_cars)
            mean_exit_time = np.nanmean(time_left_all_cars)

            # Store the results in the numpy arrays
            if variable not in max_exit_times:
                max_exit_times[variable] = {}

            if variable not in mean_exit_times:
                mean_exit_times[variable] = {}
            max_exit_times[variable][j] = max_exit_time
            mean_exit_times[variable][j] = mean_exit_time

            # model.animate_model()

    # Plot the results
    # plot_boxplot_cars_exit(max_exit_times, max_cars, num_runs, 'Maximum Exit Times', base_seed)
    # plot_boxplot_cars_exit(mean_exit_times, max_cars, num_runs, 'Mean Exit Times', base_seed)

    plot_boxplot_v_opt_exit(max_exit_times, num_runs, 'Maximum Exit Times', base_seed)
    plot_boxplot_v_opt_exit(mean_exit_times, num_runs, 'Mean Exit Times', base_seed)