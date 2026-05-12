import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import py3Dmol
from IPython.display import HTML, display

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

class Spectrum:
    def __init__(self, project, color=None, label=None, limits=None, ES_parameters = None, AMS_inputs=None, coeffs_bool=False):
        self.project = project
        self.root_path = ""
        self.final_path = self.root_path + project + "/"
        self.conversion_factor_hartree_to_eV = 27.2114
        self.limits = limits
        self.ES_parameters = ES_parameters
        self.coeffs_bool = coeffs_bool

        final_path_form = Path(self.final_path) 
        self.parent_path = str(final_path_form.parent) + "/"

        if color is None:
            self.color = 'C0'
        else:
            self.color = color

        if label is None:
            self.project_label = project
        else:
            self.project_label = label   

        #self.load_E_field()
        self.get_number_of_roots()
        self.load_linear_spectrum()
        self.load_CD_spectrum()
        self.get_energies_states()
        self.get_oscillator_and_rotatory_strength()
        if (AMS_inputs is None):
            self.get_transitions_by_nrs()
            self.parse_mo_block()
            self.generate_table_mo()
        if(self.coeffs_bool):
            self.load_coeffs()

    def load_linear_spectrum(self):
        data = np.array([np.genfromtxt(self.final_path + f"WaveT_{i}/sp_mol_0.dat") for i in ['x', 'y', 'z']])
        Es_lin = np.array([data[i][:,0] for i in range(3)])
        self.av_lin_energy_eV = self.conversion_factor_hartree_to_eV*(Es_lin[0] + Es_lin[1] + Es_lin[2]) / 3

        if (self.ES_parameters is None):
            if self.limits is not None:
                mask = (self.av_lin_energy_eV > self.limits[0]) & (self.av_lin_energy_eV < self.limits[1])
            else:
                mask = 1
            spectra_lin = np.array([data[i][:,1]*mask for i in range(3)])
        else:
            mask = (self.av_lin_energy_eV > self.ES_parameters["limits"][0]) & (self.av_lin_energy_eV < self.ES_parameters["limits"][1])
            spectra_lin = np.array([data[i][:,1]*mask for i in range(3)])

        self.av_lin_spectrum = (spectra_lin[0] + spectra_lin[1] + spectra_lin[2]) / 3
        self.x_lin_spectrum, self.y_lin_spectrum, self.z_lin_spectrum = spectra_lin[0], spectra_lin[1], spectra_lin[2] 

    def load_CD_spectrum(self):
        data = np.array([np.genfromtxt(self.final_path + f"WaveT_{i}/sp_mol_mag_0.dat") for i in ['x', 'y', 'z']])
        Es_CD = np.array([data[i][:,0] for i in range(3)])
        self.av_CD_energy_eV = self.conversion_factor_hartree_to_eV*(Es_CD[0] + Es_CD[1] + Es_CD[2]) / 3


        if (self.ES_parameters is None):
            if self.limits is not None:
                mask = (self.av_CD_energy_eV > self.limits[0]) & (self.av_CD_energy_eV < self.limits[1])
            else:
                mask = 1
            spectra_CD = np.array([data[i][:,1]*mask for i in range(3)])
        else:
            mask = (self.av_CD_energy_eV > self.ES_parameters["limits"][0]) & (self.av_CD_energy_eV < self.ES_parameters["limits"][1])
            spectra_CD = np.array([data[i][:,1]*mask for i in range(3)])

        self.av_CD_spectrum = (spectra_CD[0] + spectra_CD[1] + spectra_CD[2]) / 3
        self.x_CD_spectrum, self.y_CD_spectrum, self.z_CD_spectrum = spectra_CD[0], spectra_CD[1], spectra_CD[2] 

    def load_E_field(self):
        data = np.genfromtxt(self.final_path + f"WaveT_x/field0.dat")
        self.time = data[:,0]
        self.field = data[:,1]
        self.freq, self.magnitude = self.fft_pulse(self.time, self.field)

    def load_coeffs(self):

        times  = []
        isteps = []
        coeffs = []
        
        with open(self.final_path + f"WaveT_x/c_t_0.dat", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                values = line.split()
                isteps.append(int(values[0]))
                times.append(float(values[1]))
                
                # pair up Re and Im parts into complex numbers
                raw = [float(v) for v in values[2:]]
                C = [complex(raw[i], raw[i+1]) for i in range(0, len(raw), 2)]
                coeffs.append(C)
        
        self.times_coeffs = np.array(times)
        self.coeffs = np.array(coeffs)

    
    def fft_pulse(self, x, y):
        """Compute the FFT of the pulse and return frequency and magnitude.
        pulse: Array representing the pulse
        time_array: Array of time values in atomic units
        Returns: frequency array and magnitude array
        """
        dt = x[1] - x[0]
        N = len(x)
        freq = np.fft.fftfreq(N, d=dt)
        pulse_fft = np.fft.fft(y)
        pulse_fft = np.fft.fftshift(pulse_fft)
        freq = np.fft.fftshift(freq)
        freq = 2 * np.pi * freq * self.conversion_factor_hartree_to_eV
        magnitude = np.abs(pulse_fft)

        return freq, magnitude

    def get_electric_dipole_moment(self, i_state, j_state, direction):
        rows_electric_dipole_moment = []

        filename_electric_dipole_moment = self.final_path + f"WaveT_{direction}/ci_mut.inp"

        with open(filename_electric_dipole_moment) as f:
            for line in f:
                if not line.startswith("States"):
                    continue

                parts = line.split()

                i = int(parts[1])
                j = int(parts[3])

                if (i == i_state and j == j_state):
                    values = list(map(float, parts[4:]))
                    return values
        
        return None

    def get_oscillator_and_rotatory_strength(self, i_state=0, j_states=None):
        rows_oscillator = []
        rows_rotatory = []
        
        filename_oscillator = self.final_path + f"WaveT_x/ci_mut.inp"
    
        with open(filename_oscillator) as f:
            for line in f:
                if not line.startswith("States"):
                    continue
    
                parts = line.split()
                i = int(parts[1])
                j = int(parts[3])
    
                if i != i_state:
                    continue
    
                if j_states is not None and j not in j_states:
                    continue
    
                values = list(map(float, parts[4:]))
                rows_oscillator.append((j, *values))
    
        oscillator_strength = np.sum(np.array(rows_oscillator)[:, 1:]**2, axis=1)
       
        filename_rotatory = self.final_path + f"WaveT_x/ci_lt.inp"
    
        with open(filename_rotatory) as f:
            for line in f:
                if not line.startswith("States"):
                    continue
    
                parts = line.split()
                i = int(parts[1])
                j = int(parts[3])
    
                if i != i_state:
                    continue
    
                if j_states is not None and j not in j_states:
                    continue
    
                values = list(map(float, parts[4:]))
                rows_rotatory.append((j, *values))
    
        rotatory_strength = np.sum(np.array(rows_rotatory)[:, 1:]*np.array(rows_oscillator)[:, 1:], axis=1)

        self.roots = np.array(rows_rotatory)[1:self.number_of_states+1,0]

        if self.ES_parameters is not None:
            if (self.ES_parameters["region"] == "valence"):
                self.oscillatory_strength, self.rotatory_strength =  oscillator_strength[1:self.number_of_states+1], rotatory_strength[1:self.number_of_states+1]
            elif (self.ES_parameters["region"] == "core"):
                self.oscillatory_strength, self.rotatory_strength =  oscillator_strength[self.ES_parameters["n_roots_valence"]:self.ES_parameters["n_roots_valence"]+self.number_of_states], rotatory_strength[self.ES_parameters["n_roots_valence"]:self.ES_parameters["n_roots_valence"]+self.number_of_states]
            else:
                raise ValueError("Choose a correct option for the parameter: region")
        else:
            self.oscillatory_strength, self.rotatory_strength =  (2/3)*(self.root_energies/self.conversion_factor_hartree_to_eV)*oscillator_strength[1:self.number_of_states+1], rotatory_strength[1:self.number_of_states+1]

    def get_energies_states(self):
        roots = []
        energies = []

        filename = self.final_path + f"WaveT_x/ci_energy.inp"

        with open(filename) as f:
            for line in f:
                if not line.startswith("Root"):
                    continue

                parts = line.replace(":", "").split()
                root = int(parts[1])
                energy = float(parts[2])

                roots.append(root)
                energies.append(energy)

        if(self.ES_parameters is None):
            self.root_energies = np.array(energies)[:self.number_of_states]
        else:
            if (self.ES_parameters["region"] == "valence"):
                self.root_energies =  np.array(energies)[:self.number_of_states]
            elif (self.ES_parameters["region"] == "core"):
                self.root_energies =  np.array(energies)[self.ES_parameters["n_roots_valence"]:self.ES_parameters["n_roots_valence"]+self.number_of_states]
            else:
                raise ValueError("Choose a correct option for the parameter: region")
        

    def get_number_of_roots(self):

        if (self.ES_parameters is None):
            filename = self.final_path + f"WaveT_x/input_WaveT"
    
            with open(filename, "r") as f:
                first_line = f.readline()
    
            match = re.search(r"n_ci\s*=\s*(\d+)", first_line)
    
            if match:
                self.number_of_states = int(match.group(1))
            else:
                raise ValueError("n_ci not found")
        else:
            if (self.ES_parameters["region"] == "core"):
                self.number_of_states = self.ES_parameters["n_roots_core_eff"]
            elif (self.ES_parameters["region"] == "valence"):
                self.number_of_states = self.ES_parameters["n_roots_valence_eff"]
        

    def get_transitions_by_nrs(self):

        target_nrs = set(int(x) for x in self.roots)

        if (self.ES_parameters == None):
    
            filename = self.parent_path + f"output_AMS.txt" # File generated for AMS with extension .oxxxx
        
            with open(filename, "r") as f:
                text = f.read()
        
            start_key = "Major MO -> MO transitions for the above excitations"
            end_key   = "All SINGLET-SINGLET excitation energies"
        
            block = text.split(start_key, 1)[1].split(end_key, 1)[0]
        
            transitions = {}
            transitions_LaTeX = {}
        
            for line in block.splitlines():
                m = re.match(r"\s*(\d+):\s+(\S+)\s+->\s+(\S+)", line)
                if not m:
                    continue
        
                nr, occ, virt = m.groups()
                nr = int(nr)
        
                if nr in target_nrs and nr not in transitions:
                    transitions[nr] = f"{occ} -> {virt}"
                    transitions_LaTeX[nr] = rf"${occ} \, \to \, {virt}$"
        
                if transitions.keys() == target_nrs:
                    break
        
            self.transitions = transitions
            self.transitions_LaTeX = transitions_LaTeX

        else:

            n_roots_core = self.ES_parameters["n_roots_core"]
            n_roots_valence = self.ES_parameters["n_roots_valence"]

            filename_core = self.parent_path + f"output_AMS_core.txt" # File generated for AMS with extension .oxxxx
            filename_valence = self.parent_path + f"output_AMS_valence.txt"

            with open(filename_core, "r") as f:
                text_core = f.read()

            with open(filename_valence, "r") as f:
                text_valence = f.read()
        
            start_key = "Major MO -> MO transitions for the above excitations"
            end_key   = "All SINGLET-SINGLET excitation energies"
        
            block_core = text_core.split(start_key, 1)[1].split(end_key, 1)[0]
            block_valence = text_valence.split(start_key, 1)[1].split(end_key, 1)[0]
        
            transitions = {}
            transitions_LaTeX = {}

            transitions_core = {}
            transitions_LaTeX_core = {}

            transitions_valence = {}
            transitions_LaTeX_valence = {}
        
            for line in block_core.splitlines():
                m = re.match(r"\s*(\d+):\s+(\S+)\s+->\s+(\S+)", line)
                if not m:
                    continue
        
                nr, occ, virt = m.groups()
                nr = int(nr)
        
                if nr in target_nrs and nr not in transitions_core:
                    transitions_core[nr + int(n_roots_valence)] = f"{occ} -> {virt}"
                    transitions_LaTeX_core[nr + int(n_roots_valence)] = rf"${occ} \, \to \, {virt}$"
        
                if transitions_core.keys() == target_nrs:
                    break

            for line in block_valence.splitlines():
                m = re.match(r"\s*(\d+):\s+(\S+)\s+->\s+(\S+)", line)
                if not m:
                    continue
        
                nr, occ, virt = m.groups()
                nr = int(nr)
        
                if nr in target_nrs and nr not in transitions_valence:
                    transitions_valence[nr] = f"{occ} -> {virt}"
                    transitions_LaTeX_valence[nr] = rf"${occ} \, \to \, {virt}$"
        
                if transitions_valence.keys() == target_nrs:
                    break
        
            self.transitions = {**transitions_valence, **transitions_core}
            self.transitions_LaTeX = {**transitions_LaTeX_valence, **transitions_LaTeX_core}



    def parse_mo_block(self):

        if (self.ES_parameters == None):
            filename = self.parent_path + f"output_AMS.txt"
        else:
            filename = self.parent_path + f"output_AMS_core.txt"

        with open(filename) as f:
            text = f.read()
    
        start_key = "List of all MOs, ordered by energy, with the most significant SFO gross populations"
        end_key = "OCCUPIED-VIRTUAL DIPOLE MATRIX ELEMENTS"
        block = text.split(start_key, 1)[1].split(end_key, 1)[0]
    
        mo_data = {}
        current_mo = None
    
        mo_regex = re.compile(r"""
            ^\s*([-\d.]+)\s+   # Energy E(eV)
            ([\d.]+)\s+        # Occupation
            (\d+\s*[A-Z])      # MO label, ej. '1 A', '2 A'
        """, re.VERBOSE)
    
        sfo_regex = re.compile(r"""
            ^\s*([-\d.]+)%\s+  # % contribution
            (\d+\s*\S)\s+      # SFO
            ([-\d.]+)\s+       # E(eV) del fragmento
            ([\d.]+)\s+        # Occ
            (\d+\s*\S)         # Fragment
        """, re.VERBOSE)
    
        for line in block.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
    
            m = mo_regex.match(line)
            if m:
                # Nueva MO
                energy, occ, mo_label = m.groups()
                current_mo = mo_label.strip()
                mo_data[current_mo] = {
                    "energy": float(energy),
                    "occ": float(occ),
                    "sfo": []  
                }

                rest = line[m.end():].strip()
                if rest:
                    mo_data[current_mo]["sfo"].append(rest)
                continue
    
            if current_mo:
                mo_data[current_mo]["sfo"].append(line.strip())
    
        self.mo_data = mo_data

    def generate_table_mo(self):
        list_of_orbitals = []

        for root in self.roots:
            transition = self.transitions[int(root)].split(" -> ")
            out = [s[:-1] + ' ' + s[-1].upper() for s in transition]
            list_of_orbitals += out
        
        list_of_orbitals = list(dict.fromkeys(list_of_orbitals))
        list_of_orbitals = sorted(list_of_orbitals, key=lambda x: int(x.split()[0]))
        
        orbital_data = {}
        
        for orbital in list_of_orbitals:
            data = []
            if (self.mo_data.get(orbital)is not None):
                sfos = self.mo_data[orbital]["sfo"]
            for sfo in sfos:
                var = sfo.split()
                percentage, electronic_orbital, fragment = var[0], ''.join(var[1:3]), ''.join(var[-2:])
                data.append([percentage, electronic_orbital, fragment])
            orbital_data[orbital] = np.array(data)
        
        rows = []
        
        for mo, arr in orbital_data.items():
            for contrib, ao, atom in arr:
                rows.append({
                    "MO": mo,
                    "Contribution (%)": contrib.replace('%', ''),#float(contrib.replace('%', '')),
                    "AO": ao,
                    "Atom": atom
                })
        
        self.mo_table = pd.DataFrame(rows)

    def plot_molecule(self, df, coordinates):
        view = py3Dmol.view(width=400, height=400)
        view.addModel(coordinates, "xyz")
        
        view.setStyle({'stick': {"radius": 0.1}, 'sphere': {'scale': 0.3}})
        
        lines = coordinates.strip().split("\n")

        counter = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 4:
                continue  
            else:
                counter += 1
            
            elem, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            
            view.addLabel(
                f"{counter}{elem}",
                {
                    "position": {"x": x, "y": y, "z": z},
                    "fontSize": 14,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.6
                }
            )

            
        # --- Plot axis ---
        xs = [float(line.split()[1]) for line in lines if len(line.split())==4]
        ys = [float(line.split()[2]) for line in lines if len(line.split())==4]
        zs = [float(line.split()[3]) for line in lines if len(line.split())==4]
        
        center = {
            "x": min(xs)*1.5,
            "y": min(ys)*2,
            "z": min(zs)*1.5
        }
        
        axis_len = 1.5 
        
        # X
        view.addLine({'start': center, 'end': {'x': center['x']+axis_len, 'y': center['y'], 'z': center['z']}, 'color':'red'})
        view.addLabel('X', {'position': {'x': center['x']+axis_len+0.2, 'y': center['y'], 'z': center['z']}, 'fontSize':16,   'fontColor':'red', "backgroundOpacity": 0.0})
        
        # Y
        view.addLine({'start': center, 'end': {'x': center['x'], 'y': center['y']+axis_len, 'z': center['z']}, 'color':'green'})
        view.addLabel('Y', {'position': {'x': center['x'], 'y': center['y']+axis_len+0.2, 'z': center['z']}, 'fontSize':16,   'fontColor':'green', "backgroundOpacity": 0.0})
        
        # Z
        view.addLine({'start': center, 'end': {'x': center['x'], 'y': center['y'], 'z': center['z']+axis_len}, 'color':'blue'})
        view.addLabel('Z', {'position': {'x': center['x'], 'y': center['y'], 'z': center['z']+axis_len+0.2}, 'fontSize':16,    'fontColor':'blue', "backgroundOpacity": 0.0})
        

        view.zoomTo()
        #viewer_html = view._repr_html_()
        table_html = df.to_html(index=False)

        html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
            <div>{table_html}</div>
            <div>{view._make_html()}</div>
        </div>
        """

        display(HTML(html))
    
class Electric_Field_Pulse:
    def __init__(self, I, dt, pulse_type, FWHM=None, omega_sin=None, E_window = None, pulse_width=None, crossing_threshold=None):

        self.conversion_factor_hartree_to_eV = 27.2114
        self.conversion_factor_au_to_as = 24.18884326505
        self.I = I  # Intensity in W/cm^2
        self.FWHM = FWHM  # Full Width at Half Maximum in fs
        if(omega_sin is not None):
            self.omega_sin = omega_sin/self.conversion_factor_hartree_to_eV  # Frequency in a.u
        else:
            self.omega_sin = omega_sin
        if(E_window is not None):
            self.E_window = E_window/self.conversion_factor_hartree_to_eV  # Energy range in a.u
        else:
            self.E_window = E_window
        if(pulse_width is not None):
            self.pulse_width = pulse_width/self.conversion_factor_au_to_as # Apodization decay constant in a.u
        else:
            self.pulse_width = pulse_width

        self.E_max = self.convert_I_to_Emax(I)
        if(FWHM is not None):
            self.sigma = self.convert_FWHM_to_sigma(FWHM)
        else:
            self.sigma = None
        if(pulse_type != "sinc"):
            self.t_0 = np.ceil(5 * self.sigma)
        else:
            self.t_0 = np.ceil(200 * (1/self.E_window))

        self.time_span = 4000  # Total time span in atomic units
        self.dt = dt  # Time step in atomic units
        self.n_steps = int(self.time_span/ self.dt)
        self.time = np.linspace(0, self.n_steps * self.dt, self.n_steps)

        self.pulse_type = pulse_type
        self.crossing_thr = crossing_threshold

        if self.pulse_type == 'gaussian':
            self.create_gaussian_pulse(self.E_max, self.sigma, self.t_0, self.time)
            self.fft_pulse()
        elif self.pulse_type == 'gaussian_sin':
            self.create_gaussian_sin_pulse(self.E_max, self.sigma, self.omega_sin, self.t_0, self.time)
            self.fft_pulse()
        elif self.pulse_type == 'sinc':
            self.create_sinc_pulse(self.E_max, self.E_window, self.omega_sin, self.pulse_width, self.t_0, self.time)
            self.fft_pulse()
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")
        
        self.set_limits_plot()
        self.crossings()

    def convert_I_to_Emax(self,I):
        """Convert intensity I (in W/cm^2) to maximum electric field E_max (in atomic units).
        I: Intensity in W/cm^2
        Returns: E_max in atomic units
        """
        E_max = np.sqrt(I / (3.51e16))
        return E_max

    def convert_FWHM_to_sigma(self, FWHM):
        """Convert Full Width at Half Maximum (FWHM) to standard deviation (sigma).
        FWHM: Full Width at Half Maximum in as
        Returns: sigma in atomic units
        """
        FWHM_atomic_units = FWHM / self.conversion_factor_au_to_as  # Convert as to atomic units
        sigma = FWHM_atomic_units / (2 * np.sqrt(2 * np.log(2)))
        return sigma
    
    def create_gaussian_pulse(self, E_max, sigma, t_0, time_array):
        """Create a Gaussian pulse.
        E_max: Maximum electric field in atomic units
        sigma: Standard deviation in atomic units
        t_0: Center of the pulse in atomic units
        time_array: Array of time values in atomic units
        Returns: Array representing the Gaussian pulse
        """
        self.pulse = E_max * np.exp(-((time_array - t_0) ** 2) / (2 * sigma ** 2))

    def create_gaussian_sin_pulse(self, E_max, sigma, omega_c, t_0, time_array):
        """Create a Gaussian pulse.
        E_max: Maximum electric field in atomic units
        sigma: Standard deviation in atomic units
        omega_c: Central frequency in atomic units
        t_0: Center of the pulse in atomic units
        time_array: Array of time values in atomic units
        Returns: Array representing the Gaussian pulse
        """  
        self.pulse = E_max * np.exp(-((time_array - t_0) ** 2) / (2 * sigma ** 2))*np.sin(omega_c * (time_array))

    def create_sinc_pulse(self, E_max, T ,omega_c, pulse_width, t_0, time_array):
        """Create a Gaussian pulse.
        E_max: Maximum electric field in atomic units
        T: Window width
        omega_c: Central frequency in atomic units
        t_0: Center of the pulse in atomic units
        time_array: Array of time values in atomic units
        Returns: Array representing the Gaussian pulse
        """  
        tau_ap = pulse_width/2

        w_exp = np.exp(-(1/(tau_ap))*np.abs(time_array-t_0))
        
        def w_tukey(t, alpha):
            w = np.zeros_like(self.time)
            mask = (self.time >= t_0 - tau_ap) & (self.time <= t_0 + tau_ap)
            t_mask = self.time[~mask]
            w[~mask] = 0.5*( 1 + np.cos( (np.pi*np.abs(t_mask - t_0) - (1-alpha)*tau_ap)/(alpha*tau_ap) ) )  
            w[mask] = 1
            return w

        self.pulse = E_max * (np.sin(T*(time_array - t_0)/2)/(T*(time_array - t_0)/2)) * np.sin(omega_c*time_array)*w_exp
        

    def fft_pulse(self):
        """Compute the FFT of the pulse and return frequency and magnitude.
        pulse: Array representing the pulse
        time_array: Array of time values in atomic units
        Returns: frequency array and magnitude array
        """
        dt = self.time[1] - self.time[0]
        N = len(self.time)
        freq = np.fft.fftfreq(N, d=dt)
        pulse_fft = np.fft.fft(self.pulse)
        pulse_fft = np.fft.fftshift(pulse_fft)
        freq = np.fft.fftshift(freq)
        self.freq = 2 * np.pi * freq * self.conversion_factor_hartree_to_eV
        self.magnitude = np.abs(pulse_fft)

    def estimate_central_frequency(self, x, y):
        """Estimate the central frequency of the pulse from its FFT.
        freq: Frequency array
        magnitude: Magnitude array
        Returns: Estimated central frequency in eV
        """
        peaks, _ = find_peaks(y, height=0)
        if len(peaks) == 0:
            raise ValueError("No peaks found.")
        peak_magnitudes = y[peaks]
        max_peak_index = peaks[np.argmax(peak_magnitudes)]
        central_frequency = x[max_peak_index]
        return central_frequency
    

    def crossings(self):
        if (self.crossing_thr is None):
            thr = 2E-1
        else:
            thr = self.crossing_thr
        threshold = thr * self.magnitude.max()
        y0 = self.magnitude - threshold
        idx = np.where(np.diff(np.sign(y0)) != 0)[0]

        # interpolación lineal
        x_cross = self.freq[idx] - y0[idx] * (self.freq[idx+1] - self.freq[idx]) / (y0[idx+1] - y0[idx])
        if(self.pulse_type == 'gaussian'):
            self.crossings_E = (0, x_cross[-1])
        elif(self.pulse_type == 'gaussian_sin'):
            self.crossings_E = (x_cross[-2], x_cross[-1])
        elif(self.pulse_type == 'sinc'):
            self.crossings_E = (self.omega_sin*self.conversion_factor_hartree_to_eV - self.E_window*self.conversion_factor_hartree_to_eV/2, self.omega_sin*self.conversion_factor_hartree_to_eV + self.E_window*self.conversion_factor_hartree_to_eV/2)
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin', or 'sinc'.")

    def set_limits_plot(self):

        central_time = self.t_0

        if self.pulse_type == 'gaussian':
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(self.freq, self.magnitude)
            freq_window = 2 * self.fwhm(self.freq, self.magnitude)
        elif self.pulse_type == 'gaussian_sin':
            time_window = 5 * self.sigma
            self.time_limits = (central_time - time_window, central_time + time_window)
            central_freq = self.estimate_central_frequency(self.freq[self.freq > 0], self.magnitude[self.freq > 0])
            freq_window = 2 * self.fwhm(self.freq[self.freq > 0], self.magnitude[self.freq > 0])
        elif self.pulse_type == 'sinc':
            time_window = 100 * (1/self.E_window)
            if(central_time - time_window > 0):
                self.time_limits = (central_time - time_window, central_time + time_window)
            else:
                self.time_limits = (0, central_time + time_window)
            central_freq = self.omega_sin*self.conversion_factor_hartree_to_eV
            freq_window = 2 * self.E_window*self.conversion_factor_hartree_to_eV
        else:
            raise ValueError("Invalid pulse type. Choose 'gaussian', 'gaussian_sin' or 'sinc'.")
        if (central_freq - freq_window) < 0:
            self.freq_limits = (0, central_freq + freq_window)
        else:
            self.freq_limits = (central_freq - freq_window, central_freq + freq_window)

    def fwhm(self, x, y):
        """
        Compute the full width at half maximum (FWHM) of y(x).
        Assumes x is sorted and y has a single main peak.
        """
        y = np.asarray(y)
        x = np.asarray(x)
    
        ymax = y.max()
        half_max = ymax / 2.0
    
        # Indices where signal crosses half max
        above = y >= half_max
    
        if not np.any(above):
            return np.nan
    
        idx = np.where(above)[0]
    
        # Left crossing
        i1 = idx[0] - 1
        i2 = idx[0]
    
        # Right crossing
        i3 = idx[-1]
        i4 = idx[-1] + 1
    
        # Linear interpolation
        x_left = np.interp(half_max, [y[i1], y[i2]], [x[i1], x[i2]])
        x_right = np.interp(half_max, [y[i3], y[i4]], [x[i3], x[i4]])
    
        return x_right - x_left
    
    def __str__(self):
        return (f"Electric Field Pulse:\n"
                f"  Intensity (I): {self.I} W/cm^2\n"
                f"  FWHM: {self.FWHM} as\n"
                f"  Central frequency: {self.omega_sin} a.u.\n"
                f"  E_max: {self.E_max} a.u.\n"
                f"  Sigma: {self.sigma} a.u.\n"
                f"  Time step (dt): {self.dt} a.u.\n"
                f"  Number of steps: {self.n_steps}\n"
                f"  Pulse type: {self.pulse_type}\n"
                f"  Time at maximum: {self.t_0} a.u.\n"
                f"  Energy window: {self.E_window} a.u.\n"
                f"  Pulse width: {self.pulse_width} a.u.\n"
                f"  ")