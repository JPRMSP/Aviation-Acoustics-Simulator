# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter
import soundfile as sf
import io
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Aviation Acoustics Simulator (Advanced)", layout="wide")
st.title("✈️ Aviation Acoustics Simulator — Advanced (Units I–V)")

# -------------------------
# Helper: basic signal generators
# -------------------------
def tone(frequency, t, phase=0, amp=1.0):
    return amp * np.sin(2 * np.pi * frequency * t + phase)

def monopole(f0, t, amp=1.0):
    return amp * np.sin(2 * np.pi * f0 * t)

def dipole(f0, t, amp=1.0):
    return amp * np.sin(2 * np.pi * f0 * t) * np.cos(2 * np.pi * 0.5 * f0 * t)

def quadrupole(f0, t, amp=1.0):
    return amp * (np.sin(2 * np.pi * f0 * t) ** 2) * np.cos(2 * np.pi * f0 * t)

def broadband_jet(t, scale=1.0):
    # filtered noise with 1/f-ish spectrum
    n = np.random.normal(0, 1, len(t))
    # simple lowpass via cumulative moving average to give broadband feel
    kernel = np.exp(-np.linspace(0, 5, 101))
    kernel /= kernel.sum()
    from scipy.signal import fftconvolve
    return scale * fftconvolve(n, kernel, mode='same')

def propeller_signature(t, rpm=2000, blades=3, amp=1.0):
    # fundamental frequency ~ rpm/60 * blades (blade passing freq)
    f = (rpm / 60.0) * blades
    sig = np.zeros_like(t)
    for h in range(1, 6):
        sig += (1.0 / h) * np.sin(2 * np.pi * f * h * t)
    return amp * sig

def concorde_boom(t, mach):
    # crude N-wave generator scaled by Mach
    # create a sharp positive jump then gradual negative (approx)
    tau = 0.002 / max(0.2, mach)  # shorter for higher Mach
    n = np.zeros_like(t)
    center = t[len(t)//2]
    n += np.where(t < center, (t - center + tau) / tau, (center - t + tau) / tau)
    return n * (1.0 / mach) * 5.0

# -------------------------
# Atmospheric attenuation (simplified but tunable)
# -------------------------
def absorption_coefficient(freq_hz, T_C=15.0, RH=50.0, P_kPa=101.325):
    """
    Approximate frequency-dependent absorption coefficient (dB/m).
    This is a simplified empirical approximation (not a full ISO standard implementation).
    """
    f = np.asarray(freq_hz)
    # relax freq proxy, depends on T and humidity; simplified:
    T = T_C + 273.15
    h = RH / 100.0
    # baseline small-coef that grows with f^2 / (f^2 + f0^2)
    f0 = 1000.0 * (1.0 + (T_C - 15.0) * 0.01)  # move with temperature
    alpha = 1e-8 * (f ** 2) / (f ** 2 + f0 ** 2)  # small baseline
    # humidity increases absorption around some bands - scale factor
    alpha *= (1.0 + 2.0 * h)
    # convert to dB/m scale factor
    return alpha * 1e2  # scale up to sensible dB/m values for demo

def apply_atmospheric_attenuation(signal, fs, distance_m, T_C=15.0, RH=50.0):
    # apply a frequency-dependent attenuation in frequency domain
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    S = np.fft.rfft(signal)
    alpha_db_per_m = absorption_coefficient(freqs, T_C, RH)  # dB/m
    # total dB
    total_db = alpha_db_per_m * distance_m
    gain = 10 ** (-total_db / 20.0)
    S *= gain
    atten_signal = np.fft.irfft(S, n)
    # also apply geometric spreading (1/r) amplitude decrease
    atten_signal *= (1.0 / (1.0 + 0.1 * distance_m))
    return atten_signal

# -------------------------
# Doppler & motion model (straight line pass)
# -------------------------
def simulate_moving_source(signal_func, base_freq, fs, duration, path_params):
    """
    path_params: dict containing
      - speed_m_s
      - altitude_m
      - closest_approach_m  (y offset)
      - direction: +x direction
      - observer at origin (0,0,0)
    Returns time vector and received signal after Doppler + delay + attenuation (no atmospheric)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # source x(t) moves along x axis, observer at origin (0,0)
    v = path_params.get('speed_m_s', 200.0)
    alt = path_params.get('altitude_m', 1000.0)
    y0 = path_params.get('closest_approach_m', 500.0)
    x0 = -v * (duration / 2.0)
    x_t = x0 + v * t
    y_t = np.ones_like(t) * y0
    z_t = np.ones_like(t) * alt
    # distances and radial velocities
    r_t = np.sqrt(x_t**2 + y_t**2 + z_t**2)
    # radial velocity = projection of source velocity onto line-of-sight unit vector
    vx = v
    los_x = -x_t / r_t
    los_y = -y_t / r_t
    los_z = -z_t / r_t
    v_radial = vx * los_x  # since vy,vz zero
    c = 343.0
    # instantaneous doppler factor
    doppler_factor = c / (c - v_radial)
    # generate base (source) signal in source time: we'll resample via instantaneous phase integration
    # Build instantaneous phase: integral of 2pi * f * doppler_factor(t) dt
    inst_freq = base_freq * doppler_factor
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    # Compose signal by calling provided generator or tone
    src = signal_func(base_freq, t)
    # Replace src with phase-driven sine for tonal sources for Doppler correctness
    tonal_modes = ["tone", "monopole", "dipole", "quadrupole", "propeller"]
    try:
        # if generator is tonal, use phase-based sine; else modulate broadband by doppler amplitude factor
        if signal_func.__name__ in ('tone', 'monopole', 'dipole', 'quadrupole', 'propeller_signature'):
            received = np.sin(phase)
        else:
            # for broadband jet / noise: compress/stretch by Doppler by changing sample indices
            # a simple approach: time-warp: new_t = np.cumsum(1.0 / doppler_factor) / fs
            # resample src onto t with simple interpolation
            src_t = t
            warped_t = np.cumsum(1.0 / doppler_factor) / fs
            # normalize warped_t to duration length
            warped_t = (warped_t - warped_t[0])
            if warped_t[-1] == 0:
                warped_t = t
            else:
                warped_t = warped_t * (t[-1] / warped_t[-1])
            received = np.interp(t, warped_t, src)
    except Exception:
        received = src
    # add propagation delay (simple) by shifting signal by int(delay*fs)
    delay_samples = (r_t / c).astype(int)
    recv_shifted = np.zeros_like(received)
    for i, d in enumerate(delay_samples):
        if i + d < len(received):
            recv_shifted[i + d] += received[i]
    # scale by 1/r
    recv_shifted *= (1.0 / (1.0 + r_t))
    return t, recv_shifted, (x_t, y_t, z_t), r_t

# -------------------------
# Simple A-weighting (approx)
# -------------------------
def a_weighting(signal, fs):
    # Implement simple 2nd-order Butterworth band adjustments approximating A-weighting
    # For demo: apply a bandpass and frequency-dependent gain using filtering pipeline
    # We'll use a single bandpass 20-16000 Hz (practical) to mimic human weighting roughly
    b, a = butter(4, [20 / (fs/2), 16000 / (fs/2)], btype='band')
    return lfilter(b, a, signal)

# -------------------------
# UI Controls
# -------------------------
st.sidebar.header("Source & Flight Parameters")
source_option = st.sidebar.selectbox("Aircraft / Source Type", [
    "Monopole", "Dipole", "Quadrupole", "Propeller (piston)", "Turbofan (engine)", "Turbojet (exhaust)",
    "Jet Broadband", "Airframe (broadband)", "Concorde Sonic Boom"
])

base_freq = st.sidebar.slider("Base Tone Frequency (Hz) — for tonal sources", 50, 2000, 400)
duration = st.sidebar.slider("Simulation Duration (s)", 0.5, 5.0, 2.0, step=0.5)
fs = st.sidebar.selectbox("Sampling Rate (Hz)", [22050, 44100], index=1)

speed = st.sidebar.slider("Aircraft Speed (m/s)", 50, 900, 200)  # up to supersonic ~340 m/s
altitude = st.sidebar.slider("Altitude (m)", 0, 15000, 1000)
closest_approach = st.sidebar.slider("Closest Approach (m)", 10, 2000, 500)
rpm = st.sidebar.slider("Prop RPM (for propeller)", 500, 4000, 1800)
blades = st.sidebar.slider("Propeller Blades", 2, 8, 3)

st.sidebar.header("Atmosphere")
temp_C = st.sidebar.slider("Temperature (°C)", -40, 40, 15)
humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 40)

st.sidebar.header("Visualization")
show_3d = st.sidebar.checkbox("Show 3D Trajectory & Mach Cone", value=True)
show_contours = st.sidebar.checkbox("Show SPL Contour Map (angle vs distance)", value=True)
play_audio = st.sidebar.checkbox("Provide Audio Playback", value=True)
show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)

# -------------------------
# Choose generator
# -------------------------
def pick_generator(opt):
    if opt == "Monopole":
        return lambda f, t: monopole(f, t, amp=1.0)
    if opt == "Dipole":
        return lambda f, t: dipole(f, t, amp=1.0)
    if opt == "Quadrupole":
        return lambda f, t: quadrupole(f, t, amp=1.0)
    if opt == "Propeller (piston)":
        return lambda f, t: propeller_signature(t, rpm=rpm, blades=blades, amp=1.0)
    if opt == "Turbofan (engine)":
        # mix tonal fan tone + broadband
        return lambda f, t: 0.5 * tone(f, t) + 0.8 * broadband_jet(t, scale=0.6)
    if opt == "Turbojet (exhaust)":
        return lambda f, t: 0.3 * tone(f*1.5, t) + 1.2 * broadband_jet(t, scale=1.0)
    if opt == "Jet Broadband":
        return lambda f, t: 1.0 * broadband_jet(t, scale=1.2)
    if opt == "Airframe (broadband)":
        return lambda f, t: 0.6 * broadband_jet(t, scale=0.6) + 0.3 * np.random.normal(0, 1, len(t))
    if opt == "Concorde Sonic Boom":
        return lambda f, t: concorde_boom(t, max(1.2, speed / 343.0))
    # default
    return lambda f, t: monopole(f, t, amp=1.0)

generator = pick_generator(source_option)

# -------------------------
# Simulate moving source & Doppler
# -------------------------
path_params = {
    'speed_m_s': speed,
    'altitude_m': altitude,
    'closest_approach_m': closest_approach
}
t, received, traj_xyz, r_t = simulate_moving_source(generator, base_freq, fs, duration, path_params)

# Apply atmospheric attenuation using frequency-domain method
received_atm = apply_atmospheric_attenuation(received, fs, np.mean(r_t), T_C=temp_C, RH=humidity)

# Add simple ground reflection: create an image source mirrored across ground (z->-z) with coefficient
def add_ground_reflection(received, traj_xyz, fs, reflection_coeff=0.6):
    x_t, y_t, z_t = traj_xyz
    # image distances to observer at origin: z mirrored
    r_image = np.sqrt(x_t**2 + y_t**2 + ((-z_t) ** 2))
    delay_samples = (r_image / 343.0).astype(int)
    refl = np.zeros_like(received)
    for i, d in enumerate(delay_samples):
        if i + d < len(received):
            refl[i + d] += received[i] * reflection_coeff
    return received + refl

received_total = add_ground_reflection(received_atm, traj_xyz, fs, reflection_coeff=0.4)

# A-weight (approx) for human perception
received_a = a_weighting(received_total, fs)

# Compute SPL
def compute_spl(signal):
    p_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    spl = 20.0 * np.log10(p_rms / 20e-6 + 1e-12)
    return spl

spl_lin = compute_spl(received_total)
spl_a = compute_spl(received_a)

# -------------------------
# Left column: plots & audio
# -------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Time-domain Signal (portion)")
    fig, ax = plt.subplots(figsize=(10, 2.4))
    samples_to_show = min(len(t), int(fs * 0.03))  # show first 30ms for clarity
    ax.plot(t[:samples_to_show], received_total[:samples_to_show])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (arb.)")
    st.pyplot(fig)

    if show_spectrogram:
        st.subheader("Spectrogram (received)")
        f_spec, ti, Sxx = spectrogram(received_total, fs=fs, nperseg=1024, noverlap=512)
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.pcolormesh(ti, f_spec, 10 * np.log10(Sxx + 1e-12), shading='auto')
        ax2.set_ylim(0, 8000)
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        st.pyplot(fig2)

    st.markdown("### SPL metrics")
    st.metric("SPL (unweighted)", f"{spl_lin:.2f} dB")
    st.metric("SPL (A-weighted approx.)", f"{spl_a:.2f} dB")

    if play_audio:
        # prepare audio and allow playback
        # normalize to -1..1 range
        sig = received_total
        sig = sig / (np.max(np.abs(sig)) + 1e-12)
        # convert to 32-bit float wav into buffer
        buf = io.BytesIO()
        sf.write(buf, sig.astype(np.float32), fs, format='WAV')
        buf.seek(0)
        st.audio(buf.read(), format='audio/wav')

# -------------------------
# Right column: controls summary & 3D viz
# -------------------------
with right_col:
    st.subheader("Simulation Summary")
    st.write(f"Source: **{source_option}**")
    st.write(f"Speed: **{speed:.1f} m/s**  (Mach ≈ {speed/343.0:.2f})")
    st.write(f"Altitude: **{altitude:.1f} m**")
    st.write(f"Closest approach: **{closest_approach:.1f} m**")
    st.write(f"Atmosphere: T={temp_C}°C, RH={humidity}%")
    st.write(f"Duration: {duration:.2f} s — Sampling: {fs} Hz")

    if show_3d:
        st.subheader("3D Trajectory & Mach Cone (interactive)")
        x_t, y_t, z_t = traj_xyz
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', name='Aircraft Path',
                                     line=dict(width=4)))
        # observer at origin
        fig3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='Observer',
                                     marker=dict(size=5, color='red')))
        # Mach cone visualization (if supersonic)
        mach = speed / 343.0
        if mach > 1.0:
            # draw a cone at mid-position
            mid_idx = len(x_t)//2
            xc, yc, zc = x_t[mid_idx], y_t[mid_idx], z_t[mid_idx]
            theta = np.arcsin(1.0 / mach)
            # cone mesh
            u = np.linspace(0, 2*np.pi, 30)
            h = np.linspace(0.0, 800.0, 10)  # length of cone
            UU, HH = np.meshgrid(u, h)
            Xc = xc + HH * np.cos(UU) * np.tan(theta)
            Yc = yc + HH * np.sin(UU) * np.tan(theta)
            Zc = zc - HH  # extend downwards
            fig3d.add_trace(go.Surface(x=Xc, y=Yc, z=Zc, opacity=0.4, showscale=False, name='Mach Cone'))
        else:
            st.info("Mach < 1.0 — no Mach cone (subsonic)")

        fig3d.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
                            height=500)
        st.plotly_chart(fig3d, use_container_width=True)

# -------------------------
# SPL Contour Map: angle vs distance
# -------------------------
if show_contours:
    st.subheader("SPL Contour Map — angle vs. distance (approx)")
    # coarse grid for performance
    angles = np.linspace(0, 180, 37)  # degrees
    distances = np.concatenate((np.linspace(10, 200, 20), np.linspace(300, 2000, 10)))
    SPL_map = np.zeros((len(distances), len(angles)))
    # source directivity pattern simple model: directional gain ~ cos(theta)^n for some n per source
    directivity_orders = {
        "Monopole": 0,
        "Dipole": 2,
        "Quadrupole": 4,
        "Propeller (piston)": 6,
        "Turbofan (engine)": 2,
        "Turbojet (exhaust)": 1,
        "Jet Broadband": 0,
        "Airframe (broadband)": 2,
        "Concorde Sonic Boom": 0
    }
    n_dir = directivity_orders.get(source_option, 0)

    # compute an approximate source level (arbitrary but consistent) using computed SPL
    src_SL = spl_lin + 20.0  # offset so contours are in realistic range

    for i, d in enumerate(distances):
        for j, ang in enumerate(angles):
            # directivity
            theta_r = np.radians(ang)
            directivity_gain = np.cos(theta_r) ** n_dir if n_dir > 0 else 1.0
            directivity_gain = max(0.0, directivity_gain)
            # spherical spreading: -20*log10(d)
            spread = -20.0 * np.log10(d + 1e-6)
            # atmospheric absorption approximate: use alpha at 1000 Hz * d (dB)
            alpha_db = absorption_coefficient(np.array([1000.0]), temp_C, humidity)[0] * d
            # overall SPL approx
            SPL_map[i, j] = src_SL + 20.0 * np.log10(directivity_gain + 1e-6) + spread - alpha_db

    # smoothing for nicer appearance
    SPL_map_smooth = gaussian_filter(SPL_map, sigma=1.0)

    figc, axc = plt.subplots(figsize=(8, 4.5))
    X, Y = np.meshgrid(angles, distances)
    cs = axc.contourf(X, Y, SPL_map_smooth, levels=30)
    axc.set_xlabel("Observer Angle (deg)")
    axc.set_ylabel("Distance (m)")
    axc.set_title("Approximate SPL Contours (dB)")
    plt.colorbar(cs, ax=axc, label='dB')
    st.pyplot(figc)

# -------------------------
# Export / Save / GitHub-ready info
# -------------------------
st.markdown("---")
st.markdown("**Notes & Methodology (brief):**")
st.write("""
- All signals are **synthetic** and computed using analytic / procedural models (no datasets or pretrained models).
- Doppler is applied via instantaneous radial velocity time-warp/phase method for tonal sources; broadband sources are time-warped approximately.
- Atmospheric absorption uses a simplified frequency-dependent attenuation curve to demonstrate temperature/humidity effects.
- Ground reflection is modeled with a single image-source reflection (coarse).
- Mach cone rendered when Mach > 1.0 (geometric approximation).
- Contour map is an approximate, low-cost simulation for visualization and teaching.
""")
