import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Function to draw the column cross-section ---
def draw_column_section(b, h, d_prime, steel_layout_str, bar_dia_mm):
    fig, ax = plt.subplots(figsize=(5, 5))
    concrete_section = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='lightgray')
    ax.add_patch(concrete_section)
    try:
        layers = [int(s.strip()) for s in steel_layout_str.split(',')]
        num_layers = len(layers)
        if num_layers > 1:
            vertical_positions = np.linspace(d_prime, h - d_prime, num_layers)
        elif num_layers == 1:
            vertical_positions = [h / 2]
        else:
            vertical_positions = []

        bar_dia_cm = bar_dia_mm / 10.0
        for i, num_bars in enumerate(layers):
            y_pos = vertical_positions[i]
            if num_bars > 1:
                horizontal_positions = np.linspace(d_prime, b - d_prime, num_bars)
            elif num_bars == 1:
                horizontal_positions = [b / 2]
            else:
                horizontal_positions = []
            for x_pos in horizontal_positions:
                bar = patches.Circle((x_pos, y_pos), radius=bar_dia_cm / 2, facecolor='darkslategray')
                ax.add_patch(bar)
    except (ValueError, IndexError):
        st.error("รูปแบบการจัดเหล็กไม่ถูกต้อง (ตัวอย่าง: '10,2,2,10')")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-b * 0.1, b * 1.1)
    ax.set_ylim(-h * 0.1, h * 1.1)
    ax.set_xlabel("Width, b (cm)")
    ax.set_ylabel("Height, h (cm)")
    plt.title("Column Cross-Section")
    plt.grid(True, linestyle='--', alpha=0.6)
    return fig

# --- Core Calculation Function (Implemented) ---
def calculate_interaction_diagram(fc, fy, b, h, d_prime, steel_layout_str, bar_dia_mm):
    # 1. Material and Geometric Properties
    Es = 2.0e6  # Modulus of Elasticity of Steel in ksc
    epsilon_c_max = 0.003  # Max concrete strain

    # Beta1 calculation based on ACI code
    if fc <= 280:
        beta1 = 0.85
    elif fc < 560:
        beta1 = 0.85 - 0.05 * (fc - 280) / 70
    else:
        beta1 = 0.65
        
    # Parse steel layout
    try:
        layers_bars = [int(s.strip()) for s in steel_layout_str.split(',')]
        num_layers = len(layers_bars)
        bar_dia_cm = bar_dia_mm / 10.0
        bar_area = np.pi * (bar_dia_cm / 2)**2
        
        steel_pos = np.linspace(d_prime, h - d_prime, num_layers) if num_layers > 1 else [h/2]
        steel_areas = [n * bar_area for n in layers_bars]
        Ast_total = sum(steel_areas)
        Ag = b * h
    except (ValueError, IndexError):
        return None, None, None, None

    # 2. Loop through Neutral Axis (c) positions to generate points
    c_values = np.linspace(0.01 * h, 2 * h, 100)
    Pn_nom, Mn_nom = [], []

    # --- Add Pure Compression Point ---
    Pn_pure_comp = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    Mn_pure_comp = 0.0
    Pn_nom.append(Pn_pure_comp)
    Mn_nom.append(Mn_pure_comp)

    # --- Loop to find other points ---
    for c in c_values:
        if c < 0.01: continue

        a = beta1 * c
        if a > h: a = h # Stress block cannot be deeper than the section

        # Concrete force
        Cc = 0.85 * fc * a * b
        
        # Steel forces
        Fs_list, steel_strains = [], []
        for i in range(num_layers):
            d_i = steel_pos[i]
            As_i = steel_areas[i]
            
            # Strain and Stress from compatibility
            epsilon_s = epsilon_c_max * (c - d_i) / c
            fs = Es * epsilon_s
            
            # Check for yielding
            if fs > fy: fs = fy
            if fs < -fy: fs = -fy
            
            steel_strains.append(epsilon_s)
            
            # Force (subtracting displaced concrete for compression steel)
            if epsilon_s > 0: # Compression
                Fs = (fs - 0.85 * fc) * As_i
            else: # Tension
                Fs = fs * As_i
            Fs_list.append(Fs)

        # Sum forces and moments
        Pn = Cc + sum(Fs_list)
        
        # Moments about the plastic centroid (h/2)
        Mc = Cc * (h / 2 - a / 2)
        Ms_list = [Fs_list[i] * (h / 2 - steel_pos[i]) for i in range(num_layers)]
        Mn = Mc + sum(Ms_list)

        Pn_nom.append(Pn)
        Mn_nom.append(Mn)

    Pn_nom = np.array(Pn_nom)
    Mn_nom = np.array(Mn_nom)

    # 3. Calculate Design Strength (phi*Pn, phi*Mn)
    Pn_design, Mn_design = [], []
    epsilon_y = fy / Es
    
    # Using the same c_values to recalculate strains for phi
    c_values_for_phi = np.insert(c_values, 0, h * 1000) # Add pure compression case
    
    for c in c_values_for_phi:
        if c < 0.01: continue
        d_t = h - d_prime # Distance to extreme tension steel
        epsilon_t = epsilon_c_max * (d_t - c) / c
        
        # Determine phi based on ACI 318
        if epsilon_t <= epsilon_y: # Compression controlled
            phi = 0.65
        elif epsilon_t >= 0.005: # Tension controlled
            phi = 0.90
        else: # Transition zone
            phi = 0.65 + 0.25 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)
        
        # Find corresponding nominal point (closest c)
        idx = (np.abs(c_values_for_phi - c)).argmin()
        Pn_design.append(Pn_nom[idx] * phi)
        Mn_design.append(Mn_nom[idx] * phi)

    # Convert units for plotting (kg to tons, kg-cm to ton-m)
    Pn_nom /= 1000
    Mn_nom /= 100000
    Pn_design /= 1000
    Mn_design /= 100000

    # Remove negative Mn values for a clean plot
    mask = Mn_nom >= 0
    return Pn_nom[mask], Mn_nom[mask], Pn_design[mask], Mn_design[mask]


# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram ของเสาคอนกรีตเสริมเหล็ก (เพื่อการศึกษาเท่านั้น)")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("กำลังอัดคอนกรีต, fc' (ksc)", min_value=1.0, value=280.0)
        fy = st.number_input("กำลังครากเหล็กเสริม, fy (ksc)", min_value=1.0, value=4000.0)
    with st.expander("ขนาดและระยะหุ้ม", expanded=True):
        b = st.number_input("ความกว้างหน้าตัด, b (cm)", min_value=1.0, value=50.0)
        h = st.number_input("ความลึกหน้าตัด, h (cm)", min_value=1.0, value=50.0)
        d_prime = st.number_input("ระยะขอบถึงศูนย์กลางเหล็กนอกสุด, d' (cm)", min_value=1.0, value=6.0)
    with st.expander("ข้อมูลเหล็กเสริม", expanded=True):
        bar_dia_mm = st.selectbox("ขนาดเหล็กเสริม", [12, 16, 20, 25, 28, 32], index=3)
        steel_layout_str = st.text_input("การจัดเรียงเหล็ก (จำนวนต่อชั้น, คั่นด้วยจุลภาค)", "5,2,5")

# --- Main App Layout ---
col1, col2 = st.columns([0.8, 1.2])

with col1:
    st.header("หน้าตัดเสา (Visualization)")
    fig_section = draw_column_section(b, h, d_prime, steel_layout_str, bar_dia_mm)
    st.pyplot(fig_section)

with col2:
    st.header("Interaction Diagram")
    if st.button("คำนวณและสร้างกราฟ", type="primary"):
        Pn_nom, Mn_nom, Pn_design, Mn_design = calculate_interaction_diagram(fc, fy, b, h, d_prime, steel_layout_str, bar_dia_mm)
        
        if Pn_nom is not None:
            fig_diagram, ax = plt.subplots(figsize=(7, 8))
            ax.plot(Mn_nom, Pn_nom, marker='.', linestyle='-', color='blue', label='Nominal Strength (Pn, Mn)')
            ax.plot(Mn_design, Pn_design, marker='.', linestyle='-', color='red', label='Design Strength (φPn, φMn)')
            
            ax.set_title("P-M Interaction Diagram")
            ax.set_xlabel("Moment, M (Ton-m)")
            ax.set_ylabel("Axial Load, P (Ton)")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.legend()
            st.pyplot(fig_diagram)
        else:
            st.error("เกิดข้อผิดพลาดในการคำนวณ ตรวจสอบข้อมูลที่ป้อน")

st.markdown("---")
st.warning("**คำเตือน:** ซอฟต์แวร์นี้ใช้เพื่อการศึกษาเท่านั้น ห้ามใช้ในการออกแบบจริงโดยไม่ผ่านการทวนสอบจากวิศวกรผู้เชี่ยวชาญ")
