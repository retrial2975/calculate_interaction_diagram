import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_steel_positions(b, h, nb, nh, d_prime):
    """
    Generates steel bar positions based on the number of bars on each face.
    nb: number of bars on face parallel to b-axis (top/bottom)
    nh: number of bars on face parallel to h-axis (sides)
    Returns a list of (x, y) coordinates for each bar.
    """
    bar_positions = []
    
    # Top and bottom bars
    if nb > 0:
        x_coords_b = np.linspace(d_prime, b - d_prime, nb)
        for x in x_coords_b:
            bar_positions.append((x, d_prime)) # Bottom layer
            bar_positions.append((x, h - d_prime)) # Top layer
        
    # Side bars (excluding corners, which are already added if nb>=2)
    if nh > 2:
        y_coords_h = np.linspace(d_prime, h - d_prime, nh)[1:-1]
        for y in y_coords_h:
            bar_positions.append((d_prime, y)) # Left side
            bar_positions.append((b - d_prime, y)) # Right side
            
    # Remove duplicate coordinates and return
    return sorted(list(set(bar_positions)))

def get_layers_from_positions(steel_positions, axis):
    """
    Groups bar positions into layers based on the bending axis.
    Returns a dictionary of {layer_position: num_bars}
    """
    layers = {}
    coord_index = 1 if axis == 'X' else 0 # 1 for y-coord (X-axis bending), 0 for x-coord (Y-axis bending)
    
    for pos in steel_positions:
        layer_pos = pos[coord_index]
        if layer_pos in layers:
            layers[layer_pos] += 1
        else:
            layers[layer_pos] = 1
            
    return layers

def draw_column_section(b, h, steel_positions, bar_dia_mm):
    """
    Draws the column cross-section using generated steel positions.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    concrete_section = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='lightgray')
    ax.add_patch(concrete_section)
    
    bar_dia_cm = bar_dia_mm / 10.0
    for x, y in steel_positions:
        bar = patches.Circle((x, y), radius=bar_dia_cm / 2, facecolor='darkslategray')
        ax.add_patch(bar)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-b * 0.1, b * 1.1)
    ax.set_ylim(-h * 0.1, h * 1.1)
    ax.set_xlabel("Width, b (cm)")
    ax.set_ylabel("Height, h (cm)")
    plt.title("Column Cross-Section")
    plt.grid(True, linestyle='--', alpha=0.6)
    return fig

def calculate_interaction_diagram(fc, fy, b, h, layers, bar_area):
    """
    Main calculation function with corrected moment calculation to prevent loop-back.
    'layers' is a dictionary {position: num_bars}
    """
    Es = 2.0e6
    epsilon_c_max = 0.003
    epsilon_y = fy / Es

    if fc <= 280: beta1 = 0.85
    elif fc < 560: beta1 = 0.85 - 0.05 * (fc - 280) / 70
    else: beta1 = 0.65

    steel_pos = sorted(layers.keys())
    steel_areas = [layers[pos] * bar_area for pos in steel_pos]
    Ast_total = sum(steel_areas)
    Ag = b * h
    d_t = max(steel_pos) # Position of extreme tension steel

    Pn_nom_list, Mn_nom_list = [], []
    Pn_design_list, Mn_design_list = [], []

    # --- Add Pure Compression Point (c -> infinity) ---
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    phi_pc = 0.65 
    Pn_nom_list.append(Pn_pc)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pc * phi_pc)
    Mn_design_list.append(0.0)
    
    # --- Loop through Neutral Axis (c), extending to small c for high tension ---
    c_values = np.logspace(np.log10(0.01), np.log10(h * 5), 300)

    for c in c_values:
        a = beta1 * c
        if a > h: a = h
        
        # Concrete force and moment
        Cc = 0.85 * fc * a * b
        Mc = Cc * (h / 2.0 - a / 2.0)
        
        # Steel forces and moments
        Pn_s = 0.0 # Sum of steel forces
        Mn_s = 0.0 # Sum of steel moments

        for i, d_i in enumerate(steel_pos):
            As_i = steel_areas[i]
            epsilon_s = epsilon_c_max * (c - d_i) / c
            fs = Es * epsilon_s
            
            if fs > fy: fs = fy
            if fs < -fy: fs = -fy
            
            force = (fs - 0.85 * fc) * As_i if fs >= 0 else fs * As_i
            Pn_s += force
            Mn_s += force * (h / 2.0 - d_i)

        Pn = Cc + Pn_s
        Mn = Mc + Mn_s

        if Mn >= 0:
            epsilon_t = epsilon_c_max * (d_t - c) / c if c > 0 else float('inf')
            
            if epsilon_t <= epsilon_y: phi = 0.65
            elif epsilon_t >= 0.005: phi = 0.90
            else: phi = 0.65 + 0.25 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)

            Pn_nom_list.append(Pn)
            Mn_nom_list.append(Mn)
            Pn_design_list.append(Pn * phi)
            Mn_design_list.append(Mn * phi)
            
    # --- Add Pure Tension Point ---
    Pn_pt = -fy * Ast_total
    phi_pt = 0.90
    Pn_nom_list.append(Pn_pt)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pt * phi_pt)
    Mn_design_list.append(0.0)

    Pn_nom, Mn_nom = np.array(Pn_nom_list), np.array(Mn_nom_list)
    Pn_design, Mn_design = np.array(Pn_design_list), np.array(Mn_design_list)

    sort_indices = np.argsort(Pn_nom)[::-1]
    
    # Convert units and return sorted arrays
    return (Pn_nom[sort_indices]/1000, Mn_nom[sort_indices]/100000, 
            Pn_design[sort_indices]/1000, Mn_design[sort_indices]/100000)

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram ของเสาคอนกรีตเสริมเหล็ก (เพื่อการศึกษาเท่านั้น)")

with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    bending_axis = st.radio("เลือกแกนที่ต้องการคำนวณโมเมนต์:", ('X (Strong Axis)', 'Y (Weak Axis)'))

    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("fc' (ksc)", min_value=1.0, value=280.0)
        fy = st.number_input("fy (ksc)", min_value=1.0, value=4000.0)
    
    with st.expander("ขนาดหน้าตัด", expanded=True):
        b_in = st.number_input("ความกว้างหน้าตัด, b (cm)", min_value=1.0, value=40.0)
        h_in = st.number_input("ความลึกหน้าตัด, h (cm)", min_value=1.0, value=60.0)
    
    with st.expander("ข้อมูลเหล็กเสริม", expanded=True):
        d_prime = st.number_input("ระยะขอบถึงศูนย์กลางเหล็ก, d' (cm)", min_value=1.0, value=6.0)
        bar_dia_mm = st.selectbox("ขนาดเหล็กเสริม", [12, 16, 20, 25, 28, 32], index=3)
        st.markdown("**การจัดเรียงเหล็ก (รวมเหล็กมุม)**")
        nb = st.number_input("จำนวนเหล็กในด้านขนานแกน b (บน-ล่าง)", min_value=2, value=5)
        nh = st.number_input("จำนวนเหล็กในด้านขนานแกน h (ข้าง)", min_value=2, value=3)

# --- Main App Logic ---
steel_positions = generate_steel_positions(b_in, h_in, nb, nh, d_prime)
bar_area = np.pi * (bar_dia_mm / 10.0 / 2)**2
total_bars = len(steel_positions)
Ast_total = total_bars * bar_area
Ag_total = b_in * h_in
rho_g = (Ast_total / Ag_total)

if bending_axis == 'Y (Weak Axis)':
    calc_b, calc_h = h_in, b_in
    axis_label = "Y (Weak)"
    layers = get_layers_from_positions(steel_positions, 'Y')
else:
    calc_b, calc_h = b_in, h_in
    axis_label = "X (Strong)"
    layers = get_layers_from_positions(steel_positions, 'X')

col1, col2 = st.columns([0.8, 1.2])

with col1:
    st.header("หน้าตัดเสา (Visualization)")
    fig_section = draw_column_section(b_in, h_in, steel_positions, bar_dia_mm)
    st.pyplot(fig_section)

    # NEW: Reinforcement Summary Section
    st.markdown("---")
    st.subheader("สรุปข้อมูลเหล็กเสริม")
    st.metric(label="จำนวนเหล็กเสริมทั้งหมด", value=f"{total_bars} เส้น")
    st.metric(label="พื้นที่หน้าตัดเหล็กทั้งหมด (Ast)", value=f"{Ast_total:.2f} ตร.ซม.")
    st.metric(label="อัตราส่วนเหล็กเสริม (ρg)", value=f"{rho_g:.2%}")

with col2:
    st.header(f"Interaction Diagram (แกน {axis_label})")
    if st.button("คำนวณและสร้างกราฟ", type="primary"):
        with st.spinner("กำลังคำนวณ..."):
            Pn_nom, Mn_nom, Pn_design, Mn_design = calculate_interaction_diagram(fc, fy, calc_b, calc_h, layers, bar_area)
            
            if Pn_nom is not None:
                fig_diagram, ax = plt.subplots(figsize=(7, 8))
                ax.plot(Mn_nom, Pn_nom, marker='.', linestyle='-', color='blue', label=f'Nominal Strength')
                ax.plot(Mn_design, Pn_design, marker='.', linestyle='-', color='red', label=f'Design Strength')
                
                ax.set_title(f"P-M Interaction Diagram ({axis_label} Axis)")
                ax.set_xlabel(f"Moment, M (Ton-m)")
                ax.set_ylabel("Axial Load, P (Ton)")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                ax.legend()
                st.pyplot(fig_diagram)
            else:
                st.error("เกิดข้อผิดพลาดในการคำนวณ")

st.markdown("---")
st.warning("**คำเตือน:** ซอฟต์แวร์นี้ใช้เพื่อการศึกษาเท่านั้น ห้ามใช้ในการออกแบบจริงโดยไม่ผ่านการทวนสอบจากวิศวกรผู้เชี่ยวชาญ")
