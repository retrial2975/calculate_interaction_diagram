import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_column_section(b, h, d_prime, steel_layout_str, bar_dia_mm):
    """
    Draws the column cross-section based on user inputs.
    """
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
        st.error("รูปแบบการจัดเหล็กไม่ถูกต้อง (ตัวอย่าง: '5,2,5')")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-b * 0.1, b * 1.1)
    ax.set_ylim(-h * 0.1, h * 1.1)
    ax.set_xlabel("Width, b (cm)")
    ax.set_ylabel("Height, h (cm)")
    plt.title("Column Cross-Section")
    plt.grid(True, linestyle='--', alpha=0.6)
    return fig

def calculate_interaction_diagram(fc, fy, b, h, d_prime, steel_layout_str, bar_dia_mm):
    """
    Calculates the P-M interaction diagram with corrected phi calculation and plotting order.
    """
    # 1. Material and Geometric Properties
    Es = 2.0e6
    epsilon_c_max = 0.003
    epsilon_y = fy / Es

    if fc <= 280: beta1 = 0.85
    elif fc < 560: beta1 = 0.85 - 0.05 * (fc - 280) / 70
    else: beta1 = 0.65
        
    try:
        layers_bars = [int(s.strip()) for s in steel_layout_str.split(',')]
        num_layers = len(layers_bars)
        bar_dia_cm = bar_dia_mm / 10.0
        bar_area = np.pi * (bar_dia_cm / 2)**2
        steel_pos = np.linspace(d_prime, h - d_prime, num_layers) if num_layers > 1 else [h/2]
        steel_areas = [n * bar_area for n in layers_bars]
        Ast_total = sum(steel_areas)
        Ag = b * h
        d_t = h - d_prime # Distance to extreme tension steel
    except (ValueError, IndexError):
        return None, None, None, None

    # 2. Generate points
    Pn_nom_list, Mn_nom_list = [], []
    Pn_design_list, Mn_design_list = [], []

    # --- Add Pure Compression Point (c -> infinity) ---
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    Mn_pc = 0.0
    phi_pc = 0.65 # Pure compression is always compression controlled
    Pn_nom_list.append(Pn_pc)
    Mn_nom_list.append(Mn_pc)
    Pn_design_list.append(Pn_pc * phi_pc)
    Mn_design_list.append(Mn_pc * phi_pc)
    
    # --- Loop through Neutral Axis (c) ---
    c_values = np.logspace(np.log10(0.1), np.log10(h * 5), 300)

    for c in c_values:
        a = beta1 * c
        if a > h: a = h
        Cc = 0.85 * fc * a * b
        
        Fs_list = []
        for i in range(num_layers):
            d_i = steel_pos[i]
            As_i = steel_areas[i]
            epsilon_s = epsilon_c_max * (c - d_i) / c
            fs = Es * epsilon_s
            
            if fs > fy: fs = fy
            if fs < -fy: fs = -fy
            
            Fs = (fs - 0.85 * fc) * As_i if fs >= 0 else fs * As_i
            Fs_list.append(Fs)

        Pn = Cc + sum(Fs_list)
        Mc = Cc * (h / 2 - a / 2)
        Ms_list = [Fs_list[i] * (h / 2 - steel_pos[i]) for i in range(num_layers)]
        Mn = Mc + sum(Ms_list)

        if Mn >= 0:
            # CORRECT PHI CALCULATION based on strain (epsilon_t)
            epsilon_t = epsilon_c_max * (d_t - c) / c
            
            if epsilon_t <= epsilon_y: # Compression controlled
                phi = 0.65
            elif epsilon_t >= 0.005: # Tension controlled
                phi = 0.90
            else: # Transition zone
                phi = 0.65 + 0.25 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)

            Pn_nom_list.append(Pn)
            Mn_nom_list.append(Mn)
            Pn_design_list.append(Pn * phi)
            Mn_design_list.append(Mn * phi)
            
    # --- Add Pure Tension Point ---
    Pn_pt = -fy * Ast_total
    Mn_pt = 0.0
    phi_pt = 0.90 # Pure tension is always tension controlled
    Pn_nom_list.append(Pn_pt)
    Mn_nom_list.append(Mn_pt)
    Pn_design_list.append(Pn_pt * phi_pt)
    Mn_design_list.append(Mn_pt * phi_pt)

    Pn_nom = np.array(Pn_nom_list)
    Mn_nom = np.array(Mn_nom_list)
    Pn_design = np.array(Pn_design_list)
    Mn_design = np.array(Mn_design_list)

    # 3. FINAL STEP: Sort arrays by Pn descending to ensure correct plotting order
    sort_indices = np.argsort(Pn_nom)[::-1]
    Pn_nom = Pn_nom[sort_indices]
    Mn_nom = Mn_nom[sort_indices]
    Pn_design = Pn_design[sort_indices]
    Mn_design = Mn_design[sort_indices]

    # Convert units for plotting
    Pn_nom /= 1000
    Mn_nom /= 100000
    Pn_design /= 1000
    Mn_design /= 100000

    return Pn_nom, Mn_nom, Pn_design, Mn_design


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
        with st.spinner("กำลังคำนวณ..."):
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
