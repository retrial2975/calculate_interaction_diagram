import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Function to draw the column cross-section ---
def draw_column_section(b, h, d_prime, steel_layout_str, bar_dia_mm):
    """
    Draws the column cross-section based on user inputs.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 1. Draw Concrete Section
    concrete_section = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='lightgray')
    ax.add_patch(concrete_section)

    # 2. Parse steel layout and draw bars
    try:
        layers = [int(s.strip()) for s in steel_layout_str.split(',')]
        num_layers = len(layers)
        
        if num_layers > 1:
            # Calculate vertical positions for each layer
            vertical_positions = np.linspace(d_prime, h - d_prime, num_layers)
        elif num_layers == 1:
            vertical_positions = [h / 2]
        else:
            vertical_positions = []

        bar_dia_cm = bar_dia_mm / 10.0

        for i, num_bars in enumerate(layers):
            y_pos = vertical_positions[i]
            if num_bars > 1:
                # Calculate horizontal positions for bars in the layer
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

    # 3. Setup plot appearance
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-b * 0.1, b * 1.1)
    ax.set_ylim(-h * 0.1, h * 1.1)
    ax.set_xlabel("Width, b (cm)")
    ax.set_ylabel("Height, h (cm)")
    plt.title("Column Cross-Section")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    return fig

# --- Core Calculation Function (Placeholder) ---
def calculate_interaction_diagram(fc, fy, b, h, d_prime, steel_layout_str, bar_dia_mm):
    """
    This is the main calculation engine.
    This function needs to be implemented with the structural mechanics logic.
    It should return lists or numpy arrays of Pn and Mn values.
    """
    # =============================================================================
    # TODO: Implement the iterative calculation logic here.
    # Loop through neutral axis 'c' depths from infinity to a small value.
    # For each 'c', calculate Pn and Mn.
    # This is the most complex part of the project.
    # =============================================================================
    
    # For demonstration, returning a dummy curve
    st.warning("ส่วนการคำนวณหลักยังเป็นเพียงโค้ดตัวอย่าง (Placeholder)")
    Pn_dummy = np.array([3000, 2500, 1500, 500, 0]) * (b*h/1000) # Dummy values
    Mn_dummy = np.array([0, 150, 250, 180, 100]) * (b*h/1000) # Dummy values
    
    return Pn_dummy, Mn_dummy

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")

st.title("🏗️ Column Interaction Diagram Generator")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram ของเสาคอนกรีตเสริมเหล็ก")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    
    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("กำลังอัดคอนกรีต, fc' (ksc)", min_value=1.0, value=600.0)
        fy = st.number_input("กำลังครากเหล็กเสริม, fy (ksc)", min_value=1.0, value=5000.0)
    
    with st.expander("ขนาดและระยะหุ้ม", expanded=True):
        b = st.number_input("ความกว้างหน้าตัด, b (cm)", min_value=1.0, value=130.0)
        h = st.number_input("ความลึกหน้าตัด, h (cm)", min_value=1.0, value=50.0)
        d_prime = st.number_input("ระยะขอบถึงศูนย์กลางเหล็กนอกสุด, d' (cm)", min_value=1.0, value=7.5)

    with st.expander("ข้อมูลเหล็กเสริม", expanded=True):
        bar_dia_mm = st.selectbox("ขนาดเหล็กเสริม", [12, 16, 20, 25, 28, 32], index=4)
        steel_layout_str = st.text_input("การจัดเรียงเหล็ก (จำนวนต่อชั้น, คั่นด้วยจุลภาค)", "24,4,4,4,24")

# --- Main App Layout ---
col1, col2 = st.columns([0.8, 1.2])

with col1:
    st.header("หน้าตัดเสา (Visualization)")
    fig_section = draw_column_section(b, h, d_prime, steel_layout_str, bar_dia_mm)
    st.pyplot(fig_section)

with col2:
    st.header("Interaction Diagram")
    if st.button("คำนวณและสร้างกราฟ", type="primary"):
        Pn_values, Mn_values = calculate_interaction_diagram(fc, fy, b, h, d_prime, steel_layout_str, bar_dia_mm)
        
        # Plotting the diagram
        fig_diagram, ax = plt.subplots(figsize=(7, 8))
        ax.plot(Mn_values, Pn_values, marker='o', linestyle='-', label='Nominal Strength (Pn, Mn)')
        ax.set_title("P-M Interaction Diagram")
        ax.set_xlabel("Moment, Mn (Ton-m)")
        ax.set_ylabel("Axial Load, Pn (Ton)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()
        
        st.pyplot(fig_diagram)

st.info("หมายเหตุ: โค้ดนี้เป็นเพียงชุดเริ่มต้น ส่วนการคำนวณหลักยังต้องถูกพัฒนาและตรวจสอบความถูกต้องทางวิศวกรรมต่อไป")
