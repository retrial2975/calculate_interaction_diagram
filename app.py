import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ฟังก์ชันคำนวณต่างๆ (ไม่มีการเปลี่ยนแปลง) ---
def generate_steel_positions(b, h, nb, nh, d_prime):
    bar_positions = []
    if nb > 0:
        x_coords_b = np.linspace(d_prime, b - d_prime, nb)
        for x in x_coords_b:
            bar_positions.append((x, d_prime))
            bar_positions.append((x, h - d_prime))
    if nh > 2:
        y_coords_h = np.linspace(d_prime, h - d_prime, nh)[1:-1]
        for y in y_coords_h:
            bar_positions.append((d_prime, y))
            bar_positions.append((b - d_prime, y))
    return sorted(list(set(bar_positions)))

def get_layers_from_positions(steel_positions, axis):
    layers = {}
    coord_index = 1 if axis == 'X' else 0
    for pos in steel_positions:
        layer_pos = pos[coord_index]
        if layer_pos in layers:
            layers[layer_pos] += 1
        else:
            layers[layer_pos] = 1
    return layers

def calculate_interaction_diagram(fc, fy, b, h, layers, bar_area):
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
    d_t = max(steel_pos)
    Pn_nom_list, Mn_nom_list = [], []
    Pn_design_list, Mn_design_list = [], []
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    phi_pc = 0.65
    Pn_nom_list.append(Pn_pc)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pc * phi_pc)
    Mn_design_list.append(0.0)
    c_values = np.logspace(np.log10(0.01), np.log10(h * 5), 300)
    for c in c_values:
        a = beta1 * c
        if a > h: a = h
        Cc = 0.85 * fc * a * b
        Mc = Cc * (h / 2.0 - a / 2.0)
        Pn_s, Mn_s = 0.0, 0.0
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
    Pn_pt = -fy * Ast_total
    phi_pt = 0.90
    Pn_nom_list.append(Pn_pt)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pt * phi_pt)
    Mn_design_list.append(0.0)
    Pn_nom, Mn_nom = np.array(Pn_nom_list), np.array(Mn_nom_list)
    Pn_design, Mn_design = np.array(Pn_design_list), np.array(Mn_design_list)
    sort_indices = np.argsort(Pn_nom)[::-1]
    return (Pn_nom[sort_indices]/1000, Mn_nom[sort_indices]/100000,
            Pn_design[sort_indices]/1000, Mn_design[sort_indices]/100000)

# --- ฟังก์ชันวาดหน้าตัดเสาด้วย Plotly (ปรับปรุงการแสดงผลเหล็กเสริม) ---
def draw_column_section_plotly(b, h, steel_positions, bar_dia_mm):
    fig = go.Figure()

    # 1. วาดหน้าตัดคอนกรีต
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h,
                  line=dict(color="Black", width=2), fillcolor="LightGrey")

    # 2. วาดเหล็กเสริม (แก้ไขให้แสดงผล)
    bar_x = [pos[0] for pos in steel_positions]
    bar_y = [pos[1] for pos in steel_positions]
    
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers',
        marker=dict(
            color='DarkSlateGray', 
            size=bar_dia_mm * 0.8, # ปรับขนาด marker (เดิม bar_dia_cm * 5 อาจใหญ่ไป)
            symbol='circle',
            line=dict(width=1, color='Black') # เพิ่มเส้นขอบให้เหล็ก
        ), 
        hoverinfo='none', # ไม่ต้องแสดง tooltip สำหรับเหล็ก
        showlegend=False # ไม่ต้องแสดงใน legend
    ))
    
    # 3. ตั้งค่า Layout
    fig.update_layout(
        title="Column Cross-Section",
        xaxis_title="Width, b (cm)",
        yaxis_title="Height, h (cm)",
        yaxis_scaleanchor="x", # ทำให้สัดส่วนแกน x และ y เท่ากัน (aspect ratio = 1)
        xaxis_range=[-b*0.1, b*1.1],
        yaxis_range=[-h*0.1, h*1.1],
        width=500, height=500,
        showlegend=False
    )
    return fig

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator (Interactive)")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram ของเสาคอนกรีตเสริมเหล็ก (เพื่อการศึกษาเท่านั้น)")

# --- Sidebar Inputs ---
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
    st.markdown("---")
    st.header("ตรวจสอบแรงจากไฟล์ CSV")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ load_export.csv", type=["csv"])
    
    df_loads = None
    selected_column = None
    selected_story = None
    if uploaded_file is not None:
        try:
            df_loads = pd.read_csv(uploaded_file)
            required_cols = {'Story', 'Column', 'P', 'M2', 'M3'}
            if not required_cols.issubset(df_loads.columns):
                st.sidebar.error(f"ไฟล์ CSV ต้องมีคอลัมน์: {', '.join(required_cols)}")
                df_loads = None
            else:
                column_options = sorted(df_loads['Column'].unique())
                selected_column = st.sidebar.selectbox("เลือกหมายเลขเสาที่ต้องการตรวจสอบ:", column_options)
                if selected_column:
                    story_options = sorted(df_loads[df_loads['Column'] == selected_column]['Story'].unique())
                    selected_story = st.sidebar.selectbox("เลือกชั้นที่ต้องการตรวจสอบ:", story_options)
        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

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
    fig_section = draw_column_section_plotly(b_in, h_in, steel_positions, bar_dia_mm)
    st.plotly_chart(fig_section, use_container_width=True)

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
                fig_diagram = go.Figure()

                fig_diagram.add_trace(go.Scatter(x=Mn_nom, y=Pn_nom, mode='lines', name='Nominal Strength',
                                                 line=dict(color='blue', width=2),
                                                 hovertemplate='<b>P:</b> %{y:.2f} Ton<br><b>M:</b> %{x:.2f} Ton-m<extra></extra>'))
                
                fig_diagram.add_trace(go.Scatter(x=Mn_design, y=Pn_design, mode='lines', name='Design Strength (ΦPn, ΦMn)',
                                                 line=dict(color='red', width=2),
                                                 hovertemplate='<b>ΦP:</b> %{y:.2f} Ton<br><b>ΦM:</b> %{x:.2f} Ton-m<extra></extra>'))

                if selected_column and selected_story and df_loads is not None:
                    mask = (df_loads['Column'] == selected_column) & (df_loads['Story'] == selected_story)
                    column_data = df_loads[mask].copy()
                    if not column_data.empty:
                        column_data['P_ton'] = -column_data['P']
                        column_data['M2_ton_m'] = abs(column_data['M2'])
                        column_data['M3_ton_m'] = abs(column_data['M3'])
                        
                        plot_M = column_data['M3_ton_m'] if bending_axis.startswith('X') else column_data['M2_ton_m']
                        plot_P = column_data['P_ton']
                        plot_case = column_data['Output Case'] # Output Case สำหรับ hover
                        
                        fig_diagram.add_trace(go.Scatter(x=plot_M, y=plot_P, mode='markers', 
                            name=f'Loads for {selected_column} ({selected_story})',
                            marker=dict(color='green', size=10, symbol='diamond', line=dict(width=1, color='Black')),
                            hovertemplate='<b>Output Case:</b> %{text}<br><b>P:</b> %{y:.2f} Ton<br><b>M:</b> %{x:.2f} Ton-m<extra></extra>',
                            text=plot_case # ใช้ text attribute เพื่อส่งข้อมูล Output Case ไปแสดงใน hover
                        ))

                fig_diagram.update_layout(
                    title=f"P-M Interaction Diagram ({axis_label} Axis)",
                    xaxis_title="Moment, M (Ton-m)",
                    yaxis_title="Axial Load, P (Ton)",
                    legend_title="Legend",
                    height=700
                )
                
                # ปรับแต่งเส้นแกน X และ Y ที่ค่า 0 ให้ชัดเจน
                fig_diagram.update_xaxes(
                    zeroline=True, zerolinewidth=3, zerolinecolor='Black',
                    showgrid=True, gridwidth=1, gridcolor='LightGray'
                )
                fig_diagram.update_yaxes(
                    zeroline=True, zerolinewidth=3, zerolinecolor='Black',
                    showgrid=True, gridwidth=1, gridcolor='LightGray'
                )
                
                st.plotly_chart(fig_diagram, use_container_width=True)

                if selected_column and selected_story and df_loads is not None:
                    st.write(f"ข้อมูลแรงสำหรับเสา **{selected_column}** ที่ชั้น **{selected_story}** (จากไฟล์ CSV)")
                    mask = (df_loads['Column'] == selected_column) & (df_loads['Story'] == selected_story)
                    display_df = df_loads[mask][['Story', 'Column', 'Output Case', 'P', 'M2', 'M3']].reset_index(drop=True)
                    st.dataframe(display_df)
            else:
                st.error("เกิดข้อผิดพลาดในการคำนวณ")
st.markdown("---")
