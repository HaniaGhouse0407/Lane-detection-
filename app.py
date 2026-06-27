"""
Lane Detection System — Real-Time Autonomous Driving CV Pipeline
Author: Hania Ghouse | github.com/HaniaGhouse0407
Stack: OpenCV · NumPy · Streamlit · Canny + Hough Transform
"""
import streamlit as st
import cv2, numpy as np, time
from PIL import Image

st.set_page_config(page_title="Lane Detection", page_icon="🚗", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0A0A0A,#111827);}
.hero h1{font-size:2.4rem;font-weight:900;background:linear-gradient(135deg,#FBBF24,#F97316);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;}
.metric{background:#1F2937;border:1px solid #FBBF2444;border-radius:10px;padding:.9rem;text-align:center;}
.metric .v{font-size:1.6rem;font-weight:800;color:#FBBF24;}
.metric .l{font-size:.78rem;color:#6B7280;}
.stButton>button{background:linear-gradient(135deg,#FBBF24,#F97316);color:#000;border:none;border-radius:8px;font-weight:700;width:100%;}
div[data-testid="stFileUploader"]{background:#1F2937;border:2px dashed #FBBF2455;border-radius:12px;padding:1rem;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Detection Parameters")
    blur_k = st.slider("Gaussian Blur Kernel", 3, 15, 5, step=2)
    low_thresh = st.slider("Canny Low Threshold", 10, 100, 50)
    high_thresh = st.slider("Canny High Threshold", 50, 300, 150)
    hough_thresh = st.slider("Hough Threshold", 10, 100, 50)
    min_line_len = st.slider("Min Line Length", 10, 100, 40)
    max_line_gap = st.slider("Max Line Gap", 10, 200, 100)
    roi_height = st.slider("ROI Height %", 40, 80, 60)
    show_edges = st.toggle("Show Edge Map", False)
    show_roi = st.toggle("Show ROI Mask", False)
    line_thickness = st.slider("Lane Line Thickness", 1, 8, 3)
    line_color = st.color_picker("Lane Line Color", "#00FF64")

st.markdown('''<div class="hero"><h1>🚗 Lane Detection System</h1></div>
<p style="text-align:center;color:#6B7280">Canny Edge Detection · Hough Transform · ROI Masking · Real-Time Pipeline</p>
''', unsafe_allow_html=True)
st.divider()

def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

def detect_lanes(img_bgr, params):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (params["blur_k"], params["blur_k"]), 0)
    edges = cv2.Canny(blur, params["low"], params["high"])
    
    roi_y = int(h * (1 - params["roi_h"] / 100))
    mask = np.zeros_like(edges)
    poly = np.array([[(int(w*.05),h),(int(w*.45),roi_y),(int(w*.55),roi_y),(int(w*.95),h)]])
    cv2.fillPoly(mask, poly, 255)
    masked = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked, 1, np.pi/180,
        threshold=params["hough_t"], minLineLength=params["min_len"],
        maxLineGap=params["max_gap"])
    
    overlay = img_bgr.copy()
    left_lines, right_lines = [], []
    mid = w // 2
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = (y2-y1)/(x2-x1+1e-6)
            if abs(slope) < 0.3: continue
            if x1 < mid and x2 < mid: left_lines.append(line[0])
            else: right_lines.append(line[0])
    
    color = hex_to_bgr(params["color"])
    count = 0
    for seg_lines in [left_lines, right_lines]:
        if seg_lines:
            pts = np.array(seg_lines)
            x1m,y1m,x2m,y2m = pts[:,0].mean(),pts[:,1].mean(),pts[:,2].mean(),pts[:,3].mean()
            cv2.line(overlay,(int(x1m),int(y1m)),(int(x2m),int(y2m)),color,params["thick"])
            count += 1
    
    result = cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0)
    if params["show_roi"]:
        cv2.polylines(result, poly, True, (255,165,0), 2)
    
    return result, edges if params["show_edges"] else None, count, left_lines, right_lines

col1, col2 = st.columns([1, 1.4], gap="large")
with col1:
    st.markdown("### 📸 Input")
    uploaded = st.file_uploader("Upload road image or video frame", type=["jpg","jpeg","png","bmp"])
    
    st.markdown("**Or use a sample:**")
    sample = st.selectbox("", ["","Highway","City Street","Rain Conditions","Night Driving"], label_visibility="collapsed")
    
    if st.button("🚀 Detect Lanes", use_container_width=True):
        if uploaded or sample:
            st.session_state["run_detection"] = True
            if uploaded:
                pil = Image.open(uploaded).convert("RGB")
                st.session_state["img"] = np.array(pil)
            else:
                h2, w2 = 480, 640
                img = np.zeros((h2,w2,3), np.uint8)
                img[:] = (50,50,50)
                for i in range(0,h2,40):
                    cv2.line(img,(0,i),(w2,i),(60,60,60),1)
                cv2.line(img,(200,h2),(300,h2//2),(255,255,255),3)
                cv2.line(img,(440,h2),(340,h2//2),(255,255,255),3)
                cv2.line(img,(320,h2-50),(320,h2//2+20),(200,200,200),2)
                st.session_state["img"] = img
        else:
            st.warning("Upload an image or select a sample.")

with col2:
    st.markdown("### 🛣️ Detection Result")
    if st.session_state.get("run_detection") and "img" in st.session_state:
        with st.spinner("Running Canny + Hough pipeline..."):
            params = {"blur_k":blur_k if blur_k%2==1 else blur_k+1,"low":low_thresh,
                "high":high_thresh,"hough_t":hough_thresh,"min_len":min_line_len,
                "max_gap":max_line_gap,"roi_h":roi_height,"show_edges":show_edges,
                "show_roi":show_roi,"thick":line_thickness,"color":line_color}
            result, edges_img, n_lanes, left, right = detect_lanes(
                cv2.cvtColor(st.session_state["img"], cv2.COLOR_RGB2BGR), params)
            time.sleep(0.5)
        
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(Image.fromarray(result_rgb), caption="Lane Detection Output", use_column_width=True)
        
        if edges_img is not None:
            st.image(Image.fromarray(edges_img), caption="Canny Edge Map", use_column_width=True)
        
        c1,c2,c3,c4 = st.columns(4)
        for col, (v,l) in zip([c1,c2,c3,c4],[
            (str(n_lanes),"Lanes Detected"),
            (str(len(left)),"Left Segments"),
            (str(len(right)),"Right Segments"),
            (f"{low_thresh}-{high_thresh}","Canny Range"),
        ]):
            col.markdown(f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    else:
        st.info("Upload a road image and click Detect Lanes.")
