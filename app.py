import streamlit as st
import pandas as pd
import os  # <--- เรียกใช้ตัวจัดการไฟล์
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI ทำนายผลใบหยุด", page_icon="🤖")
st.title("🤖 AI ผู้ช่วยทำนายผลใบหยุด")

# --- 2. ฟังก์ชันโหลดข้อมูลแบบเหมาเข่ง ---
@st.cache_data
def load_data():
    folder_path = '.'  # ชื่อโฟลเดอร์ที่เก็บ Excel
    all_dfs = []
    
    # ตรวจสอบว่ามีโฟลเดอร์ไหม
    if not os.path.exists(folder_path):
        return None

    # วนลูปอ่านทุกไฟล์ในโฟลเดอร์นั้น
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    if len(files) == 0:
        return "NO_FILES"

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            # อ่านไฟล์
            df_temp = pd.read_excel(file_path)
            
            # ลบช่องว่างหัวตาราง (สำคัญมาก เพื่อให้หัวตารางตรงกันทุกไฟล์)
            df_temp.columns = df_temp.columns.str.strip()
            
            # เก็บใส่ถังรวม
            all_dfs.append(df_temp)
        except Exception as e:
            continue # ถ้าไฟล์ไหนเสีย ให้ข้ามไป

    # ถ้ารวมแล้วมีข้อมูล
    if all_dfs:
        # รวมร่าง (Concatenate)
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Data Cleaning (จัดการค่าว่าง)
        search_col = 'Full_Description'
        cols_display = ['No of HoldTag', 'Customer', 'Cables type', 'Decision result']
        
        # แทนค่า nan ด้วยขีด -
        for col in cols_display:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '-')
        
        if search_col in df.columns:
            df[search_col] = df[search_col].fillna("ไม่ระบุ")
        else:
            df[search_col] = "ไม่ระบุ" # กัน Error
            
        return df
    
    return None

# เรียกใช้งานฟังก์ชันโหลด
df = load_data()

# --- 3. ส่วนแสดงผล ---
if df is not None and isinstance(df, pd.DataFrame):
    
    # สร้างสมอง AI
    search_col = 'Full_Description'
    result_col = 'Decision result'
    
    vectorizer = TfidfVectorizer()
    # แปลงข้อมูลเป็นตัวเลข (Training)
    tfidf_matrix = vectorizer.fit_transform(df[search_col].astype(str))

    st.success(f"✅ โหลดฐานข้อมูลครบทุกไฟล์แล้ว! (รวม {len(df)} รายการ)")

    # Sidebar: ปรับจำนวนผลลัพธ์
    top_n = st.sidebar.slider("จำนวนเคสที่แสดง", 1, 10, 3)

    # Main: ช่องค้นหา
    query = st.text_input("ระบุปัญหา:", placeholder="เช่น สายมีรอยถลอก...")

    if query:
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-top_n:][::-1]

        st.markdown("---")
        found_count = 0
        for idx in top_indices:
            score = similarity[idx]
            if score > 0.01:
                found_count += 1
                row = df.iloc[idx]
                decision = row.get(result_col, '-').lower()
                
                # Logic สี
                if any(x in decision for x in ['scrap', 'reject', 'ทิ้ง']):
                    st.error(f"🔴 {row.get(result_col)} ({score*100:.0f}%)")
                elif any(x in decision for x in ['pass', 'ผ่าน', 'accept', 'ok', 'อนุโลม']):
                    st.success(f"🟢 {row.get(result_col)} ({score*100:.0f}%)")
                else:
                    st.warning(f"🟡 {row.get(result_col)} ({score*100:.0f}%)")

                with st.expander(f"รายละเอียดใบหยุด: {row.get('No of HoldTag', '-')}"):
                    st.write(f"**ลูกค้า:** {row.get('Customer', '-')}")
                    st.write(f"**ชนิด:** {row.get('Cables type', '-')}")
                    st.write(f"**อาการ:** {row.get(search_col, '-')}")

        if found_count == 0:
            st.warning("ไม่พบเคสที่คล้ายกัน")

elif df == "NO_FILES":
    st.error("❌ ไม่พบไฟล์ Excel ในโฟลเดอร์ data เลยครับ")
else:

    st.error("❌ หาโฟลเดอร์ 'data' ไม่เจอ! กรุณาตรวจสอบว่าสร้างโฟลเดอร์ data และใส่ไฟล์ Excel ไว้ใน GitHub แล้ว")
