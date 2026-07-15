"""
Minimal Streamlit UI hitting the FastAPI backend. Kept deliberately thin —
all ML logic lives in the backend, this is just a demo shell.

Run:
    streamlit run frontend/app.py
Set API_URL env var if backend isn't on localhost:8000.
"""
import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Glance Fashion Retrieval", layout="wide")
st.title("👗 Fashion & Context Image Retrieval")

query = st.text_input("Describe what you're looking for",
                       placeholder="Someone wearing a blue shirt sitting on a park bench")
profile = st.selectbox("Optional style profile", [None, "professional", "minimalist",
                                                     "creative", "trendy", "classic"])
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        resp = requests.post(f"{API_URL}/search",
                              json={"query": query, "top_k": top_k, "profile": profile})
    if resp.status_code != 200:
        st.error(f"Search failed: {resp.text}")
    else:
        data = resp.json()
        st.caption(f"Parsed attributes: {data['parsed_attributes']}")

        cols = st.columns(min(top_k, 5))
        for i, result in enumerate(data["results"]):
            col = cols[i % len(cols)]
            with col:
                image_path = os.path.join(os.path.dirname(__file__), "..", "data", result["path"])
                if os.path.exists(image_path):
                    st.image(image_path, use_column_width=True)
                exp = result["explanation"]
                st.markdown(f"**Score:** {exp['final_score']}")
                st.markdown(f"**Matched:** {exp['matched_count']}")
                for attr, val in exp["matched_attributes"].items():
                    st.markdown(f"✔ {attr}: {val}")
                if exp["vibe_matched"]:
                    st.markdown(f"✔ vibe: {exp['vibe_matched']}")

                thumbs_up, thumbs_down = st.columns(2)
                if thumbs_up.button("👍", key=f"up_{result['image_id']}"):
                    requests.post(f"{API_URL}/feedback", json={
                        "query": query, "image_id": result["image_id"], "relevant": True})
                if thumbs_down.button("👎", key=f"down_{result['image_id']}"):
                    requests.post(f"{API_URL}/feedback", json={
                        "query": query, "image_id": result["image_id"], "relevant": False})

st.divider()
st.subheader("Add a new image to the dataset")
uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
if uploaded and st.button("Index this image"):
    resp = requests.post(f"{API_URL}/upload",
                          files={"file": (uploaded.name, uploaded.getvalue())})
    st.json(resp.json())
