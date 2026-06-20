# Reference: AI TradeOS Prototype V0.1 - Zero Budget Edition (Date: 2026)
import os
import json
import requests
import pandas as pd
import streamlit as st
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "tvly-your-key")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk-your-key")

# Initialize OpenAI client to point to Groq's ultra-fast API
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# We use Llama 3.1 8B which is lightning fast and free on Groq
LLM_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="AI TradeOS - V0.1", layout="wide")
st.title("🌍 AI TradeOS: Pakistan Sesame Sourcing (Zero-Budget)")

# ==========================================
# 2. DATA MODELS
# ==========================================


class QuoteAnalysis(BaseModel):
    supplier_name: str
    product_spec: str
    price_per_mt: float
    currency: str
    incoterm: str
    payment_terms: str
    delivery_time_days: int


# ==========================================
# 3. UI TABS & LOGIC
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["🔍 Supplier Finder", "✉️ RFQ Generator", "📊 Quote Analyzer"])

# ------------------------------------------
# TAB 1: SUPPLIER FINDER (Tavily API)
# ------------------------------------------
with tab1:
    st.header("Find Pakistani Sesame Exporters")
    search_query = st.text_input(
        "Search Query", value="top wholesale sesame seed exporters in Pakistan B2B contact")

    if st.button("Search Suppliers"):
        with st.spinner("Searching the web via Tavily AI..."):
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": search_query,
                "search_depth": "advanced",
                "include_answer": False,
                "max_results": 10
            }
            try:
                response = requests.post(
                    "https://api.tavily.com/search", json=payload, headers=headers)
                data = response.json()

                if "results" in data:
                    df = pd.DataFrame(data["results"])[
                        ["title", "url", "content"]]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("No results found.")
            except Exception as e:
                st.error(f"Search Failed: {str(e)}")

# ------------------------------------------
# TAB 2: RFQ GENERATOR (Groq API)
# ------------------------------------------
with tab2:
    st.header("Generate Request for Quotation (RFQ)")
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity (MT)", min_value=1, value=25)
        packing = st.selectbox(
            "Packing", ["50 kg PP bags", "25 kg PP bags", "Jute bags"])
    with col2:
        incoterm = st.selectbox("Preferred Incoterm", [
                                "FOB Karachi", "CIF Bandar Abbas", "EXW"])
        target_port = st.text_input(
            "Destination Port", value="Bandar Abbas, Iran")

    if st.button("Generate Email"):
        with st.spinner("Drafting professional RFQ..."):
            prompt = f"""
            Act as a professional Iranian import manager. Write a concise B2B Request for Quotation (RFQ) email to a Pakistani supplier.
            Product: Sesame Seeds
            Quantity: {quantity} MT
            Packing: {packing}
            Incoterm: {incoterm}
            Destination: {target_port}
            
            Tone: Serious, direct. Ask for best price, payment terms, and recent COA. Output ONLY the email body.
            """
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                st.text_area(
                    "Generated RFQ Email", value=response.choices[0].message.content, height=300)
            except Exception as e:
                st.error(f"LLM Error: {str(e)}")

# ------------------------------------------
# TAB 3: QUOTE ANALYZER (Groq JSON Mode)
# ------------------------------------------
with tab3:
    st.header("Analyze & Compare Quotes")
    quote_text = st.text_area(
        "Raw Quote Text", height=200, placeholder="Paste supplier email here...")

    if st.button("Analyze Quote"):
        with st.spinner("Extracting structured data..."):
            prompt = f"""
            Extract trading terms from the text below. 
            You MUST return a valid JSON object with EXACTLY these keys:
            "supplier_name" (string), "product_spec" (string), "price_per_mt" (number), "currency" (string), "incoterm" (string), "payment_terms" (string), "delivery_time_days" (number).
            If a value is missing, use "Unknown" or 0 for numbers.
            
            Text:
            {quote_text}
            """
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system",
                            "content": "You are a data extraction API. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

                # Parse JSON and validate via Pydantic
                raw_json = json.loads(response.choices[0].message.content)
                validated_data = QuoteAnalysis(**raw_json)

                st.success("Data Extracted Successfully!")
                st.json(validated_data.model_dump())

            except ValidationError as ve:
                st.error("Data validation failed. The AI missed some fields.")
                st.json(raw_json)
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")
