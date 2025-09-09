# ui_v2.py
# Version 2 – Add Retrieval (Knowledge Grounding)
# RAG-lite: CSV → SQLite → TF-IDF retrieve → LLM (or fallback) → grounded comparison

import os, io, csv, math, json, sqlite3, re, time
import streamlit as st
from typing import List, Dict, Tuple, Any
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # still works without OpenAI installed

st.set_page_config(page_title="V2 — Retrieval Grounding (RAG-lite)", page_icon="🧭", layout="wide")
st.title("V2 — Retrieval (RAG-lite)")
st.caption("Query → retrieve from CSV-backed SQLite → feed context to LLM → grounded answer. Adds the “R” in RAG.")

# ----------------------------
# 0) Helpers: CSV → lists of dicts (no pandas)
# ----------------------------
def read_csv_to_dicts(file_bytes: bytes) -> List[Dict[str, str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]

# ----------------------------
# 1) SQLite setup
# ----------------------------
def load_into_sqlite(products: List[Dict[str, str]], reviews: List[Dict[str, str]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            name TEXT,
            supplier TEXT,
            category TEXT,
            price REAL,
            rating REAL,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE reviews (
            review_id TEXT PRIMARY KEY,
            product_id TEXT,
            stars REAL,
            text TEXT
        )
    """)
    p_cols = ["product_id","name","supplier","category","price","rating","description"]
    for r in products:
        cur.execute(
            f"INSERT OR REPLACE INTO products ({','.join(p_cols)}) VALUES (?,?,?,?,?,?,?)",
            [r.get("product_id"), r.get("name"), r.get("supplier"), r.get("category"),
             float(r.get("price", 0) or 0), float(r.get("rating", 0) or 0), r.get("description")]
        )
    rv_cols = ["review_id","product_id","stars","text"]
    for r in reviews:
        cur.execute(
            f"INSERT OR REPLACE INTO reviews ({','.join(rv_cols)}) VALUES (?,?,?,?)",
            [r.get("review_id"), r.get("product_id"),
             float(r.get("stars", 0) or 0), r.get("text")]
        )
    conn.commit()
    return conn

# ----------------------------
# 2) Tiny TF-IDF retriever (no sklearn)
# ----------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())

def build_docs_for_index(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT product_id,name,supplier,category,price,rating,description FROM products")
    rows = cur.fetchall()
    docs = []
    for pid, name, supplier, category, price, rating, desc in rows:
        txt = " ".join(str(x) for x in [pid, name, supplier, category, price, rating, desc] if x is not None)
        docs.append({
            "product_id": pid, "name": name, "supplier": supplier, "category": category,
            "price": float(price or 0), "rating": float(rating or 0),
            "description": desc, "text": txt
        })
    return docs

def build_tfidf_index(docs: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[Dict[int, float]], Dict[int, float]]:
    # vocab -> id
    vocab: Dict[str, int] = {}
    df_counts: Dict[int, int] = {}
    N = len(docs)
    doc_vecs: List[Dict[int, float]] = []

    # pass 1: vocab + df
    for d in docs:
        tokens = set(tokenize(d["text"]))
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            df_counts[vocab[tok]] = df_counts.get(vocab[tok], 0) + 1

    # idf
    idf: Dict[int, float] = {}
    for tid, df in df_counts.items():
        idf[tid] = math.log((N + 1) / (df + 1)) + 1.0

    # pass 2: tfidf vectors
    for d in docs:
        toks = tokenize(d["text"])
        tf: Dict[int, int] = {}
        for t in toks:
            tid = vocab.get(t)
            if tid is not None:
                tf[tid] = tf.get(tid, 0) + 1
        vec: Dict[int, float] = {}
        length_sq = 0.0
        for tid, f in tf.items():
            w = (f / max(1, len(toks))) * idf.get(tid, 0.0)
            vec[tid] = w
            length_sq += w*w
        # normalize
        norm = math.sqrt(length_sq) or 1.0
        vec = {k: v/norm for k, v in vec.items()}
        doc_vecs.append(vec)

    return vocab, doc_vecs, idf

def vectorize_query(q: str, vocab: Dict[str, int], idf: Dict[int, float]) -> Dict[int, float]:
    toks = tokenize(q)
    tf: Dict[int, int] = {}
    for t in toks:
        tid = vocab.get(t)
        if tid is not None:
            tf[tid] = tf.get(tid, 0) + 1
    vec: Dict[int, float] = {}
    length_sq = 0.0
    L = max(1, len(toks))
    for tid, f in tf.items():
        w = (f / L) * idf.get(tid, 0.0)
        vec[tid] = w
        length_sq += w*w
    norm = math.sqrt(length_sq) or 1.0
    return {k: v/norm for k, v in vec.items()}

def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())

def retrieve_topk(query: str, docs: List[Dict[str, Any]], vocab, doc_vecs, idf, k=8) -> List[Tuple[Dict[str, Any], float]]:
    qv = vectorize_query(query, vocab, idf)
    scores = [(docs[i], cosine_sparse(qv, doc_vecs[i])) for i in range(len(docs))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

# ----------------------------
# 3) Reviews + context building
# ----------------------------
def fetch_reviews_for(conn: sqlite3.Connection, product_id: str, limit=3) -> Tuple[float, List[Dict[str, Any]]]:
    cur = conn.cursor()
    cur.execute("SELECT stars, text FROM reviews WHERE product_id = ? ORDER BY stars DESC", (product_id,))
    rows = cur.fetchall()
    if not rows:
        return 0.0, []
    avg = sum(r[0] or 0 for r in rows) / len(rows)
    snips = [{"stars": r[0] or 0.0, "text": r[1]} for r in rows[:limit]]
    return avg, snips

def build_context(conn: sqlite3.Connection, retrieved: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    items = []
    for d, score in retrieved:
        avg, snips = fetch_reviews_for(conn, d["product_id"], limit=3)
        items.append({
            "product_id": d["product_id"],
            "name": d["name"],
            "supplier": d["supplier"],
            "category": d["category"],
            "price": d["price"],
            "rating": d["rating"] if d["rating"] else avg,
            "retrieval_score": round(float(score), 4),
            "review_avg": round(avg, 2),
            "reviews": snips
        })
    return {"items": items}

# ----------------------------
# 4) LLM wrapper (with graceful fallback)
# ----------------------------
def have_openai() -> bool:
    return (OpenAI is not None) and bool(os.environ.get("OPENAI_API_KEY", "").strip())

def summarize_with_llm(context: Dict[str, Any], query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (raw_text, parsed_json_or_none)
    """
    system_prompt = (
        "You are a helpful procurement analyst. Use ONLY the provided context items "
        "to answer. Compare suppliers/products grounded in the data: names, prices, ratings. "
        "Return a JSON with shape:\n"
        "{\n"
        '  "comparison": [\n'
        '    {"supplier": "STRING", "products": [{"product_id":"", "name":"", "price":NUMBER, "rating":NUMBER}]}\n'
        "  ],\n"
        '  "summary": "STRING",\n'
        '  "recommendation": "STRING"\n'
        "}"
    )

    if have_openai():
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            user_prompt = (
                f"Query: {query}\n"
                f"Context JSON:\n{json.dumps(context)}\n\n"
                "Respond with STRICT JSON only."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=700,
            )
            raw = resp.choices[0].message.content
            try:
                return raw, json.loads(raw)
            except Exception:
                return raw, None
        except Exception as e:
            return f"<<LLM error>> {e}", None

    # ---- Fallback (no OpenAI): simple, deterministic template grounded on context
    items = context.get("items", [])
    by_supplier: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_supplier.setdefault(it["supplier"], []).append(it)
    comparison = []
    for sup, plist in by_supplier.items():
        plist = sorted(plist, key=lambda x: (x["price"], -x.get("rating", 0)))
        comparison.append({
            "supplier": sup,
            "products": [
                {"product_id": p["product_id"], "name": p["name"], "price": p["price"], "rating": p.get("rating", 0)}
                for p in plist
            ]
        })
    # pick a recommendation: lowest price among top matches; tie-breaker by rating
    best = None
    for sup, plist in by_supplier.items():
        cand = min(plist, key=lambda p: (p["price"], -p.get("rating", 0)))
        if (best is None) or (cand["price"] < best["price"]) or \
           (cand["price"] == best["price"] and cand.get("rating", 0) > best.get("rating", 0)):
            best = cand
    summary = "Grounded comparison generated without LLM (fallback)."
    rec = f"Recommend {best['name']} from {best['supplier']} at ₹{best['price']} (rating {best.get('rating',0)})." if best else "No clear recommendation."
    parsed = {"comparison": comparison, "summary": summary, "recommendation": rec}
    return json.dumps(parsed, ensure_ascii=False), parsed

# ----------------------------
# 5) UI: Upload CSVs → Build index → Query
# ----------------------------
st.markdown("### 1) Load your data (CSV)")
left, right = st.columns(2)
with left:
    products_file = st.file_uploader("Upload products.csv", type=["csv"], key="products_csv")
    st.markdown(
        "Expected headers: `product_id,name,supplier,category,price,rating,description`"
    )
with right:
    reviews_file = st.file_uploader("Upload reviews.csv", type=["csv"], key="reviews_csv")
    st.markdown("Expected headers: `review_id,product_id,stars,text`")

# Optional tiny demo data
with st.expander("No CSVs handy? Load a tiny demo dataset"):
    if st.button("Load demo CSVs"):
        demo_products = """product_id,name,supplier,category,price,rating,description
SOAP-100,Bath Soap 100g,Supplier A,Personal Care,25,4.2,Gentle cleansing soap
SOAP-200,Bath Soap 200g,Supplier B,Personal Care,45,4.0,Larger pack, mild fragrance
SHAM-250,Shampoo 250ml,Supplier A,Hair Care,120,4.4,SLS-free shampoo
SHAM-200,Shampoo 200ml,Supplier B,Hair Care,95,4.3,Value pack shampoo
PASTE-75,Toothpaste 75g,Supplier C,Oral Care,60,4.5,Fluoride toothpaste
"""
        demo_reviews = """review_id,product_id,stars,text
R1,SOAP-100,5,Great value
R2,SOAP-100,4,Mild on skin
R3,SOAP-200,4,Lasts long
R4,SHAM-250,5,Super clean feel
R5,SHAM-250,4,Smells good
R6,SHAM-200,4,Good for price
R7,PASTE-75,5,Fresh breath
"""
        st.session_state["products_demo"] = read_csv_to_dicts(demo_products.encode())
        st.session_state["reviews_demo"] = read_csv_to_dicts(demo_reviews.encode())
        st.success("Loaded demo data into session. You can proceed to build the index below.")

# Choose source (uploaded vs demo)
products_data = None
reviews_data = None
if products_file and reviews_file:
    products_data = read_csv_to_dicts(products_file.read())
    reviews_data  = read_csv_to_dicts(reviews_file.read())
elif "products_demo" in st.session_state and "reviews_demo" in st.session_state:
    products_data = st.session_state["products_demo"]
    reviews_data  = st.session_state["reviews_demo"]

if products_data and reviews_data:
    st.success(f"Loaded {len(products_data)} products and {len(reviews_data)} reviews.")
    with st.expander("Peek products", expanded=False):
        st.json(products_data[:5])
    with st.expander("Peek reviews", expanded=False):
        st.json(reviews_data[:5])

    if st.button("Build SQLite & Index"):
        t0 = time.time()
        conn = load_into_sqlite(products_data, reviews_data)
        docs = build_docs_for_index(conn)
        vocab, doc_vecs, idf = build_tfidf_index(docs)
        st.session_state["conn"] = conn
        st.session_state["docs"] = docs
        st.session_state["vocab"] = vocab
        st.session_state["doc_vecs"] = doc_vecs
        st.session_state["idf"] = idf
        st.success(f"Built index over {len(docs)} products in {int((time.time()-t0)*1000)} ms.")
else:
    st.info("Upload both CSVs (or load the demo data) to continue.")

st.markdown("### 2) Ask a grounded question")
query = st.text_input(
    "Example: “Compare supplier A and supplier B” or “Find cheapest supplier for shampoo 250ml and summarize reviews.”",
    value="Compare Supplier A and Supplier B"
)

if st.button("Retrieve & Answer"):
    if not all(k in st.session_state for k in ["conn","docs","vocab","doc_vecs","idf"]):
        st.error("Please build the index first.")
        st.stop()

    conn = st.session_state["conn"]
    docs = st.session_state["docs"]
    vocab = st.session_state["vocab"]
    doc_vecs = st.session_state["doc_vecs"]
    idf = st.session_state["idf"]

    # --- Retrieval
    scored = retrieve_topk(query, docs, vocab, doc_vecs, idf, k=8)
    context = build_context(conn, scored)

    tab_ret, tab_ans, tab_table = st.tabs(["Retrieved items", "Answer (LLM or fallback)", "Comparison table"])
    with tab_ret:
        st.json(context)

    # --- Answer
    raw, parsed = summarize_with_llm(context, query)
    with tab_ans:
        st.subheader("Raw model (or fallback) output")
        st.code(raw, language="json")
        if parsed:
            st.subheader("Parsed JSON")
            st.json(parsed)

    # --- Render comparison table
    with tab_table:
        st.write("Grounded comparison by supplier (from retrieved items):")
        # Build a simple table from parsed or context
        rows = []
        by_supplier = {}
        if parsed and "comparison" in parsed:
            for blk in parsed["comparison"]:
                sup = blk["supplier"]
                for p in blk.get("products", []):
                    rows.append([sup, p.get("product_id"), p.get("name"), p.get("price"), p.get("rating")])
        else:
            for it in context["items"]:
                rows.append([it["supplier"], it["product_id"], it["name"], it["price"], it.get("rating", 0)])

        if rows:
            # render as markdown table (no pandas dependency)
            header = "| Supplier | Product ID | Name | Price | Rating |\n|---|---|---|---:|---:|\n"
            body = "\n".join(
                f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows
            )
            st.markdown(header + body)
        else:
            st.info("No rows to display.")

st.markdown("---")
st.markdown("**What changed vs V1?**")
st.markdown(
    "- **Grounding**: Answers are backed by your CSV data via SQLite.\n"
    "- **Retrieval**: Tiny TF-IDF retriever routes only relevant rows to the LLM.\n"
    "- **Safer output**: Even without an API key, a deterministic fallback gives a grounded comparison.\n"
)
