# probe_matches.py
import os, textwrap
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# ---- Load env
print("Loading environment variables...")
load_dotenv(find_dotenv(usecwd=True))

# Use os.getenv() instead of os.environ[] to work with .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"SUPABASE_URL loaded: {bool(SUPABASE_URL)}")
print(f"SUPABASE_KEY loaded: {bool(SUPABASE_KEY)}")
print(f"OPENAI_API_KEY loaded: {bool(OPENAI_API_KEY)}")

# Validate all required environment variables are present
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL not found in .env file")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print("All environment variables loaded successfully!")

EMBED_MODEL = "text-embedding-3-small"      # must match your table's 1536-dims
PDF_PATH = "human-nutrition-text.pdf"       # used as a filter in metadata
TOP_K = 3


queries = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
    "What are micronutrients?"
]


def main():
    print("Connecting to Supabase and OpenAI...")
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Connected successfully!")

    for q in queries:
        print(f"\nProcessing query: {q}")
        # embed query
        e = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding


        # call your RPC with a metadata filter to this PDF
        resp = sb.rpc("match_documents", {
            "query_embedding": e,
            "match_count": TOP_K,
            "filter": {"source": PDF_PATH}
        }).execute()


        rows = resp.data or []
        print("\n" + "="*90)
        print(f"QUERY: {q}")
        if not rows:
            print("  (no matches)")
            continue


        for rank, r in enumerate(rows, start=1):
            page = (r.get("metadata") or {}).get("page", "?")
            sim  = r.get("similarity", None)
            sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "?"
            preview = textwrap.shorten(r.get("content","").replace("\n"," "), width=160)
            print(f"  [{rank}] page {page}  sim={sim_str}  chunk_index={r.get('chunk_index')}")
            print(f"      {preview}")

    print("\n✅ All queries completed!")


if __name__ == "__main__":
    main()


