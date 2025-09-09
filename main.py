import typer
from app.store import load_csv
from app.agent import RecommenderAgent
from rich import print as rprint

app = typer.Typer(add_completion=False)

@app.command()
def ingest(products_csv: str, reviews_csv: str):
    """Load CSVs into SQLite."""
    load_csv(products_csv, reviews_csv)
    rprint("[green]Ingest complete[/green]")

@app.command()
def recommend(query: str, k: int = 5):
    """Get top-K recommendations with review summaries."""
    agent = RecommenderAgent()
    recs = agent.recommend(query, top_k=k)
    for i, rec in enumerate(recs, 1):
        p = rec.product
        rprint(f"[bold]{i}. {p.title}[/bold]  (score={rec.score:.3f})  "
               f"[dim]{p.category or ''} | ₹{p.price if p.price is not None else 'NA'}[/dim]")
        if rec.summary:
            rprint(f"   summary: {rec.summary}")
        if rec.pros:
            rprint(f"   pros: {', '.join(rec.pros[:5])}")
        if rec.cons:
            rprint(f"   cons: {', '.join(rec.cons[:5])}")

@app.command()
def summarize(product_id: str):
    """Summarize reviews for a specific product."""
    agent = RecommenderAgent()
    js = agent.summarize_product(product_id)
    rprint(js)

if __name__ == "__main__":
    app()
