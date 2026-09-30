import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopping_agent.dataset import preprocess
from shopping_agent.settings import settings
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="smoke/demo only; omit to process all records")
args = parser.parse_args()
print(preprocess(settings.data_dir, settings.data_dir / "products.jsonl", args.limit))
