from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopping_agent.dataset import download_dataset
from shopping_agent.settings import settings
print("downloading ShoppingBench to", settings.data_dir)
for path in download_dataset(settings.data_dir): print(path)
