"""
Optional NodeRAG integration (Lựa chọn C).

NodeRAG is an all-in-one GraphRAG framework. Setup steps:

1. pip install NodeRAG
2. Create project folder:
   noderag_project/
     input/
       tech_company_corpus.txt  (copy from data/)
3. Build:
   python -m NodeRAG.build -f noderag_project
4. Edit Node_config.yaml with your OpenAI API key
5. Query:
   from NodeRAG import NodeConfig, NodeSearch
   config = NodeConfig.from_main_folder("noderag_project")
   search = NodeSearch(config)
   ans = search.answer("Ai sáng lập OpenAI?")
   print(ans.response)
6. Visualize:
   python -m NodeRAG.Vis.html -f noderag_project -n 600
"""

from pathlib import Path
import shutil

from src.config import CORPUS_PATH, PROJECT_ROOT


def setup_noderag_project(target_dir: Path | None = None) -> Path:
    """Copy corpus into NodeRAG project input folder."""
    target = target_dir or PROJECT_ROOT / "noderag_project"
    input_dir = target / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CORPUS_PATH, input_dir / "tech_company_corpus.txt")
    print(f"NodeRAG input ready at: {input_dir}")
    print("Next: pip install NodeRAG && python -m NodeRAG.build -f", target)
    return target


if __name__ == "__main__":
    setup_noderag_project()
