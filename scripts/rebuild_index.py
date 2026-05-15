import sys
import os

# Add the parent directory to the path so we can import the pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline

def main():
    print("Force rebuilding the vector index...")
    pipeline = RAGPipeline()
    pipeline.initialize(force_rebuild=True)
    print("Done!")

if __name__ == "__main__":
    main()
