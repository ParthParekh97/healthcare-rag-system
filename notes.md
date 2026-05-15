# Development Notes - Healthcare RAG

## To-Do
- [x] Basic ingestion pipeline
- [x] Streamlit UI
- [x] Grounding validation logic
- [ ] Add support for more medical PDF types (currently just text/simple PDF)
- [ ] Improve chunking strategy for long clinical tables
- [ ] Add more specific medical evaluation metrics (maybe RAGAS?)

## Notes
- `all-MiniLM-L6-v2` seems fast enough for now, but might need a larger model if we scale to thousands of docs.
- The similarity threshold (0.35) is a bit loose, watch out for false positives in retrieval.
- Need to check if Gemini-2.5-flash handles long contexts better than 1.5.

## Issues
- Some PDFs with nested columns are chunking weirdly. Loader might need a better regex.
- Validation grounding score sometimes dips when using very technical abbreviations.
