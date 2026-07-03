from hybrid_rag.text import light_stem, tokenize


def test_tokenize_keeps_identifiers():
    toks = tokenize("Got ERR_CONN_RESET from api-v1.2 (HTTP 429)", stem=False)
    assert "err_conn_reset" in toks and "api-v1.2" in toks and "429" in toks


def test_light_stem_conflates_inflections():
    assert light_stem("indexing") == light_stem("indexes") == "index"
    assert light_stem("429") == "429"
