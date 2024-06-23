# RAGsperiments

A scratch repo for playing around with RAG.

## Notes

[Beyond the Basics of Retrieval for Augmenting Generation.](https://parlance-labs.com/education/rag/ben.html)

### Retrieval basics

* Compact MVP
    * This is the standard query/doc embedding + cosine similarity
    * AKA bi-encoder
    * A bi-encoder is:
        * Single vector representations
        * Docs and query embeddings computed separately
* Cross-encoders
    * Solves the problem of document and query being unaware of one another
    * You embed all documents AND the query at once -- but inefficient
    * Reranking is another solution
        * Use a powerful for expensive model to score a subset of documents retrieved by an efficient model
* Tf-idf/full text
    * Embeddings lose signal -- keywords don't
    * Always add this!
    * Can also add BM25, powered by tf-idf
    * Compute is so trivial and performance so good you may as well add it!
* At this point, pipeline is bi-encoder AND weighted tf-idf to db, scored to both, reranked
    * .7 to cosine, .3 to BM25 is a great starter for 10
* Metadata filtering
    * NER ([GliNER](https://github.com/urchade/GLiNER)?) to get business/query relevant info in metadata
    * Use extracted entities to pre-filter docs
* Compact MVP++
    * Query and docs embedded
    * Query and docs tf-idf'd
    * Docs add metadata with NER
    * Metadata filtered by query NER
    * Cosine search
    * BM25 search
    * Combine scores
    * Reranking
    * Results!
* Where to go next
    * Sparse ([SPLADE](https://github.com/naver/splade)) -- strong in-domain
    * Multi-vector ([ColBERT](https://github.com/stanford-futuredata/ColBERT)) -- strong out of domain

[Advanced Retrieval-Augmented Generation Techniques](https://www.youtube.com/watch?v=RZl4pe88sUU)

Adds:

* Query rewriting
    * Uses DSPy for this (for both prompt AND query)
* Semantic chunking or LLM chunking -- factoid database
* Fine tune embedding model
* Autocut similar groups -- so instead of a threshold, take the derivative of the similarity scores and cut when it drops
* Fine tune the LLM for the domain
