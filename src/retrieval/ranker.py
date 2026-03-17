"""Ranker for re-ranking and deduplicating search results."""

from src.models.search_result import SearchResult


class Ranker:
    """Re-ranks search results and removes duplicates."""

    def rerank(self, results: list[SearchResult], query: str = None) -> list[SearchResult]:
        """Re-rank results by relevance and update rank positions.

        Args:
            results: List of SearchResult objects to re-rank.
            query: Optional query string (unused for now, for future enhancement).

        Returns:
            Re-ranked list sorted by score (descending) with updated rank positions.
        """
        # Sort by score descending
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        # Update rank positions
        for i, result in enumerate(sorted_results, start=1):
            result.rank = i

        return sorted_results

    def deduplicate(
        self, results: list[SearchResult], similarity_threshold: float = 0.95
    ) -> list[SearchResult]:
        """Remove near-duplicate results based on content similarity.

        Args:
            results: List of SearchResult objects.
            similarity_threshold: Similarity score threshold (0-1) above which to consider duplicates.

        Returns:
            Deduplicated list with similar results removed, sorted by score descending.
        """
        if not results:
            return []

        # Keep track of which results to keep
        kept_indices = []

        for i, result in enumerate(results):
            is_duplicate = False

            # Compare with all previously kept results
            for kept_idx in kept_indices:
                kept_result = results[kept_idx]

                # Calculate Jaccard similarity between content
                content1 = result.document.content
                content2 = kept_result.document.content

                similarity = self._calculate_similarity(content1, content2)

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_indices.append(i)

        # Return deduplicated results
        deduplicated = [results[i] for i in kept_indices]

        # Re-rank to ensure proper ordering and rank values
        return self.rerank(deduplicated)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts at character level.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0

        # Convert to character sets
        set1 = set(text1)
        set2 = set(text2)

        # Calculate Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union
