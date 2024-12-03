using System;
using System.Collections.Generic;

namespace AllMiniLmL6V2Sharp
{
    public interface IEmbedder
    {
        /// <summary>
        /// Generates an embedding array for the given sentence.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>Sentence embeddings</returns>
        float[] GenerateEmbedding(string sentence);
        /// <summary>
        /// Generates an embedding array for the given sentences.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>An enumerable of embeddings.</returns>
        IEnumerable<(string sentence, float[] embedding)> GenerateEmbeddings(IEnumerable<string> sentences);
    }
}
