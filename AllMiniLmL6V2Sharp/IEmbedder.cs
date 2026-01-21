using System.Collections.Generic;

namespace OoBDev.AllMiniLmL6V2Sharp;

/// <summary>
/// Interface for generating sentence embeddings.
/// </summary>
public interface IEmbedder
{
    /// <summary>
    /// Generates an embedding array for the given sentance.
    /// </summary>
    /// <param name="sentence">Text to embed.</param>
    /// <returns>Sentance embeddings</returns>
    IEnumerable<float> GenerateEmbedding(string sentence);
    /// <summary>
    /// Generates an embedding array for the given sentances.
    /// </summary>
    /// <param name="sentences">Text to embed.</param>
    /// <returns>An enumerable of embeddings.</returns>
    IEnumerable<IEnumerable<float>> GenerateEmbeddings(IEnumerable<string> sentences);
}
