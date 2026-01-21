using System.Collections.Generic;

namespace OoBDev.AllMiniLmL6V2Sharp.Tokenizer;

/// <summary>
/// Interface for text tokenization.
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Tokenizes the input text.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>An enumerable of tokens.</returns>
    IEnumerable<Token> Tokenize(string text);
    /// <summary>
    /// Encodes the text into a fixed-length sequence.
    /// </summary>
    /// <param name="sequenceLength">The desired sequence length.</param>
    /// <param name="text">The text to encode.</param>
    /// <returns>An enumerable of encoded tokens.</returns>
    IEnumerable<EncodedToken> Encode(int sequenceLength, string text);

}
