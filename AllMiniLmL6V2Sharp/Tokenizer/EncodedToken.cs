namespace OoBDev.AllMiniLmL6V2Sharp.Tokenizer;

/// <summary>
/// Represents an encoded token with BERT model inputs.
/// </summary>
public class EncodedToken
{
    /// <summary>
    /// Gets or sets the input ID (vocabulary index).
    /// </summary>
    public long InputIds { get; set; }
    /// <summary>
    /// Gets or sets the token type ID (segment index).
    /// </summary>
    public long TokenTypeIds { get; set; }
    /// <summary>
    /// Gets or sets the attention mask (1 for real tokens, 0 for padding).
    /// </summary>
    public long AttentionMask { get; set; }
}
