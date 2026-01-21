namespace OoBDev.AllMiniLmL6V2Sharp.Tokenizer;

/// <summary>
/// Represents a text token with its associated metadata.
/// </summary>
public class Token
{
    /// <summary>
    /// Initializes a new instance of the Token class.
    /// </summary>
    /// <param name="value">The token string value.</param>
    /// <param name="segmentIndex">The segment index for sentence pair tasks.</param>
    /// <param name="vocabularyIndex">The index in the vocabulary.</param>
    public Token(string value, long segmentIndex, long vocabularyIndex)
    {
        Value = value;
        VocabularyIndex = vocabularyIndex;
        SegmentIndex = segmentIndex;
    }

    /// <summary>
    /// Gets or sets the token string value.
    /// </summary>
    public string Value { get; set; } = string.Empty;
    /// <summary>
    /// Gets or sets the vocabulary index.
    /// </summary>
    public long VocabularyIndex { get; set; }
    /// <summary>
    /// Gets or sets the segment index.
    /// </summary>
    public long SegmentIndex { get; set; } 
}
